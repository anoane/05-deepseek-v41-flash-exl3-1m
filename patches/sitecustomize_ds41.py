# Runtime patches for DeepSeek-V4.1-Flash EXL3 on vLLM (sm_120). Mounted over the image's sitecustomize.py.
# Derived from the sfxnz recipe's docker/patch/sitecustomize.py (MIT); sm_120 additions are marked 'sm_120:'.

# install the apport exception handler if available
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

import sys

if "/opt/dsv41-patch" not in sys.path:
    sys.path.insert(0, "/opt/dsv41-patch")

# Load vLLM general plugins in every process (API, EngineCore, workers).
# VLLM_PLUGINS=vllm_exl3 is not enough on this image: EngineCore can resolve
# --quantization exl3 before load_general_plugins() runs.
try:
    from vllm.plugins import load_general_plugins

    load_general_plugins()
except Exception:
    pass

# DSV4.1 mapper picks weight_scale vs weight_scale_inv from
# quant_config.weight_block_size == [32, 32]. Exl3Config keeps that field
# inside non_routed_quantization, so copy it onto the config object.
try:
    from vllm_exl3.exl3 import Exl3Config

    _exl3_from_config = Exl3Config.from_config.__func__

    @classmethod
    def _exl3_from_config_with_block_size(cls, config):
        inst = _exl3_from_config(cls, config)
        nr = getattr(inst, "non_routed_quantization", None) or {}
        wbs = nr.get("weight_block_size") or config.get("weight_block_size")
        if wbs is not None:
            try:
                inst.weight_block_size = list(wbs)
            except AttributeError:
                pass  # vllm-exl3 >= 8f4517e8 exposes weight_block_size as a read-only property
        return inst

    Exl3Config.from_config = _exl3_from_config_with_block_size
except Exception:
    pass

# DSv4 sparse-MLA mixed warmup still dummy-forwards through DeepGEMM paged-MQA
# (block_kv must be 32 or 64) after autotune is disabled. Skip that warmup.
try:
    import vllm.model_executor.warmup.kernel_warmup as _kw

    _kw.deepseek_v4_sparse_mla_attention_warmup = lambda worker: None
    _kw.kernel_warmup = lambda worker: None
except Exception:
    pass

try:
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.worker.worker_base import CompilationTimes

    import os as _os
    # sm_120: the GB10 recipe no-op'd this to dodge a DeepGEMM warmup assert, but it also skips CUDA graph
    # capture (decode ran eager at ~31 tok/s). With the MXFP4 indexer that assert no longer fires.
    if _os.environ.get("DSV41_SKIP_WARMUP", "0") == "1":
        Worker.compile_or_warm_up_model = lambda self: CompilationTimes(0.0, 0.0)
except Exception:
    pass

# FlashInfer SM120 DSV4 decode is compiled only for page_block_size=64.
# Upstream V4.1 hardcodes SWA pages to 32 (DeepGEMM paged-MQA).
try:
    from sm120_page import coerce_swa_block_size
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

    _swa_init = DeepseekV4SWACache.__init__

    def _swa_init_sm120_page(self, *args, **kwargs):
        if "block_size" in kwargs:
            kwargs["block_size"] = coerce_swa_block_size(kwargs["block_size"])
        elif len(args) >= 7:
            args = list(args)
            args[6] = coerce_swa_block_size(args[6])
            args = tuple(args)
        return _swa_init(self, *args, **kwargs)

    DeepseekV4SWACache.__init__ = _swa_init_sm120_page
except Exception:
    pass

# Indexer + compressed MLA share a packed KV group, so they must agree.
# DeepGEMM paged-MQA asserts block_kv in {32, 64}; FlashInfer DSV4 decode
# wants page 64. Upstream reports 128 on SM12, which then has no common size
# if only the indexer is pinned to 64.
try:
    from sm120_page import indexer_kernel_block_sizes
    from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
        DeepseekV4FlashInferMLASparseBackend,
    )
    from vllm.v1.attention.backends.mla.indexer import DeepseekV4IndexerBackend

    _kbs = staticmethod(lambda: list(indexer_kernel_block_sizes()))
    DeepseekV4IndexerBackend.get_supported_kernel_block_sizes = _kbs
    DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes = _kbs
except Exception:
    pass

# --language-model-only still flattens vision_max_n_token onto hf_config, so
# SWA prefill index rows widen to window+1024=1152. SM120 DSV4 decode topk is
# {128,192,256,512,1024}. Do not zero vision_n_layers: VL checkpoints ship
# gate.bias_vl and load_weights KeyErrors without that param.
try:
    from sm120_page import text_only_max_image_tokens
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

    _v41_cfg_init = DeepseekV41Config.__init__

    def _v41_cfg_init_text_only(self, *args, **kwargs):
        _v41_cfg_init(self, *args, **kwargs)
        self.vision_max_n_token = text_only_max_image_tokens(
            getattr(self, "vision_max_n_token", 0), True
        )

    DeepseekV41Config.__init__ = _v41_cfg_init_text_only
except Exception:
    pass

# sm_120: allow the MXFP4 sparse-indexer K cache on SM120. vLLM gates it to sm_10x, but DeepGEMM's
# paged MQA logits accept FP4 at block_kv 32/64 on arch 12, and V4.1's ratio-2 compressed layers
# page at block_size/2 = 32 tokens, which the FP8 indexer path rejects on SM120 (needs 64).
try:
    import vllm.v1.attention.backends.mla.indexer as _dsa_idx

    def _dsa_indexer_uses_fp4_sm120(vllm_config):
        kv = vllm_config.attention_config.resolve_indexer_kv_dtype("fp8")
        if kv not in _dsa_idx.DSA_INDEXER_KV_DTYPES:
            raise ValueError(f"indexer_kv_dtype={kv!r} is not supported by the DeepSeek sparse indexer")
        return kv == "mxfp4"

    _dsa_idx.dsa_indexer_uses_fp4 = _dsa_indexer_uses_fp4_sm120
    for _m in ("vllm.models.deepseek_v4_1.attention", "vllm.models.deepseek_v4.attention"):
        try:
            __import__(_m, fromlist=["_"]).dsa_indexer_uses_fp4 = _dsa_indexer_uses_fp4_sm120
        except Exception:
            pass
except Exception:
    pass

# sm_120: optional custom exl3_moe kernel (XMOE_DIR/<XMOE_EXT>, built by kernels/build.sh), e.g. XMOE_EXT=xmoe.
# The plugin resolves exllamav3_ext.exl3_moe at call time, so rebinding the module attribute is enough.
try:
    import os as _os3
    _xm = _os3.environ.get("XMOE_EXT")
    if _xm:
        import sys as _sys3, importlib as _il3, torch as _torch3  # torch first: the extension links libtorch
        _p = _os3.path.join(_os3.environ.get("XMOE_DIR", "/opt/xmoe/build"), _xm)
        if _p not in _sys3.path:
            _sys3.path.insert(0, _p)
        _xmod = _il3.import_module(_xm)
        import exllamav3_ext as _ev3
        _ev3.exl3_moe = _xmod.exl3_moe
        _ev3.exl3_moe_max_concurrency = _xmod.exl3_moe_max_concurrency
except Exception as _e3:
    import sys as _sys4
    print(f"[sitecustomize] XMOE_EXT hook failed: {_e3!r}", file=_sys4.stderr)

# sm_120: hybrid dense MXFP8 (DENSE_HYBRID_M=<rows>, needs --linear-backend marlin). Marlin (W8A16) wins at decode row
# counts, FlashInfer CUTLASS (W8A8, FP8 tensor cores) at prefill row counts (dense was 9% of prefill on FlashInfer vs
# 20% on Marlin). Keep both layouts and route per call by rows. Compilation mode is NONE here, so the branch runs at
# call time: decode CUDA graphs (<=48 rows) capture Marlin, eager prefill takes FlashInfer. Costs one extra FP8 copy
# of the dense weights (~3.3 GiB/rank on V4.1): lower KV_MEM to compensate.
try:
    import os as _os5
    _hm = int(_os5.environ.get("DENSE_HYBRID_M", "0") or 0)
    if _hm > 0:
        import torch as _t5
        from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel as _MK5
        _mk5_pwal = _MK5.process_weights_after_loading
        _mk5_apply = _MK5.apply_weights
        _hyb5_n = [0, 0]

        def _hyb5_pwal(self, layer):
            from vllm.model_executor.layers.quantization.utils.mxfp8_utils import MXFP8_BLOCK_SIZE, swizzle_mxfp8_scale
            w = layer.weight.data
            fi = None
            if w.dim() == 2 and w.dtype == _t5.float8_e4m3fn:
                N, K = w.shape
                if K % MXFP8_BLOCK_SIZE == 0 and K >= 128 and N >= 128:
                    s2d = layer.weight_scale.data[:N, :K // MXFP8_BLOCK_SIZE].contiguous()
                    b = getattr(layer, "bias", None)
                    fi = (w.contiguous(), swizzle_mxfp8_scale(s2d, M=N, K=K).contiguous(), N, K,
                          None if b is None else b.data.clone())
            _mk5_pwal(self, layer)  # rebinds layer.weight/weight_scale/bias to Marlin layouts; fi keeps the originals
            layer._hyb5_fi = fi
            _hyb5_n[0 if fi is not None else 1] += 1
            if sum(_hyb5_n) in (1, 50, 100, 200, 400):
                print(f"[sitecustomize] DENSE_HYBRID_M={_hm}: {_hyb5_n[0]} layers hybrid, {_hyb5_n[1]} Marlin-only", file=sys.stderr)

        def _hyb5_apply(self, layer, x, bias=None):
            fi = getattr(layer, "_hyb5_fi", None)
            if fi is not None and x.numel() // x.shape[-1] > _hm:
                from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
                from vllm.utils import flashinfer as _vfi5
                w, s, N, K, b0 = fi
                xs = x.shape
                xq, xsc = mxfp8_e4m3_quantize(x.reshape(-1, K).contiguous(), is_sf_swizzled_layout=True)
                out = _vfi5.mm_mxfp8(xq, w.t(), xsc, s, out_dtype=x.dtype, backend="cutlass")
                if bias is not None:
                    out = out + (b0 if b0 is not None else bias)
                return out.view(*xs[:-1], N)
            return _mk5_apply(self, layer, x, bias)

        _MK5.process_weights_after_loading = _hyb5_pwal
        _MK5.apply_weights = _hyb5_apply
except Exception as _e5:
    print(f"[sitecustomize] DENSE_HYBRID hook failed: {_e5!r}", file=sys.stderr)
"""Serve the EXL3-quantized lm_head. TWO changes that only work together.

PROVEN CHAIN:
 1. pack ships head.{trellis,suh,svh,mul1}; tensor_storage.head says
    quant_format="exl3", bits_per_weight=6  -> the head IS exl3-quantized.
 2. quantization_config.json has NO non_routed_exl3 and non_routed_quantization=null,
    so Exl3Config.non_routed_exl3 == {} -> _matches_non_routed_exl3() is False for every
    prefix (exl3.py:1523-4) -> ParallelLMHead returns None (exl3.py:1698) -> head served
    UNQUANTIZED, expecting a single .weight.
 3. the VL mapper only has suffix rule "head.weight" -> "language_model.lm_head.weight",
    which never fires for our four names -> AutoWeightsLoader:
       ValueError: There is no module or parameter named 'head'
 diffbot's 2.0bpw pack ships an UNQUANTIZED head.weight, which is why they never hit this.

FIX A: inject non_routed_exl3 = {"modules": ["lm_head"], "bits": 6, "codebook": "mul1"}
        _prefix_has_suffix('language_model.lm_head','lm_head') is True (live-tested).
FIX B: add prefix rule "head." -> "language_model.lm_head." so all four tensors map.
Neither works alone.
"""
import sys

# ---- FIX A: declare the head as non-routed exl3 -----------------------------
try:
    from vllm_exl3.exl3 import Exl3Config

    _orig_from_config = Exl3Config.from_config.__func__

    @classmethod
    def _from_config_with_head(cls, config):
        cfg = dict(config)
        if not cfg.get("non_routed_exl3"):
            hb = int(cfg.get("head_bits", 6))
            cb = str(cfg.get("codebook", "mul1"))
            cfg["non_routed_exl3"] = {
                "modules": ["lm_head"],
                "bits": hb,
                "codebook": cb,
            }
            print("[exl3_head_patch] injected non_routed_exl3 modules=['lm_head'] "
                  "bits=%d codebook=%s" % (hb, cb), file=sys.stderr)
        return _orig_from_config(cls, cfg)

    Exl3Config.from_config = _from_config_with_head
except Exception as _e:
    print("[exl3_head_patch] FIX A FAILED: %r" % (_e,), file=sys.stderr)

# ---- FIX B: map head.* onto language_model.lm_head.* ------------------------
try:
    from vllm.models.deepseek_v4_1.nvidia import vl_model as _vl

    _orig_mk = _vl._make_deepseek_v4_vl_weights_mapper

    def _mk_with_exl3_head(expert_dtype, linear_scale_name):
        wm = _orig_mk(expert_dtype, linear_scale_name)
        pref = dict(getattr(wm, "orig_to_new_prefix", {}) or {})
        if "head." not in pref:
            pref["head."] = "language_model.lm_head."
            try:
                wm.orig_to_new_prefix = pref
            except AttributeError:
                import dataclasses
                wm = dataclasses.replace(wm, orig_to_new_prefix=pref)
            print("[exl3_head_patch] mapper: head. -> language_model.lm_head.", file=sys.stderr)
        return wm

    _vl._make_deepseek_v4_vl_weights_mapper = _mk_with_exl3_head
except Exception as _e:
    print("[exl3_head_patch] FIX B FAILED: %r" % (_e,), file=sys.stderr)


# =============================================================================
# P9: serve the Mia-AiLab 3.0bpw pack's 622 NON-ROUTED exl3 modules.
#
# The pack exl3-quantizes everything (attention, shared experts, indexer,
# compressor, engram wkv, head) but declares no `non_routed_exl3`, so without
# this the modules are served unquantized and expect a bare `.weight`.
#
# FIX C: three constructors hardcode quant_config=None but the pack DID
#        quantize them -> inject the live quant_config by prefix.
# FIX D: attn.wo_a is a block-diagonal grouped BMM (o_groups=8, in=4096,
#        out=8*1024) stored as 8 independent exl3 `slice.N` linears. exl3's
#        merged-shard support assumes a SHARED input, so it cannot represent
#        this. deep_gemm_fp8_o_proj already has a dense-bf16 branch
#        (use_fp8 = wo_a.weight.dtype == float8_e4m3fn), so reconstruct the 8
#        slices to one dense bf16 [8192,4096] weight at load time.
#        Cost ~2.68 GiB vs ~0.84 GiB packed. No new kernel, no _o_proj patch.
# =============================================================================

_P9_NON_ROUTED_MODULES = [
    "fused_wqa_wkv",                # attn.wq_a + attn.wkv            (2 shards)
    "wq_b",                         # attn.wq_b AND indexer.wq_b
    "wo_b",
    "shared_experts.gate_up_proj",  # shared_experts.w1 + w3          (2 shards)
    "shared_experts.down_proj",     # shared_experts.w2
    "indexer.wk",
    # FIX I: "compressor.fused_wkv_wgate" REMOVED. Declaring it made the module exl3, which
    # loaded fine but replaced its `.weight` -- and attention.py:777 (compressor_kv_score)
    # reaches into `compressor.fused_wkv_wgate.weight.T` DIRECTLY for a raw matmul, bypassing
    # forward() and the quant method. A sweep of the whole deepseek_v4_1 tree found this is the
    # ONLY such direct access on the nvidia path (amd/rocm.py:565,652 are ROCm-only; the
    # "head.weight" hits are WeightsMapper rename strings). So the module stays dense and is
    # filled by the load-time dequant in _p9_transform, exactly as wo_a is.
    "engram.wkv",
    "lm_head",
]
# wo_a is DELIBERATELY ABSENT -> stays a dense bf16 weight, filled by FIX D.
_P9_BITS = 5                        # verified: 621 of 622 modules are K=5
_P9_LAYER_BITS = {"lm_head": 6}     # verified: head trellis [320,8080,96] -> K=6

# ---- FIX A': declare ALL 622 non-routed modules, not just the head ---------
# This wraps Exl3Config.from_config a SECOND time, outside the head-only
# wrapper in the block above. Ours runs first and sets non_routed_exl3 with the
# full list; the inner head-only wrapper then sees it already set
# (`if not cfg.get("non_routed_exl3")`) and correctly does nothing.
try:
    from vllm_exl3.exl3 import Exl3Config as _p9_Exl3Config

    _p9_prev_from_config = _p9_Exl3Config.from_config.__func__

    @classmethod
    def _p9_from_config(cls, config):
        cfg = dict(config)
        if not cfg.get("non_routed_exl3"):
            cfg["non_routed_exl3"] = {
                "modules": list(_P9_NON_ROUTED_MODULES),
                "bits": _P9_BITS,
                "layer_bits": dict(_P9_LAYER_BITS),
                "codebook": str(cfg.get("codebook", "mul1")),
            }
            print("[p9] non_routed_exl3: %d suffixes, bits=%d, layer_bits=%r"
                  % (len(_P9_NON_ROUTED_MODULES), _P9_BITS, _P9_LAYER_BITS),
                  file=sys.stderr)
        return _p9_prev_from_config(cls, cfg)

    _p9_Exl3Config.from_config = _p9_from_config
    print("[p9] FIX A' armed (full non-routed module list)", file=sys.stderr)
except Exception as _e:
    print("[p9] FIX A' FAILED: %r" % (_e,), file=sys.stderr)

# ---- FIX C: inject quant_config where the constructor hardcodes None --------
try:
    import vllm.model_executor.layers.linear as _p9_L
    from vllm.config import get_current_vllm_config as _p9_gcvc

    # FIX I: compressor.fused_wkv_wgate removed — attention.py:777 reads its `.weight`
    # directly, so it must stay dense. indexer.wk is safe: the tree sweep found no direct
    # `.weight` access on it anywhere on the nvidia path.
    _P9_NEEDS_QC = ("indexer.wk",)

    def _p9_patch_linear(cls, label):
        _orig = cls.__init__

        def __init__(self, *a, **kw):
            if kw.get("quant_config") is None:
                pfx = kw.get("prefix", "") or ""
                if any(pfx == s or pfx.endswith("." + s) for s in _P9_NEEDS_QC):
                    try:
                        qc = _p9_gcvc().quant_config
                    except Exception:
                        qc = None
                    if qc is not None:
                        kw["quant_config"] = qc
                        print("[p9] quant_config -> %s %s" % (label, pfx),
                              file=sys.stderr)
            return _orig(self, *a, **kw)

        cls.__init__ = __init__

    _p9_patch_linear(_p9_L.MergedColumnParallelLinear, "MergedColumnParallelLinear")
    _p9_patch_linear(_p9_L.ReplicatedLinear, "ReplicatedLinear")
    print("[p9] FIX C armed (indexer.wk only; compressor handled by FIX I)",
          file=sys.stderr)
except Exception as _e:
    print("[p9] FIX C FAILED: %r" % (_e,), file=sys.stderr)

# ---- FIX D: reconstruct attn.wo_a from its 8 exl3 slices -------------------
try:
    import re as _p9_re
    import torch as _p9_torch
    from vllm.models.deepseek_v4_1.nvidia import model as _p9_dsm
    from vllm.model_executor.models.utils import (
        is_pp_missing_parameter as _p9_is_pp_missing,
    )

    _P9_N_SLICES = 8
    _P9_WO_A_RE = _p9_re.compile(
        r"^(?P<base>(?:.*\.)?layers\.\d+\.attn\.wo_a)"
        r"\.slice\.(?P<si>\d+)\.(?P<leaf>trellis|suh|svh|mul1|mcg)$"
    )

    def _p9_marker(t):
        # exl3 codebook markers: pass the tensor through only when non-zero.
        if t is None:
            return None
        try:
            if int(t.reshape(-1)[0].item()) == 0:
                return None
        except Exception:
            return None
        return t

    def _p9_dense_from_slices(group):
        """8 exl3 slices (in=4096, out=1024) -> dense [8192, 4096]."""
        from vllm_exl3.exl3 import make_linear_exl3

        dev = _p9_torch.device("cuda", _p9_torch.cuda.current_device())
        rows = []
        for i in range(_P9_N_SLICES):
            d = group[i]
            lin = make_linear_exl3(
                d["trellis"].to(dev),
                d["suh"].to(dev),
                d["svh"].to(dev),
                _p9_marker(d.get("mcg")),
                _p9_marker(d.get("mul1")),
                out_dtype=_p9_torch.float16,
            )
            # get_weight_tensor() -> (in_features, out_features) = (4096, 1024)
            w = lin.get_weight_tensor()
            rows.append(w.t().contiguous())   # -> (1024, 4096)
            del lin, w
        out = _p9_torch.cat(rows, dim=0)      # -> (8192, 4096), group-major
        del rows
        return out

    # FIX E: drop the SUPERSEDED fp8 engram.wkv pair.
    # The index lists only the pack's exl3 engram.wkv.{trellis,suh,svh,mul1},
    # but it points the two engram.embed keys at base-repo shards 47/48 — and
    # the loader uses the index only to choose FILES, then yields every key in
    # them. Those shards also carry fp8 `engram.wkv.weight [25600,6144]` and
    # `engram.wkv.scale [800,192]`. weight_utils.py:969 skips only
    # .engram.embed.weight/.scale, so the wkv pair leaks through; the mapper
    # rewrites `.scale` -> `.weight_scale_inv` and params_dict has no such
    # entry (model.py:813) -> KeyError. The exl3 version supersedes it.
    _P9_DROP_RE = _p9_re.compile(
        r"^(?:.*\.)?layers\.\d+\.engram\.wkv\."
        r"(?:weight|scale|weight_scale|weight_scale_inv)$"
    )

    # FIX I: attn.compressor.{wkv,wgate} -> one dense fused_wkv_wgate.weight.
    # Same shape of problem as wo_a, and the same remedy. Geometry read from the pack:
    #   wkv/wgate  suh[5120] svh[512] trellis[320,32,80]  -> in 5120, out 512, K=5
    # MergedColumnParallelLinear(5120, [head_dim, head_dim]) wants weight [1024, 5120] with
    # the wkv rows first (stacked_params_mapping maps wkv->shard 0, wgate->shard 1).
    # Layer 20 has compress_ratio == 1 -> has_gate False -> wkv ONLY -> [512, 5120]. Verified:
    # layers.20.attn.compressor.wgate is genuinely absent from the pack. Because absence is
    # only knowable once the key stream ends, compressor groups are flushed at end-of-stream.
    _P9_COMPRESSOR_RE = _p9_re.compile(
        r"^(?P<base>(?:.*\.)?layers\.\d+\.attn\.compressor)"
        r"\.(?P<proj>wkv|wgate)\.(?P<leaf>trellis|suh|svh|mul1|mcg)$"
    )

    def _p9_dense_one(d):
        """One exl3 group -> dense (out_features, in_features), i.e. Linear.weight order."""
        from vllm_exl3.exl3 import make_linear_exl3

        dev = _p9_torch.device("cuda", _p9_torch.cuda.current_device())
        lin = make_linear_exl3(
            d["trellis"].to(dev),
            d["suh"].to(dev),
            d["svh"].to(dev),
            _p9_marker(d.get("mcg")),
            _p9_marker(d.get("mul1")),
            out_dtype=_p9_torch.float16,
        )
        w = lin.get_weight_tensor().t().contiguous()   # (in,out) -> (out,in)
        del lin
        return w

    def _p9_transform(model, weights):
        buf = {}
        cbuf = {}
        done = 0
        comp = 0
        skipped = 0
        dropped = 0
        for name, w in weights:
            if _P9_DROP_RE.match(name):
                dropped += 1
                continue
            mc = _P9_COMPRESSOR_RE.match(name)
            if mc is not None:
                cbase = mc.group("base")
                ctarget = cbase + ".fused_wkv_wgate.weight"
                if _p9_is_pp_missing(ctarget, model):
                    skipped += 1
                    continue
                cbuf.setdefault(cbase, {}).setdefault(
                    mc.group("proj"), {})[mc.group("leaf")] = w
                continue
            m = _P9_WO_A_RE.match(name)
            if m is None:
                yield name, w
                continue
            base = m.group("base")
            target = base + ".weight"
            # Under PP every rank sees every name. Only the owning rank should
            # reconstruct: the CMP (sm_80) cannot run exl3 kernels at all.
            if _p9_is_pp_missing(target, model):
                skipped += 1
                continue
            g = buf.setdefault(base, {})
            g.setdefault(int(m.group("si")), {})[m.group("leaf")] = w
            if len(g) == _P9_N_SLICES and all(
                all(k in d for k in ("trellis", "suh", "svh")) for d in g.values()
            ):
                dense = _p9_dense_from_slices(g)
                buf.pop(base, None)
                done += 1
                yield target, dense.to(_p9_torch.bfloat16)
        # FIX I: flush compressor groups at END OF STREAM. A missing `wgate` is legitimate
        # (layer 20 has compress_ratio == 1 -> has_gate False, and the pack genuinely has no
        # layers.20.attn.compressor.wgate), and that absence is only knowable once the key
        # stream is exhausted — hence the deferred flush rather than an eager one.
        for cbase in sorted(cbuf):
            parts = []
            for proj in ("wkv", "wgate"):   # order = MergedColumnParallelLinear shard order
                d = cbuf[cbase].get(proj)
                if d is None:
                    continue
                if not all(k in d for k in ("trellis", "suh", "svh")):
                    raise RuntimeError(
                        "[p9] incomplete compressor group %s.%s: %r"
                        % (cbase, proj, sorted(d))
                    )
                parts.append(_p9_dense_one(d))
            if not parts:
                continue
            dense = parts[0] if len(parts) == 1 else _p9_torch.cat(parts, dim=0)
            comp += 1
            yield cbase + ".fused_wkv_wgate.weight", dense.to(_p9_torch.bfloat16)
        if buf:
            raise RuntimeError(
                "[p9] incomplete wo_a slice groups: %r"
                % ({k: sorted(v) for k, v in buf.items()},)
            )
        print("[p9] wo_a reconstructed: %d layers; compressor fused: %d; "
              "%d tensors not owned by this rank; dropped %d superseded fp8 engram.wkv"
              % (done, comp, skipped, dropped), file=sys.stderr)

    _p9_orig_load_weights = _p9_dsm.DeepseekV4Model.load_weights

    def _p9_load_weights(self, weights):
        return _p9_orig_load_weights(self, _p9_transform(self, weights))

    _p9_dsm.DeepseekV4Model.load_weights = _p9_load_weights
    print("[p9] FIX D armed (wo_a: 8 exl3 slices -> dense bf16 [8192,4096])",
          file=sys.stderr)
except Exception as _e:
    print("[p9] FIX D FAILED: %r" % (_e,), file=sys.stderr)

# ---- FIX F: rank-aware offload budget --------------------------------------
# --cpu-offload-gb is PER RANK: PP=2 x 50 GiB authorised ~100 GiB of pinned host
# RAM on a 141 GB box and the OOM killer took the worker (P9b, 01:50:47, avail
# hit 1 G). Worse, rank 1 IS the CMP 170HX, whose entire role in this topology is
# to hold its layers RESIDENT because its PCIe link is Gen2 x1 (~0.38 GB/s H2D).
# Offloading from the CMP to host RAM inverts the design.
#
# Zeroing is done LAZILY in wrap_modules rather than __init__: the offloader is
# built in GPUModelRunner.__init__ (model_runner.py:349) and I did not want to
# depend on whether the PP group is initialized that early. wrap_modules runs
# during weight loading, well after distributed init. It is also idempotent --
# uva.py notes wrap_modules may be called more than once.
try:
    from vllm.model_executor.offloader.uva import UVAOffloader as _p9_UVA

    _p9_uva_orig_wrap = _p9_UVA.wrap_modules

    def _p9_uva_wrap(self, modules_generator, prefix=""):
        try:
            from vllm.distributed.parallel_state import get_pp_group
            pp_rank = int(get_pp_group().rank_in_group)
        except Exception as _re:
            pp_rank = 0
            print("[p9] FIX F: PP rank unavailable (%r) -> budget left alone"
                  % (_re,), file=sys.stderr)
        if pp_rank != 0 and self.cpu_offload_max_bytes:
            print("[p9] FIX F: PP rank %d is the RESIDENT tier -> offload budget "
                  "%.2f GiB -> 0"
                  % (pp_rank, self.cpu_offload_max_bytes / (1024 ** 3)),
                  file=sys.stderr)
            self.cpu_offload_max_bytes = 0
        return _p9_uva_orig_wrap(self, modules_generator, prefix)

    _p9_UVA.wrap_modules = _p9_uva_wrap
    print("[p9] FIX F armed (offload budget zeroed on non-zero PP ranks)",
          file=sys.stderr)
except Exception as _e:
    print("[p9] FIX F FAILED: %r" % (_e,), file=sys.stderr)

# ---- FIX G: instrument the offloader. MEASURE, do not infer. ---------------
# Three single-GPU runs at budgets 110/115/116 all ran host RAM to exhaustion, and
# `Total CPU offloaded parameters` has NEVER printed (wrap_modules never finishes), so the
# budget has never been OBSERVED to be honoured. I have already been wrong twice about this
# subsystem by inferring from partial trajectories (P3c, and the withdrawn overhead model in
# P12). This prints the offloader's own accounting next to the kernel's Shmem figure.
#
# The `gap` column is the whole point: Shmem minus the bytes the offloader thinks it moved.
#   gap flat + small  -> budget IS honoured; the overshoot lives outside the pinned pool
#                        (e.g. weights that cannot fit the GPU spilling to host).
#   gap grows         -> the offloader's accounting misses real allocations
#                        (uva.py makes TWO host copies per param: to("cpu") then pin_memory()).
try:
    from vllm.model_executor.offloader.uva import UVAOffloader as _p9_UVA_I

    _p9_orig_maybe = _p9_UVA_I._maybe_offload_to_cpu
    _p9_spy = {"n": 0}

    def _p9_meminfo():
        sh = av = -1.0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("Shmem:"):
                        sh = int(line.split()[1]) / 1048576.0
                    elif line.startswith("MemAvailable:"):
                        av = int(line.split()[1]) / 1048576.0
        except Exception:
            pass
        return sh, av

    def _p9_maybe(self, module, prefix=""):
        out = _p9_orig_maybe(self, module, prefix)
        _p9_spy["n"] += 1
        n = _p9_spy["n"]
        if n <= 3 or n % 5 == 0:
            sh, av = _p9_meminfo()
            off = self.cpu_offload_bytes / (1024 ** 3)
            mx = self.cpu_offload_max_bytes / (1024 ** 3)
            print("[spy] mod=%d offloaded=%.2f/%.2f GiB | Shmem=%.1f MemAvail=%.1f "
                  "| gap=%.2f GiB" % (n, off, mx, sh, av, sh - off),
                  file=sys.stderr, flush=True)
        return out

    _p9_UVA_I._maybe_offload_to_cpu = _p9_maybe
    print("[p9] FIX G armed (offloader spy: offloaded vs Shmem, gap column)",
          file=sys.stderr)
except Exception as _e:
    print("[p9] FIX G FAILED: %r" % (_e,), file=sys.stderr)

# ---- FIX H: ParallelLMHead must receive quant_config -----------------------
# P17 got past the memory wall and died on the LAST module:
#   ValueError: There is no module or parameter named 'lm_head.mul1' ...
#   available parameters belonging to lm_head (ParallelLMHead) are: {'lm_head.weight'}
# The head was served UNQUANTIZED despite FIX A' declaring "lm_head", because
# model.py:1028-1032 builds it with NO quant_config at all:
#     self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size,
#                                   prefix=maybe_prefix(prefix, "lm_head"))
# and exl3.py:1683-1686 notes vLLM only consults quant_config for embedding-family
# layers when the model passes it. So get_quant_method is never called and the
# declaration, though correct, is unreachable. Same class as FIX C.
#
# The pack's head is exl3 K=6: trellis [320, 8080, 96] -> in 5120 (320*16),
# out 129280 (8080*16), 96/16 = 6 bits. Both dims are 128-divisible, which exl3 requires.
# The error names it `lm_head.mul1` (not `language_model.lm_head.mul1`), so inside the inner
# LLM module the prefix is plain `lm_head`; match both forms.
try:
    import inspect as _p9_inspect
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead as _P9_PLH,
    )
    from vllm.config import get_current_vllm_config as _p9_gcvc2

    _p9_plh_orig_init = _P9_PLH.__init__
    try:
        _p9_plh_accepts_qc = "quant_config" in _p9_inspect.signature(
            _p9_plh_orig_init
        ).parameters
    except Exception:
        _p9_plh_accepts_qc = False

    def _p9_plh_init(self, *a, **kw):
        if _p9_plh_accepts_qc and kw.get("quant_config") is None:
            pfx = kw.get("prefix", "") or ""
            if pfx == "lm_head" or pfx.endswith(".lm_head"):
                try:
                    qc = _p9_gcvc2().quant_config
                except Exception:
                    qc = None
                if qc is not None:
                    kw["quant_config"] = qc
                    print("[p9] FIX H: quant_config -> ParallelLMHead %r" % (pfx,),
                          file=sys.stderr)
        return _p9_plh_orig_init(self, *a, **kw)

    _P9_PLH.__init__ = _p9_plh_init
    print("[p9] FIX H armed (ParallelLMHead accepts_quant_config=%s)"
          % (_p9_plh_accepts_qc,), file=sys.stderr)
except Exception as _e:
    print("[p9] FIX H FAILED: %r" % (_e,), file=sys.stderr)

# ---- FIX J: the CMP 170HX as an EXPERT RESIDENCY TIER ----------------------
# [FINDING] established the CMP cannot host a LAYER: v4.1's last kv-sharing group is an
# indivisible 20 layers, and there is no sm_80 v4.1 ATTENTION (flash_mla absent; FlashInfer
# sparse-MLA gates are sm90/sm120). All of that stands. But EXPERTS ARE NOT ATTENTION —
# routed experts are exl3 GEMMs, and those were PROVED to run on sm_80:
#     mean_rel_err 0.00057 ; 0.05 ms @ rows=1 => ~527 GB/s effective weight read
# versus ~42 GB/s streaming the same weights from host RAM over the Blackwell's Gen5 link.
# The only new cost is activations over the CMP's Gen2 x1 link, measured at 0.082 ms/token
# round trip -> ~0.98 ms/token across 12 layers, ~1% of a 77 ms/token decode.
# This is the project's own rule applied literally: ACTIVATIONS over slow links, never WEIGHTS.
#
# Sizing from the pack (measured, not assumed): 40 MoE layers x 384 experts x 4.762 GiB.
# CMP has 63.4 GiB, so ~12 layers (57.1 GiB) fit with ~6 GiB of headroom.
#
# TWO HOOKS ONLY:
#  J1 create_weights -> move the 10 MoE params to the CMP while they are still EMPTY, so the
#     weight loader copies CPU->cuda:1 directly and they never occupy Blackwell VRAM. This must
#     happen at create_weights, NOT process_weights_after_loading: the offloader runs during
#     make_layers (before weights load), so a later move would let these 57 GiB land in host RAM
#     first. Device binding is safe -- build_exl3_fused_state:707 takes
#     `device = layer.w13_trellis.device` from the TENSOR, not torch.cuda.current_device(),
#     so the pointer tables and fused temps follow the params. pin_exl3_expert_map re-pins on a
#     device change (its cache check includes `cached.device == device`).
#  J3 apply -> marshal activations to the CMP and the result back.
#
# The host offloader needs NO patch: setting uva.py's own `_vllm_is_uva_offloaded` marker on the
# moved params makes its existing skip ("parameters an earlier wrap_modules call already
# offloaded") pass over them. A per-module skip would have been WRONG -- a decoder layer holds
# attention on cuda:0 and experts on cuda:1, and _maybe_offload_to_cpu is called per layer.
#
# Disabled unless DS41_CMP_DEVICE is set, so it is inert in every existing configuration.
try:
    import os as _p9_os
    import re as _p9_re_j
    import torch as _p9_torch_j

    _P9_CMP_DEVICE = _p9_os.environ.get("DS41_CMP_DEVICE", "").strip()
    _P9_CMP_LAYERS = _p9_os.environ.get("DS41_CMP_LAYERS", "").strip()

    def _p9_parse_layers(spec):
        out = set()
        for part in str(spec).replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(part))
        return out

    _P9_CMP_SET = _p9_parse_layers(_P9_CMP_LAYERS) if _P9_CMP_DEVICE else set()
    _P9_MOE_PARAMS = ("w13_trellis", "w13_suh", "w13_svh", "w13_mcg", "w13_mul1",
                      "w2_trellis", "w2_suh", "w2_svh", "w2_mcg", "w2_mul1")
    _P9_LAYER_IDX_RE = _p9_re_j.compile(r"layers\.(\d+)\.")

    def _p9_layer_index(layer):
        s = str(getattr(layer, "layer_name", "") or getattr(layer, "prefix", "") or "")
        m = _P9_LAYER_IDX_RE.search(s)
        return int(m.group(1)) if m else None

    if _P9_CMP_SET:
        from vllm_exl3.exl3 import Exl3MoEMethod as _P9_MOE

        _p9_moe_orig_create = _P9_MOE.create_weights
        _p9_moe_orig_apply = _P9_MOE.apply
        _P9_CMP_PLACED = []

        def _p9_moe_create(self, layer, *a, **kw):
            _p9_moe_orig_create(self, layer, *a, **kw)
            li = _p9_layer_index(layer)
            if li is None or li not in _P9_CMP_SET:
                return
            moved = nbytes = 0
            for nm in _P9_MOE_PARAMS:
                p = getattr(layer, nm, None)
                if p is None:
                    continue
                p.data = p.data.to(_P9_CMP_DEVICE)
                # uva.py's own marker: makes its existing skip pass over these params
                # instead of UVA-mapping them into host RAM. No offloader patch needed.
                p._vllm_is_uva_offloaded = True
                moved += 1
                nbytes += p.data.numel() * p.data.element_size()
            layer._p9_cmp_device = _P9_CMP_DEVICE
            _P9_CMP_PLACED.append(li)
            print("[p9] FIX J: layer %d routed experts -> %s (%d params, %.2f GiB)"
                  % (li, _P9_CMP_DEVICE, moved, nbytes / (1024 ** 3)),
                  file=sys.stderr, flush=True)

        def _p9_moe_apply(self, layer, x, topk_weights, topk_ids, shared_experts,
                          shared_experts_input):
            dst = getattr(layer, "_p9_cmp_device", None)
            if dst is None:
                return _p9_moe_orig_apply(self, layer, x, topk_weights, topk_ids,
                                          shared_experts, shared_experts_input)
            src = x.device
            out = _p9_moe_orig_apply(
                self, layer,
                x.to(dst, non_blocking=True),
                topk_weights.to(dst, non_blocking=True),
                topk_ids.to(dst, non_blocking=True),
                shared_experts, shared_experts_input)
            return out.to(src, non_blocking=True)

        _P9_MOE.create_weights = _p9_moe_create
        _P9_MOE.apply = _p9_moe_apply
        print("[p9] FIX J armed: %d layers -> %s (%s)"
              % (len(_P9_CMP_SET), _P9_CMP_DEVICE,
                 ",".join(str(i) for i in sorted(_P9_CMP_SET))), file=sys.stderr)
    else:
        print("[p9] FIX J idle (set DS41_CMP_DEVICE + DS41_CMP_LAYERS to enable)",
              file=sys.stderr)
except Exception as _e:
    print("[p9] FIX J FAILED: %r" % (_e,), file=sys.stderr)
