# DeepSeek-V4.1-Flash EXL3 3.0bpw at 1M context

A **204 GiB** decoder served at **1,048,576 tokens** of context on one 96 GB card, ~60 GiB of
DDR5, an engram streamed from SSD, and a mining GPU holding twelve layers of experts.

```
RTX PRO 6000 (96 GB, sm_120)  attention + 28 layers of experts + KV
CMP 170HX    (64 GB, sm_80)   routed experts for layers 28-39, fully resident
host DDR5                     60.3 GiB of expert weights, UVA-mapped, unpinned
SSD                           189 GiB of engram tables, never resident
```

| | |
|---|---|
| Context | **1,048,576** — KV holds 1,797,989 tokens (1.71×) |
| Decode | ~14 tok/s at short context |
| Prefill | **335 tok/s** (142,278 tokens in 424.7 s) |
| Load | 420 s (57 GiB crosses a Gen2 ×1 link) |
| Patches | 10, none in the model's forward path |

---

### Hardware this was built and measured on

| | |
|---|---|
| Host | Proxmox VE 9.2.x, kernel 7.0.x-pve, Secure Boot **disabled** |
| CPU | AMD Ryzen 9 9950X3D (16C/32T) — 24 vCPU passed to the guest |
| RAM | 160 GiB DDR5 allocated to the guest (157 GiB usable) — **no swap, deliberately** |
| GPU 0 | NVIDIA RTX PRO 6000 Blackwell Workstation — 97,887 MiB, `sm_120`, `10de:2bb1`, PCIe Gen5 x16 (~42 GB/s H2D measured), 400 W default limit / 600 W max |
| GPU 1 | NVIDIA CMP 170HX — 65,536 MiB **after unlock** (8 GB stock), `sm_80` (GA100), `10de:20c2`, PCIe **Gen2 x1 (~0.38 GB/s)**, 200 W default limit / 250 W max |
| Guest | Ubuntu 24.04.4 LTS, kernel 6.8.0-139-generic, NVIDIA driver 610.43.02 |
| Storage | 5.8 TB NVMe (~93 GB free with all packs resident) |

**This recipe uses every tier in the table**: GPU 0 for attention + KV + 28 expert layers,
GPU 1 for 12 fully-resident expert layers, ~60 GiB of DDR5 for streamed experts, and the NVMe
for ~189 GiB of engram tables that are never resident. Free disk is the binding constraint.

> **This is a point-in-time recipe and may be slightly outdated or incomplete.**
> It was transcribed from a working system rather than written as a clean-room guide: driver,
> engine and image versions move quickly, some steps that were obvious in the moment are
> under-documented, and a few numbers were measured once rather than averaged. Every measured
> figure below is specific to the hardware in the table above — on a different PCIe topology,
> a different RAM size, or a card without the CMP's x1 bottleneck, the tuning will differ.
> Read it as a worked example with its reasoning shown, not as a turnkey script.

## The two measurements that decided everything

Both were *estimated* first, and both estimates were wrong by enough to make the model look
impossible on this hardware.

### 1. Pinned host memory costs 26.5% more than the budget says

`uva.py` allocates **twice** per parameter — `p.data.to("cpu")` then `cpu_data.pin_memory()` —
but counts only the final tensor. Measured, with the offloader's own counter beside
`/proc/meminfo`:

```
mod=10  offloaded= 38.10 GiB   Shmem= 48.2   ratio 1.265
mod=25  offloaded=109.53 GiB   Shmem=109.6   ratio 1.001   <- pinning disabled
```

```
host_bytes = cpu_offload_gb x 1.265    with pinning (default)
host_bytes = cpu_offload_gb x 1.000    with VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
```

**Set that env var or nothing fits.** Six runs died before this was instrumented. On CUDA,
`get_accelerator_view_from_cpu_tensor` does not require pinned memory (only the XPU branch
does), so UVA still works from pageable memory.

Also: `--cpu-offload-gb` is **per rank**, and it is a **floor** on host use, not a ceiling — the
offloader pins its way up to the budget before the GPU is full, so overshooting wastes RAM
*and* starves the GPU.

### 2. KV is ~1,792 B/token at 1M — measure in the regime you will run in

At `ctx 4096` the same server reports 35,320 B/token. That is **block-granularity floor noise**,
not a per-token cost, and a conclusion drawn from it had to be formally withdrawn. At a long
context it is ~1,792 B/token, so a full 1M window needs **~1.75 GiB**. Reserving 9 GiB caused a
CUDA OOM. With `--max-num-seqs 1` there is no reason to reserve concurrency headroom at all.

---

## The patch stack

The pack exl3-quantizes **622 non-routed modules** but declares no `non_routed_exl3`, so stock
vLLM serves them unquantized and they fail on contact. All ten fixes live in one bind-mounted
`sitecustomize.py` (`patches/sitecustomize_ds41.py`); **the model's own code is never edited**.

| fix | what |
|---|---|
| **A′** | declare the 622 modules as **8 suffixes** (`_prefix_has_suffix` matches by suffix), `bits=5`, `layer_bits={"lm_head":6}` |
| **C** | hand `quant_config` to `indexer.wk` — its constructor hardcodes `None` |
| **D** | rebuild `attn.wo_a` from 8 `slice.N` linears into a dense bf16 weight |
| **E** | drop the superseded fp8 `engram.wkv` pair leaked by the base-repo shards |
| **F** | make the offload budget rank-aware (zero it on non-zero PP ranks) |
| **G** | instrument the offloader — this is what found the 1.265× multiplier |
| **H** | hand `quant_config` to `ParallelLMHead`, or the exl3 head is unreachable |
| **I** | keep `compressor.fused_wkv_wgate` **dense** — `attention.py:777` reads `.weight.T` directly |
| **J** | put 12 layers of routed experts on the CMP |

### Two worth explaining

**D — `wo_a` is a block-diagonal grouped BMM** (8 groups, 4096→1024 each) stored as eight
independent `slice.N` linears. `.slice.` has **zero** handling anywhere in vLLM, and exl3's
merged-shard support assumes a *shared* input, so it cannot represent this. The fix leaves the
module undeclared — it keeps a dense bf16 `[8192,4096]` weight — and reconstructs the slices at
load, because `deep_gemm_fp8_o_proj` already contains a bf16 `torch.bmm` branch gated on
`wo_a.weight.dtype == float8_e4m3fn`. No new kernel, no change to attention. Costs ~1.8 GiB.

**I — a fix that broke the forward path.** Fix C originally covered the compressor too. That
made it load correctly and then crash, because `attention.py:777` bypasses the quant method and
reads `.weight.T` for a raw matmul. Lesson: **giving a module a `quant_config` changes its
parameter surface** — grep the tree for direct `.weight` access *before* shipping, not after.
A sweep found that line is the only such access on the CUDA path.

---

## The CMP as an expert tier

The obvious idea — give the CMP a pipeline stage — **is impossible**, for two independent
reasons:

- v4.1 shares compressed KV (`kv_source_layer_id = max(s ≤ layer)` over `[2,8,14,20]`) and a PP
  boundary may not split a group. The last group, **layers 20-39, is indivisible**: 20 layers
  ≈ 98 GiB against ~59 GiB usable.
- `_select_dsv4_attn_cls` returns the SM120 class only when `capability.major == 12`; everything
  else falls to FlashMLA, which **is not installed**, and FlashInfer's sparse-MLA gates are
  sm90/sm120 only.

**The blocker is attention, not the MoE GEMM.** Experts are exl3 GEMMs, and those run correctly
on `sm_80` — measured at `mean_rel_err 0.00057` and **527 GB/s** effective weight read, versus
~42 GB/s streaming the same weights from host. So fix **J** moves expert *weights* only;
attention, indexer, compressor, engram and head all stay on the Blackwell, and only
*activations* cross the ×1 link — 0.082 ms per token, round trip.

Two hooks, and one thing that needed no patch:

- `create_weights` — move the ten MoE params to `cuda:1` while still **empty**, so the loader
  writes CPU→CMP directly and they never occupy Blackwell VRAM.
- `apply` — marshal activations across and results back.
- The host offloader needed **no patch**: setting its own `_vllm_is_uva_offloaded` marker makes
  its existing skip pass over those params. A per-*module* skip would have been wrong — a
  decoder layer holds attention on one card and experts on the other.

### What it bought

| | without CMP | with CMP |
|---|---|---|
| Host offload | 117.45 GiB | **60.31 GiB** |
| RAM free | 15 G | **69 G** |
| Prefill | 311 tok/s | **335 tok/s** |
| Decode | 13.42 tok/s | 13.99-14.83 (noisy) |

**The honest headline is memory, not speed.** ~57 GiB of DDR5 came back. The throughput gain is
modest because the two cards **alternate rather than overlap** — sampling both during prefill
shows the Blackwell at 100% while the CMP sits at 0%, then the reverse. One sequential forward,
so the CMP's bandwidth advantage applies only to its 30% share. Overlapping them, or splitting
experts *within* a layer, is where the rest of the gain is; it was not attempted.

---

## Running it

`launchers/p24_boot.sh` is canonical (`p23` = same without tool calling, `p22` = single-card).

```bash
-e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1     # or nothing fits
-e DS41_CMP_DEVICE=cuda:1 -e DS41_CMP_LAYERS=28-39 # the expert tier
-e DSV41_ENGRAM_DISK=1                             # engram from SSD
--max-model-len 1048576 --max-num-seqs 1 --max-num-batched-tokens 2048
--kv-cache-dtype fp8 --kv-cache-memory 3221225472  # ~1.8M tokens
--gpu-memory-utilization 0.99
--offload-backend uva --cpu-offload-gb 60
--cpu-offload-params w13_trellis w13_suh w13_svh w2_trellis w2_suh w2_svh
--quantization exl3 --language-model-only --tokenizer-mode deepseek_v41
--enable-auto-tool-choice --tool-call-parser deepseek_v41
```

`launchers/build_sm80.sh` builds the dual-architecture image
(`TORCH_CUDA_ARCH_LIST="8.0;12.0a"`) and **refuses to commit** unless `cuobjdump` proves both
`sm_80` and `sm_120a` cubins exist.

The engram tables are **not** in the EXL3 repo — `tools/make_engram_index.py` and
`tools/link_engram_into_pack.py` hardlink the two 94.56 GiB base-repo shards into the pack and
merge their four embed keys into the index.

---

## Field notes

| symptom | what it actually was |
|---|---|
| `Failed core proc(s): {}` | an **empty** dict is the SIGKILL signature — read `dmesg`, not the traceback |
| `content: None` | a chat reply truncated by `max_tokens` returns null content; the field is `reasoning`, **not** `reasoning_content` |
| 1 token, empty text | `/v1/completions` against a chat-tuned model emits EOS immediately |
| `uva.py:61` in a traceback | the enclosing generator frame, not the fault — read past it |
| `14,227 tok/s` in the log | a counter-flush artifact of one 10 s window; the wall clock says 311 |
| `AssertionError: no sm_120` | nvcc emits `sm_120a`; an exact-equality arch check rejects a good build |

`docs/RUNLOG.md` is the complete chronological log — every phase, every fix, and every
diagnosis that had to be withdrawn.

---

## Not done

- The two cards alternate; pipelining them is where the remaining throughput is.
- A full 1M fill has not been run (~52 min of prefill); capacity is proven, 142K was pushed.
- No systemd unit — it runs under `docker run --rm` and does not survive a reboot.
- DSpark / MTP speculative decoding untouched.

Sanitized: domains are `example.invalid`, tokens are `<API_TOKEN>`. Hardware, byte counts and
measured numbers are real.
