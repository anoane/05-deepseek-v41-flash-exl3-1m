# DS4.1 bring-up run log
Format: ## [phase] timestamp — what was tried / what happened / what was changed

## [P1.1] 2026-09-13T22:29Z — pull day-0 vLLM image — OK
  docker pull vllm/vllm-openai:deepseekv41-flash-0909   -> rc=0, 21.4 GB
  This image OWNS DeepseekV41ForCausalLM. Stock vllm/vllm-openai wheels cannot load it.

## [P1.2] 2026-09-13T22:35Z — harvest sm120 patches from the diffbot recipe — OK
  hf download diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000 (112 non-weight files)
  -> /root/ds41/diffbot-recipe
  ALL THREE blockers from vLLM issue #56702 are solved there:
   #3 FlashInfer 32-token extra pages : patches/flashinfer/sparse_mla_sm120_prefill.cu
      (bind-mounted over flashinfer/data/csrc + an empty-aot dir masking the prebuilt kernel so it JIT-builds)
      their words: "any prompt longer than 64 tokens killed the engine"
   #2 block-size mismatch : sitecustomize.py overrides get_supported_kernel_block_sizes on
      DeepseekV4IndexerBackend + DeepseekV4FlashInferMLASparseBackend, and lifts vLLM's sm_10x-only
      gate on the MXFP4 indexer cache (dsa_indexer_uses_fp4) so 32-token pages become legal
   #1 SWA width 1152 : patches DeepseekV41Config.__init__ to clamp vision_max_n_token
      (do NOT zero vision_n_layers -> load_weights KeyError on gate.bias_vl)
  Also ships: xmoe CUDA MoE kernel (1.25-1.95x over stock exl3_moe), Marlin dense GEMM routing,
  quantize scripts, benches.
  PINS: VLLM_EXL3_REF=d3cfd394920360d69f820d2dc96f8292a9e10283
        EXLLAMAV3_REF=5be886578ec80324c2c715269387be2058724b6e
        base image sha256:<API_TOKEN>

  FINDINGS THAT CHANGE OUR PLAN:
  a) DSV41_ENGRAM_DISK=1 (engram from SSD, our topology) REQUIRES EAGER=1 -> no CUDA graphs.
     Their measured cost: 31 tok/s eager+NVMe vs 68 tok/s graphs+pinned-RAM engram. Accept and measure.
  b) KV is even cheaper than assumed: "V4.1 KV is tiny: 8 GiB holds ~3.8M tokens" -> 1M ctx ~2 GiB.
     Set --kv-cache-memory explicitly; do not let it infer.
  c) Their topology is TP2 on 2x96GB Blackwell with NO expert offload (2.0bpw fits natively).
     Ours (3.0bpw + 1 Blackwell + UVA DDR5 + CMP over x1) is unprecedented; patches transfer, placement does not.
  d) serve.sh needs: --block-size 64, --quantization exl3, --language-model-only, --tokenizer-mode deepseek_v41,
     --kernel-config '{"enable_flashinfer_autotune":false,"enable_jit_warmup":false}',
     --disable-custom-all-reduce, --attention-config '{"indexer_kv_dtype":"mxfp4"}'

## [P1.3-prep] 2026-09-13T22:40Z — decisions before building the image
  * XMOE_EXT MUST BE EMPTY FOR US. recipe README Limitations: "The kernel build trims ExLlamaV3's
    instance table to 2-bit MCG, which is what the pack uses." Their xmoe kernel is compiled for
    2-bit MCG ONLY. Our pack is 3.0bpw codebook=mul1 -> xmoe would not have our instance.
    Consequence: run stock exllamav3 exl3_moe, forfeit their 1.25-1.95x MoE speedup.
    Upside: kernels/build.sh is skipped entirely (it is optional).
  * Text only anyway (--language-model-only), so their "vision needs FlashInfer sm120 topk 1152"
    limitation does not affect us.
  * Build with NOBUILDKIT=1: recipe notes "BuildKit hung twice on our host".
  * build.sh also clones the sfxnz recipe at RECIPE_REF=460ac89c7df1eea4db8a731f1feb807f1584f359
    for docker/patch/ (apply_engram_disk.py, fix_vllm_exl3_setup.py, sm120_page.py, sitecustomize.py).
  * Disk was 91 G free -> reclaimed docker build cache (64.24 GB, 0 B active) + dead container.
    No tagged image deleted.

## [P1.3] 2026-09-13T22:37Z — build sm_120 serving image — OK (rc=0, ~3.5 min)
  cmd: TAG=dsv41-flash-exl3-sm120 NOBUILDKIT=1 MAX_JOBS=24 bash docker/build.sh
  result: dsv41-flash-exl3-sm120:latest  ac7217c14f2a  22.6 GB   (disk 100G -> 99G, thin layer over base)
  installed: vllm-exl3-0.4.2 (pin d3cfd394), exllamav3-1.4.9 (pin 5be88657, TORCH_CUDA_ARCH_LIST=12.0a)
  patched : apply_engram_disk.py -> engram.py + weight_utils.py   (engram-from-SSD machinery)
            sitecustomize.py -> /usr/lib/python3.12/sitecustomize.py (the 3 sm120 fixes)
  baked   : ENV VLLM_PLUGINS=vllm_exl3, ENV DSV41_ENGRAM_DISK=1, ENTRYPOINT ["vllm","serve"]
  NOTE: the in-build assertion `import exllamav3_ext; assert hasattr(...,"exl3_moe")` ran with
        "No CUDA runtime is found" (build container has no GPU). That proves the .so imports,
        NOT that it initialises on a real device. Re-verify with --gpus all before trusting it.
  MAX_JOBS: Dockerfile hardcodes 48; overridden to 24 (actual core count).

## [P1.3-verify] 2026-09-13T22:38Z — image verified against REAL GPUs — OK
  docker run --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID dsv41-flash-exl3-sm120
    torch 2.13.0+cu130 / cuda 13.0 / devices 2
    cuda:0 RTX PRO 6000 Blackwell sm_120 95.0 GiB | cuda:1 "NVIDIA Graphics Device" sm_80 63.4 GiB
    exl3_moe: True   exl3_moe_max_concurrency: True   vllm_exl3.runtime_diagnostics: True
    vllm 0.1.dev20904+g179dd0fa9   DeepseekV41Config importable: True
  patches in place: sm120_page.py present, engram.py 2x DSV41_ENGRAM_DISK, weight_utils.py 3x engram,
    ENV VLLM_PLUGINS=vllm_exl3 + DSV41_ENGRAM_DISK=1,
    BOTH FlashInfer bind-mount targets exist (flashinfer/data/csrc and
    flashinfer_jit_cache/jit_cache/sparse_mla_sm120 -> masked by empty-aot to force the 32-page JIT build)

  GOTCHA #1: image sitecustomize.py is 119 lines but the recipe's is ~200 (XMOE + DENSE_HYBRID hooks).
    serve.sh bind-mounts the recipe copy over /usr/lib/python3.12/sitecustomize.py at RUN time, so the
    richer one wins. Do NOT rely on the baked copy.
  GOTCHA #2: container warns "Detected different devices ... set CUDA_DEVICE_ORDER=PCI_BUS_ID".
    On this heterogeneous box rank<->device order matters (PP would put the trunk on the wrong card).
    PIN CUDA_DEVICE_ORDER=PCI_BUS_ID in every run.
  MY BUG: first probe died on exllamav3.__version__ (no such attribute). Image was fine.

## [P2] 2026-09-13T22:40Z — first boot — FAILED, but NOT for the reason I predicted
  cmd: TP1, CUDA_VISIBLE_DEVICES=0, ctx 4096, eager, NO offload, NO engram, stock MoE
  got to: "Resolved architecture: DeepseekV41ForCausalLM" -> "Using DeepSeek's fp8_ds_mla KV cache
          format" -> silence for 27 s -> SIGTERM. vLLM reported only
          "RuntimeError: Engine core initialization failed ... Failed core proc(s): {}"  (EMPTY dict)

  *** REAL CAUSE (kernel, not vLLM): HOST OOM KILL ON SHARED MEMORY ***
    Out of memory: Killed process 18349 (VLLM::EngineCor)
      total-vm:384577048kB   anon-rss:995216kB   shmem-rss:134238912kB
    oom-kill:constraint=CONSTRAINT_NONE, global_oom, task=VLLM::EngineCor
  ANALYSIS:
   - anon RSS < 1 GB but shmem 134 GB -> the loader was mapping the 205 GB pack into SHARED memory,
     not filling anon RAM with weights. My "it will fail on VRAM placement (~117 GiB deficit)"
     prediction was WRONG; it never reached placement.
   - empty "Failed core proc(s): {}" is the signature of SIGKILL: no Python traceback exists.
     Do NOT read that as "no error info"; go to dmesg/journalctl.
   - swap 0 => kernel had no relief valve; global OOM fired at ~134 GB shmem.
   - `--shm-size 32g` is NOT a cap here: with --ipc host, shmem grows into host RAM.

  LESSON: a vLLM "Engine core initialization failed" with an EMPTY failed-proc dict means the child
  was killed by the kernel. Always check `dmesg -T | grep -i oom` before theorising.

  FIXES FOR P2b:
   1. drop `--ipc host` and set a real `--shm-size` bound so shmem cannot eat host RAM
   2. VLLM_LOGGING_LEVEL=DEBUG so the child's progress is actually recorded
   3. drop the stray -e VLLM_EXL3_REF (vLLM warned: unknown env var) - my leak, cosmetic

## [P2-correction] 2026-09-13T22:45Z — my "engram shard 48 missing" alarm was FALSE
  Both shards present and complete:
    model-00047-of-00048.safetensors 101535150936
    model-00048-of-00048.safetensors 101537926640   (190 G total, no .incomplete)
  Cause of the false alarm: I piped `ls -la ... | head -5`, which cut the listing.
  MY INSTRUMENT, not a real gap. Engram is ready for P3.

## [P2b] 2026-09-13T22:43Z — same OOM. MY FIX WAS AIMED AT THE WRONG KNOB.
  changed vs P2: dropped --ipc host, --shm-size 32g -> 8g, added VLLM_LOGGING_LEVEL=DEBUG + RAM monitor
  monitor caught it in-band:
    [22:42:50] avail=124G shmem=0.0 GB
    [22:43:00] avail=71G  shmem=52.1 GB
    [22:43:10] avail=6G   shmem=128.0 GB   -> OOM
  kernel: Killed process 21849 (VLLM::EngineCor) anon-rss:995312kB shmem-rss:134238912kB
          (IDENTICAL shmem to P2: 134238912 kB) ; also OOM-killed redis-server as collateral

  => --shm-size does NOT bound this. It sizes /dev/shm; the loader's shared mapping of the
     safetensors is not /dev/shm. Removing --ipc host changed nothing either.

  *** REAL ROOT CAUSE (DEBUG revealed it) ***
    worker requested memory: 87.41GiB
    Offloader set to NoopOffloader (no offloading).
  204.12 GiB pack vs 87.41 GiB GPU budget and NO offload target -> loader maps the surplus into
  host shared memory until the kernel kills it.
  P2's PREMISE was wrong: "minimal boot without offload" is not a simpler case for this model,
  it is an impossible one. Offload is a PRECONDITION for booting, not a later phase.
  => P3 (engram) and P4 (UVA offload) must fold into the FIRST successful boot.

  Offload API in THIS build (vllm/config/offload.py):
    OffloadConfig.offload_backend = "auto" | "uva" | "prefetch"   (auto picks uva when cpu_offload_gb>0)
    OffloadConfig.uva = UVAOffloadConfig(cpu_offload_gb: float, cpu_offload_params: set[str])
    OffloadConfig.prefetch = PrefetchOffloadConfig(offload_group_size, offload_num_in_group, ...)
  `vllm serve --help` shows NO offload flags -> pass via config path, as the recipe does.

## [P3] 2026-09-13T22:46Z — UVA offload WORKED; failed later on a missing engram index
  flags all parsed (dash-cased EngineArgs names were right):
    offload_backend=uva  cpu_offload_gb=120.0
    cpu_offload_params=['w13_trellis','w13_suh','w13_svh','w2_trellis','w2_suh','w2_svh']
    engram_config=EngramConfig(cpu_offload=False)
    hf_overrides={'engram_table_dir': '/root/workspace/DeepSeek-V4.1-Flash-engram'}
  *** Offloader set to UVAOffloader ***  (was NoopOffloader in P2/P2b)
  NO OOM this time: shmem stayed 0.0 G, avail 128->124 G. Failed fast in Python instead. GOOD.

  FAILURE (exact):
    FileNotFoundError: Engram tensors missing from index:
      layers.1.engram.embed.weight / layers.1.engram.embed.scale
    at engram_disk.py:45, from engram.py:750 DiskEngramTable(model_dir, layer_id, dim, block_size)
    <- Engram(...) at model.py:189, forced by make_layers -> uva.py:61 wrap_modules generator.
  NOTE: uva.py:61 in the traceback is only the enclosing generator frame. The UVA offloader was
  NOT the fault. Read past it.

  CAUSE: DiskEngramTable.__init__ (engram_disk.py:39-41) reads
    model_dir/model.safetensors.index.json and requires the 4 keys
    layers.{1,14}.engram.embed.{weight,scale}. My engram dir had both 101 GB shards but NO index
    (hf download --include pulled only the two .safetensors). Data present, map absent.
  VERIFIED the tensors really are there, from the shard headers:
    shard 47: layers.1.engram.embed.weight  [384006168, 256] / .scale [384006168, 8]
    shard 48: layers.14.engram.embed.weight [384016682, 256] / .scale [384016682, 8]
    (+ k_weight [4,5120], q_weight [4,5120], wkv.weight [25600,6144], wkv.scale [800,192])
    dim=256 matches w_shape[1]; scale width 8 == dim//block_size for block_size=32. Consistent.
  FIX: generate model.safetensors.index.json in the engram dir FROM THE SHARD HEADERS
       (/root/ds41/make_engram_index.py) rather than copying diffbot's 14 MB index, so every
       entry is verified against the actual files.

  ALSO: "Unknown vLLM environment variable: VLLM_EXL3_REQUIRE_UVA_EXPERTS" is EXPECTED --
        it is a vllm_exl3 plugin var, not a vLLM one. Not the fault.
  MY BUG: a nested-quote docker/bash -lc probe died with "unmatched '". Use heredoc'd script files.

## [P3b] 2026-09-13T22:49Z — engram index in the ENGRAM dir did NOT help — wrong dir
  Identical FileNotFoundError. Cause found in engram.py:1085:
      model_dir=get_current_vllm_config().model_config.model
  DiskEngramTable is handed the EXL3 **PACK** path. `engram_table_dir` / --hf-overrides is NEVER
  consulted on this path -> my override was inert. Pack index had the 12 engram k/q/wkv keys but
  `embed.weight present: False` (embed lives only in base-repo shards 47/48).
  MY BUG: sed renamed container+log but not the `echo "P3 EXIT"` string -> P3b logged as "P3 EXIT".
          A run log that mislabels phases is worse than none. Fixed.

## [P3c] 2026-09-13T22:52Z — engram SOLVED; now OOM on the UVA budget
  FIX APPLIED: hardlink shards 47/48 into the PACK dir (links=2, same fs -> 0 extra disk) and merge
  ONLY the 4 embed keys into the pack index: 192452 -> 192456 keys, backup .bak.preengram.
  RESULT: "Engram tensors missing from index" occurrences = 0. Engram is DONE.
  (side effect: json.dump(indent=1) reformatted the 15 MB index 15501050 -> 14346600 bytes.
   Semantically identical, but I rewrote an artifact I did not need to reformat.)

  RED HERRING RULED OUT: json.decoder.JSONDecodeError came from
      vllm/usage/usage_lib.py:172 _report_usage_worker -> cpuinfo/cpuinfo.py:2762 json.loads
  i.e. the TELEMETRY thread shelling out to cpuinfo and getting empty output. Background thread,
  unrelated to weights/index, appears in any run. Do NOT chase it.

  *** REAL FAULT: --cpu-offload-gb 120 IS TOO SMALL, AND THE REMAINDER GOES TO SHMEM ***
  uva.py:_maybe_offload_to_cpu, per-parameter accounting:
     if self.cpu_offload_bytes >= self.cpu_offload_max_bytes: break     # stops offloading
     should_offload = any(f".{param}." in f".{prefix}{name}." ...)      # only matching params count
     cpu_data = p.data.to("cpu"); pin_memory(); get_accelerator_view_from_cpu_tensor(...)
     self.cpu_offload_bytes += p.numel()*p.element_size()
  Routed experts are 196.83 GiB. With a 120 GiB cap the offloader fills up, breaks, and the
  remaining ~77 GiB of experts are NOT offloaded and NOT on the GPU -> host shared memory.
  Trajectory confirms: shmem 0 -> 24 -> 52 -> 78 -> 104 -> 105 GB across the 40 FusedMoEConfig
  constructions, then OOM (kill at 22:53:20, shmem-rss 132406848 kB, anon 2.7 GB).

  CONSEQUENCE FOR THE PLAN: there is no "offload the surplus" mode. Offload is per-parameter and
  all-or-nothing in encounter order; the offloader cannot leave the remainder on the GPU.
  196.83 GiB of experts does NOT fit 142 GB of RAM.
  => the CMP must become a RESIDENT TIER (PP=2) so the offloaded share drops under the RAM ceiling.
     This is exactly the user's requested topology; I approached it from the wrong end.

## [P3d-spy] 2026-09-13T22:59Z — INSTRUMENTED the offloader. Both my diagnoses were WRONG.
  Method: monkeypatched UVAOffloader._maybe_offload_to_cpu via sitecustomize to print, per module,
  the real named_parameters() and the running budget. Observation instead of more theorising.

  WHAT IT ACTUALLY SEES:
    module #3 DeepseekV4DecoderLayer params=30 prefix=''
      moe-ish: ffn.experts.routed_experts.w13_trellis / w13_suh / w13_svh / w13_mcg / w13_mul1
               ffn.experts.routed_experts.w2_trellis / w2_suh / w2_svh   (+ ffn.shared_experts.*)
      budget: bytes=0            max=128849018880
    module #4 budget: bytes= 5113184256   (+4.76 GiB)
    module #5 budget: bytes=10226368512   (+4.76 GiB)
    module #6 budget: bytes=15339552768   (+4.76 GiB)

  => THE OFFLOADER IS WORKING. cpu_offload_bytes climbs ~4.76 GiB per decoder layer, matching the
     4.92 GiB/layer computed from the pack. My --cpu-offload-params names were CORRECT.

  TWO WITHDRAWN DIAGNOSES (both mine, same mechanism):
   1. [P3c] "budget filled at 120 GiB then break -> remainder to shmem" — WRONG, cpu_offload_bytes
      was 0 at that point only because the log line at uva.py:64 prints solely at the END of
      wrap_modules, and the process died before finishing. Absence of the summary line is NOT
      evidence that nothing was offloaded.
   2. [P3c-follow-up] "cpu_offload_params matches NONE" — WRONG. I compared my names against
      CHECKPOINT keys (layers.N.ffn.experts.M.w1.trellis). The offloader matches against MODULE
      named_parameters (ffn.experts.routed_experts.w13_trellis), which vllm_exl3 creates in
      create_weights (exl3.py:1832-1917) at layer-construction time, i.e. BEFORE wrap_modules
      consumes the generator in make_layers. Right names all along.

  THE ACTUAL ARITHMETIC (now unavoidable):
    40 layers x 4.76 GiB = ~190 GiB of routed experts to offload
    budget 120 GiB, host RAM 142 GB total, swap 0
    -> budget exhausts around layer ~25; the remaining ~15 layers of experts are neither offloaded
       nor on the GPU -> host shmem -> OOM. Same endpoint, correct mechanism this time.

  CONFIRMS THE PLAN: the CMP must be a RESIDENT TIER (PP=2) so the offloaded share falls under the
  RAM ceiling. From budget_math2: 16 layers resident -> 118.10 GiB offload (fits);
  20 resident -> 98.41 GiB (comfortable); 24 -> 78.73 GiB.
  Also seen: StageMissingLayer prefix='vision.'/'aligner.' (text-only OK) and engram.q_weight/
  k_weight/wkv.weight present on layer 1 (engram constructs correctly).

## [P5] 2026-09-13T23:03Z — PP=2 WORKS (CMP participating) but OOM: the budget is PER-RANK
  cmd: --pipeline-parallel-size 2 --tensor-parallel-size 1 --cpu-offload-gb 96
       NCCL_P2P_DISABLE=1, --disable-custom-all-reduce, engram on SSD, eager, ctx 4096

  PROGRESS — the CMP is finally a resident tier (user's topology, first time working):
    [23:03:21] shmem= 4.2G | gpu0= 8226MiB gpu1= 6463MiB
    [23:03:36] shmem=48.2G | gpu0=40578MiB gpu1=11675MiB
    [23:03:52] shmem=91.2G | gpu0=77016MiB gpu1=16887MiB
    [23:04:07] shmem=120.5G| gpu0=14586MiB gpu1=22079MiB     <- gpu0 drops = the kill
  rank 0 assigned: "rank 0 in world size 2 ... PP rank 0"; Worker_PP0 loading.
  kernel: Killed process 44385 (VLLM::Worker_PP) anon-rss:4947104kB shmem-rss:107386752kB

  *** MY ERROR: --cpu-offload-gb IS PER WORKER, NOT GLOBAL ***
  Two PP ranks x 96 GiB = up to 192 GiB of pinned host RAM authorised on a 142 GB box.
  Both ranks load CONCURRENTLY (gpu0 and gpu1 both climbing in the same samples), so the two
  offload pools race into the same RAM and OOM together.

  FIX DIRECTION: size the budget so BOTH ranks together fit under ~120 GB of usable RAM, i.e.
  roughly <=55-60 GiB per rank, and/or shift the layer split so rank 0 (Blackwell) carries fewer
  offloaded experts. Needs the real per-rank layer counts, not a guess.

  NOTE: this failure is DIFFERENT from P2/P2b/P3c. Those had no offload / an impossible total.
  Here offload works, PP works, the CMP holds 22 GB — only the aggregate budget is wrong.

## [P6] 2026-09-13T23:08Z — 50 GiB/rank FIXED THE OOM; new failure = unmapped exl3 lm_head
  change vs P5: --cpu-offload-gb 96 -> 50 (it is PER RANK; 2 x 50 = 100 GiB aggregate)
  RESULT: no kernel kill since 23:04:12. shmem peaked 76.4 G and RECEDED. gpu1 reached 22,079 MiB
          -> the CMP loaded its PP share cleanly. Furthest any attempt has got.
  NEW FAILURE (rank 0, during load_weights):
    ValueError: There is no module or parameter named 'head' in DeepseekV41ForCausalLM

  DIAGNOSIS CHAIN (all verified, no guessing):
   - pack ships head.trellis / head.suh / head.svh / head.mul1 ; ZERO lm_head.* keys
   - quantization_config.tensor_storage["head"]: quant_format="exl3", bits_per_weight=6,
     trellis [320,8080,96] int16 — structurally identical to layers.0.attn.wo_b (a known exl3
     linear). So the head IS exl3-quantized.
   - BUT quantization_config.json has NO non_routed_exl3 key and non_routed_quantization=null
     -> Exl3Config.non_routed_exl3 == {} -> _matches_non_routed_exl3() False for every prefix
        (exl3.py:1523-1524) -> ParallelLMHead hits `return None` (exl3.py:1698)
     -> head served UNQUANTIZED, expecting a single .weight
   - the VL mapper (vl_model.py:98-105) has ONLY a SUFFIX rule
        orig_to_new_suffix {"head.weight": "language_model.lm_head.weight"}
     Our four names do not end in "head.weight" -> no rule fires -> bare 'head' reaches
     AutoWeightsLoader -> ValueError.
   - WHY DIFFBOT NEVER HIT THIS: their 2.0bpw pack ships an UNQUANTIZED `head.weight` (1 key).
     Confirmed from their index. Their recipe is silent on this because it cannot occur there.

  NOT PP-specific and NOT sm_80: the mapper is identical under TP1 (P3c died in the engram before
  reaching load_weights, which is why this surfaced only now).

  FIX (two parts, neither works alone) -> /root/ds41/exl3_head_patch.py:
    A) inject non_routed_exl3 = {"modules": ["lm_head"], "bits": 6, "codebook": "mul1"}
       via Exl3Config.from_config (it reads config.get("non_routed_exl3"), exl3.py:209).
       Live-tested: _prefix_has_suffix('language_model.lm_head','lm_head') = True
                    _prefix_has_suffix('language_model.lm_head','head')    = False
    B) add mapper prefix rule "head." -> "language_model.lm_head."
  I staged B alone first and did NOT run it — without A the head is an unquantized ParallelLMHead
  wanting .weight, so B alone would have mapped 4 trellis tensors onto the wrong module shape.

## [arch check] exllamav3 in this image is sm_120 ONLY
  cuobjdump -lelf: 112 cubins, ALL sm_120. (strings showing sm_80/sm_30/sm_20 = string-table
  noise, not code.) So the CMP cannot execute exl3 kernels from this image.
  This did NOT cause any failure so far — rank 0 (Blackwell) always failed first — but it is a
  REAL constraint for the final topology and must be resolved before the CMP can run experts.

## [P8-design] 2026-09-14T01:55Z — the 622 non-routed modules: COMPLETE DESIGN, read from source

### The declaration is 22 suffixes, not 622 names
`_prefix_has_suffix` (exl3.py:1388) = `prefix == s or prefix.endswith("." + s)`.
2488 decoder non-routed exl3 leaf tensors / 4 leaves = **622 modules**, but only **22 distinct
module suffixes**. My earlier "622-module declaration" pricing was wrong by ~30x.

### Checkpoint module -> graph module (every row verified in the image's own source)
| checkpoint | graph module | constructor | quant_config? | exl3 support |
|---|---|---|---|---|
| attn.wq_a + attn.wkv | attn.fused_wqa_wkv | attention.py:305 MergedColumnParallelLinear([1280,512]) | YES | OK n_shards=2 |
| attn.wq_b | attn.wq_b | :314 ColumnParallelLinear | YES | OK |
| attn.wo_a.slice.0-7 | attn.wo_a | :324 ColumnParallelLinear, is_bmm=True | YES | **NO - gap** |
| attn.wo_b | attn.wo_b | :337 RowParallelLinear | YES | OK |
| ffn.shared_experts.w1/w3 | ...gate_up_proj | deepseek_v4/nvidia/model.py:884 DeepseekV4MLP | YES | OK n_shards=2 |
| ffn.shared_experts.w2 | ...down_proj | (mapper :944 renames w2->down_proj) | YES | OK |
| attn.indexer.wq_b | indexer.wq_b | :1064 ReplicatedLinear | YES | OK |
| attn.indexer.wk | indexer.wk | :1097 ReplicatedLinear | **None** | needs patch |
| attn.compressor.wkv/wgate | compressor.fused_wkv_wgate | compressor.py:219 | **None** | needs patch |
| engram.wkv | engram.wkv | engram.py:1089 ReplicatedLinear | YES | OK |
| head | lm_head | ParallelLMHead | (patch A) | OK |

### WHY THE FUSIONS ARE NOT A PROBLEM (my earlier fear was wrong)
Exl3LinearMethod.create_weights (exl3.py:2588) already handles merged linears:
  n_shards = len(output_partition_sizes); fused trellis concat on dim1;
  **suh is PER-SHARD, shape [n_shards, in_per_partition]** (exl3.py:2683-2686);
  svh concatenated; per-shard mcg/mul1 markers.
I had assumed wq_a and wkv would need a SHARED input Hadamard scale and that independent
calibration would therefore break concatenation. Wrong: exl3 stores one suh per shard.
stacked_params_mapping (model.py:701-704) feeds shard_id 0/1, and _make_weight_loader
(exl3.py:2765-2856) accepts int/str/span shard ids. So fused_wqa_wkv and fused_wkv_wgate
load natively once the modules are DECLARED. The old
`KeyError: 'layers.0.attn.fused_wqa_wkv.mul1'` was only ever a symptom of
UnquantizedLinearMethod giving the module a bare `.weight`.

### LAYER 20 ASYMMETRY IS BY DESIGN, NOT A DEFECT
Inventory showed compressor.wkv on 4 layers but compressor.wgate on only 3 (layer 20 lacks it).
compressor.py:205 `self.has_gate = compress_ratio > 1`; :216 `wkv_wgate_sizes =
[head_dim, head_dim] if has_gate else [head_dim]`. config compress_ratios[20] == 1.
=> layer 20 builds a 1-shard fusion. Nothing to fix.

### THE ONE REAL GAP: wo_a, AND ITS CHEAP FIX
`.slice.` has ZERO handling anywhere in vLLM (grepped linear.py + the whole deepseek_v4_1 tree).
wo_a is a BLOCK-DIAGONAL grouped matmul: o_groups=8, in = n_heads*head_dim/n_groups
= 64*512/8 = 4096, out = n_groups*o_lora_rank = 8*1024 = 8192. Each of the 8 slices has its
OWN input chunk. exl3's merged-shard support assumes a SHARED input, so it cannot represent
this; `slice.N` is purely an exl3-packer construct. diffbot's 2.0bpw pack has
`wo_a.weight`+`wo_a.scale` and NO slices at all -> they never hit this.

FIX (no new kernel, no _o_proj patch): deep_gemm_fp8_o_proj
(deepseek_v4/nvidia/ops/o_proj.py:49) computes `use_fp8 = wo_a.weight.dtype == float8_e4m3fn`
and its **else branch already does the bf16 grouped bmm**:
    grouped_weight = wo_a.weight.view(n_groups, o_lora_rank, -1)
    torch.bmm(o_proj_input.transpose(0,1), grouped_weight.transpose(1,2), out=z.transpose(0,1))
So: leave wo_a UNDECLARED (stays a dense bf16 [8192,4096] weight), and intercept the
slice keys at load: reconstruct each slice to dense and write it into row-block N.
`fused_inv_rope_fp8_quant(..., quantize=use_fp8)` disables its own quantization too.
Reconstruction is supported, not hand-rolled: exllamav3_ext exposes
reconstruct / reconstruct_slice / reconstruct_had_slice / unpack_trellis, and
LinearEXL3.get_weight_tensor() (exllamav3/modules/quant/exl3.py:227) does the full job
(inner reconstruct -> preapply_had_l -> *suh -> preapply_had_r -> *svh).
had_k = had_n = 128 (exl3_lib/quantize.py:15) and preapply_had_l/r need 128-divisible dims:
in 4096 = 32*128, out 1024 = 8*128. Both qualify.
COST: 8*4096*1024*40 = 1.34 G params -> 2.68 GiB bf16 vs ~0.84 GiB at K=5. ~+1.8 GiB VRAM.

### GEOMETRY CROSS-CHECK (pack headers vs config, independent)
layers.0.attn.wo_a.slice.0: suh[4096] svh[1024] trellis[256,64,80] int16
  -> in 256*16=4096, out 64*16=1024, 80=16*K -> K=5.   head: trellis[320,8080,96] -> K=6.
config: o_groups 8, o_lora_rank 1024, q_lora_rank 1280, head_dim 512, num_attention_heads 64.
All consistent. 621 modules at K=5, head at K=6.

### PLAIN (non-exl3) TENSORS — no surprises
attn_sink, attn_norm/ffn_norm/q_norm/kv_norm/compressor.norm/indexer.k_norm,
ffn.gate.weight+bias, indexer.weights_proj.weight, hc_* hyper-connection tensors,
engram.embed.weight/.scale + k_weight/q_weight, embed.weight, norm.weight.
Every one corresponds to a module built with quant_config=None or no quant at all. Consistent.

## [P8b-verify] 2026-09-14T02:05Z — sm_80 verdict AT THE PIN, and loader pre-flight

### The CMP is NOT excluded by the plugin — only by a build flag
`Exl3Config.get_min_capability()` returns **80**, comment: "LinearEXL3 uses CUDA >= Ampere;
GB10 is SM121" (exl3.py:1575-1578). So vllm_exl3 declares Ampere support.
The only reason the CMP cannot run exl3 today is that the image was COMPILED sm_120-only:
`TORCH_CUDA_ARCH_LIST=12.0a` in exactly two places —
  diffbot-recipe/recipe/docker/Dockerfile.sm120:26  and  recipe/docker/build.sh:34.
Verified AT THE PINNED REF 5be886578ec80324c2c715269387be2058724b6e (fetched; `cat-file -t`
= commit) — NOT at the two stale checkouts on the box (08849e35 / 05310963), which are a
different ref and would not have been valid evidence:
  * Hopper/Blackwell-only intrinsics (wgmma | tcgen05 | cp.async.bulk | OMMA | e2m1 |
    blockscaled) across all *.cu/*.cuh at the pin: **NONE**.
  * every __CUDA_ARCH__ guard is a DOWNWARD branch: `< 750`, `== 860`, `> 890`, `>= 800`
    (compat.cuh:12, codebook.cuh:10, exl3_gemm_inner.cuh:16, exl3_gemm_kernel.cuh:95/207/244,
    routing.cu:46). A `> 890` branch implies a working <= 890 path.
  * setup.py / CMakeLists.txt at the pin contain NO hardcoded gencode/arch -> torch's
    cpp_extension consumes TORCH_CUDA_ARCH_LIST directly.
=> adding `8.0` to those two lines and rebuilding SHOULD yield CMP-capable cubins.
Still a hypothesis until the build runs and `cuobjdump -lelf` shows sm_80 cubins.

### Loader pre-flight (read before writing any patch, to avoid burning boot cycles)
 * SCALAR mul1 IS HANDLED. Checkpoint stores `mul1` shape [] int32; exl3 allocates
   [n_shards,1]. _make_weight_loader (exl3.py:2904-2913) does
   `loaded_val = loaded_weight.detach().item() if numel() > 0 else 0` into a (1,) slot.
 * per-shard destinations are exactly as needed (exl3.py:2942-2954):
   trellis -> param[:, out_tiles_start:out_tiles_end, :] ; suh -> param[shard_idx] ;
   svh -> param[out_start:out_end]. So shard_id 0/1 from stacked_params_mapping lands right.
 * LinearEXL3.__init__ sets self.su=None/self.sv=None and takes suh/svh directly, so
   get_weight_tensor()'s `if self.su` short-circuits correctly; self.mcg/self.mul1 are BOOLs
   derived from the tensors and are what ext.reconstruct / BC_LinearEXL3 expect.
   => make_linear_exl3(...).get_weight_tensor() is a valid dequant path.
 * K VERIFIED ACROSS ALL 622 MODULES (not spot-checked): histogram {5: 621, 6: 1},
   consistent per suffix. So bits=5 + layer_bits={"lm_head":6} is exact.

### Mapper timing (decides where the wo_a hook goes)
DeepseekV41LLMForCausalLM.load_weights (model.py:1113-1115) ->
  AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)
so the mapper runs at the TOP and AutoWeightsLoader then delegates into
DeepseekV4Model.load_weights (model.py:696). No mapper rule touches .trellis/.suh/.svh/.mul1
(only `\.scale$` is rewritten). The FIX D regex accepts an optional leading prefix so it is
correct whether or not `model.`/`language_model.model.` has been stripped.

### PP CORRECTNESS TRAP CAUGHT BEFORE RUNNING
Under PP the weights iterator yields EVERY name to EVERY rank. A naive wo_a transform would
make rank 1 (the CMP, sm_80) reconstruct layers it does not own, running sm_120 kernels.
FIX D therefore hooks DeepseekV4Model.load_weights (where `self` is in scope) and gates on
is_pp_missing_parameter(base + ".weight", self), so each rank reconstructs only its own.

### MY BUG (heredoc)
I appended the P8 block with `ssh gpu 'cat >> RUNLOG.md' <<EOF` but put `echo "appended"`
INSIDE the heredoc, so it was written into the log as literal text instead of being run.
Detected (no output where output was expected), removed, P8 block verified intact.

## [P8c-clarify] 2026-09-14T02:12Z — "22 suffixes" vs "9 suffixes": both right, different sides
P8 says 22 distinct suffixes; the FIX A' declaration prints 9. Not a contradiction, but I would
have confused my future self:
  * 22 = CHECKPOINT-side distinct module suffixes (wq_a and wkv separate, w1/w2/w3 separate,
    compressor.wkv and wgate separate, wo_a.slice.0-7 counted as 8).
  * 9  = GRAPH-side suffixes actually declared, after vLLM's fusions and renames collapse them:
    wq_a+wkv -> fused_wqa_wkv; w1+w3 -> shared_experts.gate_up_proj; w2 -> shared_experts.down_proj;
    compressor.wkv+wgate -> compressor.fused_wkv_wgate; attn.wq_b and indexer.wq_b share suffix wq_b.
    Declared: fused_wqa_wkv, wq_b, wo_b, shared_experts.gate_up_proj, shared_experts.down_proj,
              indexer.wk, compressor.fused_wkv_wgate, engram.wkv, lm_head.
  * wo_a's 8 slices are in NEITHER count: deliberately undeclared, handled by FIX D.
Both numbers describe the same 622 checkpoint modules.

## [P9] 2026-09-14T01:43Z — the 622 non-routed modules LOAD. New failure: a superseded fp8 engram.wkv
cmd: /root/ds41/p9_boot.sh  (= p7_pp2_boot.sh with sitecustomize_p9.py; ONE variable changed)
sitecustomize_p9.py = sitecustomize_head.py + p9_snippet.py (477 lines, py_compile OK)

RESULT: furthest any attempt has reached. rc=1 at 01:44:19 (ran 71 s).
  * FIX A' / FIX C / FIX D armed in EVERY process (API server, engine core, both workers).
  * `[p9] non_routed_exl3: 9 suffixes, bits=5, layer_bits={'lm_head': 6}`
  * FIX C fired EXACTLY 8 times, and on exactly the right layers:
      MergedColumnParallelLinear ...layers.{2,8,14,20}.attn.compressor.fused_wkv_wgate
      ReplicatedLinear           ...layers.{2,8,14,20}.attn.indexer.wk
    matching the inventory (4 compressor layers, 4 indexer-with-wk layers). No over-matching.
  * Offloader set to UVAOffloader; `Total CPU offloaded parameters: 50.78`.
  * load trajectory: gpu0 13210 -> 44212 -> 52060 MiB, gpu1 6303 -> 11335 -> 16367 MiB,
    shmem 0 -> 7.0 -> 52.3 -> 76.4 G, avail 122 -> 36 G. NO kernel OOM kill.
  * CONFIRMED BY THE PREFIXES: the VL wrapper IS active even under --language-model-only
    (`language_model.model.layers.N...`). The FIX D regex's optional-prefix design was needed.
  * NO exl3 shape/marker error anywhere -> the 9-suffix declaration, the 2-shard fusions and
    the scalar mul1 markers all loaded correctly. The P8 design is validated.

FAILURE (rank 0, load_weights):
  KeyError: 'layers.1.engram.wkv.weight_scale_inv'   at model.py:813 `param = params_dict[name]`

DIAGNOSIS — VERIFIED FROM THE FILES, NOT INFERRED:
 * merged index engram entries: layers.{1,14}.engram.wkv.{trellis,suh,svh,mul1} ->
   model-000{02,15}-of-00041 (the PACK's exl3 copy); engram.embed.{weight,scale} ->
   shards 47/48; k_weight/q_weight -> the pack. NO engram.wkv.weight/.scale in the index.
 * shards 47/48 physically hold SIX keys each (read from the safetensors headers):
     engram.embed.weight [384006168,256] F8_E4M3 | embed.scale [...,8] F8_E8M0
     engram.k_weight [4,5120] BF16 | engram.q_weight [4,5120] BF16
     engram.wkv.weight [25600,6144] F8_E4M3 | engram.wkv.scale [800,192] F8_E8M0
 * => the loader uses the index only to choose FILES, then yields EVERY key in each file.
   Because the index points embed at 47/48, those files are opened and their fp8 wkv pair
   is yielded even though the index never names it.
 * the engram-disk patch skips ONLY (".engram.embed.weight", ".engram.embed.scale")
   (weight_utils.py:969) — that is why the 94 GiB tables stay on disk but wkv leaks.
 * mapper rule `re.compile(r"\.scale$") -> ".weight_scale_inv"` produces exactly the missing
   name. The pack's exl3 engram.wkv supersedes the fp8 pair, so it must be DROPPED.
 * k_weight/q_weight are duplicated (shards AND pack) but land on real params -> harmless.

FIX E (in _p9_transform): drop ^(.*\.)?layers\.N\.engram\.wkv\.(weight|scale|weight_scale|
weight_scale_inv)$ . Deliberately does NOT touch engram.wkv.{trellis,suh,svh,mul1}.

OPEN QUESTION — NOT EXPLAINED, DO NOT HAND-WAVE:
  [rank0] CUDACachingAllocator: "expandable_segments: memory mapping failed with OOM on
  device 0 while trying to map 20971520 bytes (free: 19595264, total: 102013927424)" at
  01:44:00 — i.e. ~19.6 MB free of 95 GiB — while nvidia-smi read 52060 MiB used 8 s later.
  Hypothesis (UNTESTED): UVA-mapped pinned host memory counts against device memory
  accounting, so 50.78 GiB offload + ~52 GiB resident ~= 103 GB > 95 GiB. If true,
  cpu_offload_gb is bounded by (GPU capacity - resident weights), which would change the
  budget arithmetic. It was a WARNING, not the fatal error. Watch it at KV-cache allocation.

## [P9b] 2026-09-14T01:48Z — ALL FIVE FIXES VALIDATED. Killed by the per-rank offload budget again.
change vs P9: + FIX E (drop superseded fp8 engram.wkv). Nothing else.

*** THE 622 NON-ROUTED MODULES NOW LOAD. The P8 design is fully confirmed. ***
  (Worker_PP0) [p9] wo_a reconstructed: 20 layers (640 slice-tensors not owned by this
               rank); dropped 4 superseded fp8 engram.wkv tensors
  Every number is exactly as predicted, which is the real check:
    20 layers        = 40 / 2 under PP=2
    640 slice-tensors = 20 layers x 8 slices x 4 leaves  (the other rank's share)
    4 dropped        = 2 keys (weight, scale) x 2 engram layers (1 and 14)
  (Worker_PP0) Loading weights took 43.23 seconds
  (Worker_PP0) Model loading took 50.69 GiB memory and 71.995427 seconds
  NO exl3 shape error, NO marker error, NO KeyError. FIX A' + C + D + E all correct:
   - 9-suffix non_routed_exl3 declaration covering 622 modules (K=5, head K=6)
   - fused_wqa_wkv / fused_wkv_wgate 2-shard loads via stacked_params_mapping
   - scalar mul1 markers
   - wo_a dequantized from 8 exl3 slices to a dense bf16 [8192,4096] BMM weight
   - superseded fp8 engram.wkv dropped

FAILURE: host OOM kill, NOT a model problem.
  kernel: 01:50:47 Out of memory: Killed process 156852 (VLLM::Worker_PP)
          shmem-rss:67385472kB (67 GB), total-vm:636283196kB
  trajectory: avail 121 -> 36 -> 21 -> 13 -> 9 -> 4 -> 1 G ; shmem 0 -> 76 -> 104 -> 116 G
              gpu0 53282 MiB (rank 0 DONE at 01:49:42) ; gpu1 6303 -> 56665 MiB, still climbing

ROOT CAUSE (the P5 lesson in a new form): --cpu-offload-gb is PER RANK, so BOTH ranks
offloaded ~50 GiB = ~100 GiB aggregate. Worse, rank 1 IS THE CMP — the card whose entire
purpose in this topology is to hold its layers RESIDENT because its PCIe link is Gen2 x1.
Offloading from the CMP to host RAM is not merely wasteful, it inverts the design.

ARITHMETIC — AN EVEN 20/20 SPLIT CANNOT WORK EITHER WAY:
  routed experts 196.83 GiB / 40 layers = 4.92 GiB/layer
  20 layers = 98.4 GiB of experts, but the CMP has only ~59 GiB usable at util 0.92
  (observed: gpu1 reached 56,665 MiB and was STILL loading when the kill landed).
  Usable: Blackwell ~88 GiB @0.92, CMP ~59 GiB @0.92, host ~100 GiB pinned. Total 247 > 204,
  so the model FITS — but only with an UNEVEN split:
    rank 0 (Blackwell): ~28 layers = 137.8 GiB experts -> ~88 resident + ~50 offloaded
    rank 1 (CMP)      : ~12 layers =  59.0 GiB experts -> fully resident, offload 0
  This is precisely the topology the user specified. It needs TWO knobs that P9b lacked:
    (a) an uneven PP layer partition
    (b) a RANK-AWARE offload budget (0 on the CMP)

OPEN QUESTION FROM P9 IS NOW LESS LIKELY TO BE REAL: the device-0 allocator warning
("free: 19529728 of 102013927424") reappeared at 01:48:56 and the run continued to load
fine for another 2 minutes, so it is not fatal on its own. Still unexplained; keep watching.

## [P10-design] 2026-09-14T01:57Z — the two knobs P9b lacked, plus the sm_80 rebuild
Staged and verified, not yet run (waiting on the dual-arch image build).

### Canonical patch stack renamed
sitecustomize_ds41.py = sitecustomize_head.py + p9_snippet.py = 535 lines, py_compile OK.
Six fixes, all present and each verified by grep before launch:
  FIX A  head-only non_routed_exl3   (original, now inert -- A' sets the key first)
  FIX A' full 9-suffix non_routed_exl3 declaration covering 622 modules (K=5, head K=6)
  FIX C  quant_config injected into compressor.fused_wkv_wgate and indexer.wk
  FIX D  wo_a: 8 exl3 slices -> dense bf16 [8192,4096] BMM weight
  FIX E  drop the superseded fp8 engram.wkv pair from base-repo shards 47/48
  FIX F  rank-aware offload budget (NEW)
(sitecustomize_p9.py is left in place but p10_boot.sh mounts sitecustomize_ds41.py.)

### FIX F — why the budget must be rank-aware
--cpu-offload-gb is PER RANK. P9b therefore authorised ~100 GiB of pinned host RAM on a
141 GB box and died. Worse, rank 1 IS the CMP, the card that must stay RESIDENT because
its link is Gen2 x1 (~0.38 GB/s H2D measured): offloading its experts to host RAM inverts
the entire topology. FIX F wraps UVAOffloader.wrap_modules and sets cpu_offload_max_bytes
= 0 whenever get_pp_group().rank_in_group != 0.
DONE LAZILY IN wrap_modules, NOT __init__: the offloader is constructed in
GPUModelRunner.__init__ (v1/worker/gpu/model_runner.py:349) and I did not want to depend on
whether the PP group is initialized that early. wrap_modules runs during weight loading,
after distributed init. uva.py's own comment says wrap_modules may be called more than
once, so the patch is idempotent (second call sees the budget already 0).
API verified, not guessed: get_pp_group() at parallel_state.py:1584 (asserts _PP is not
None), GroupCoordinator.rank_in_group at parallel_state.py:445.

### The 30/10 split, from measured numbers
An even 20/20 split is impossible in BOTH directions:
  experts 196.83 GiB / 40 = 4.92 GiB per layer; ~5.05 GiB/layer with attention+shared.
  20 layers = ~98 GiB but the CMP has only 64 GiB total, ~58.9 GiB at util 0.92.
  Observed in P9b: gpu1 reached 56,665 MiB and was STILL climbing when the kill landed.
Chosen 30/10:
  rank 0 Blackwell ~87.9 GiB usable (97887 MiB x 0.92):
      non-expert ~7.3 GiB (embed 1.32 + attn 2.57 + shared 0.66 + engram 0.18 + head 0.38
      + other 0.84 + wo_a bf16 premium ~1.35) + KV 2 GiB -> ~78.6 GiB for experts
      30 layers = 147.6 GiB of experts -> offload ~69 GiB  => --cpu-offload-gb 70
  rank 1 CMP ~58.9 GiB usable: 10 layers = ~50.5 GiB, FULLY RESIDENT, offload 0
Host peak becomes ~70 GiB (one rank), not ~100 GiB. Chose 30/10 over 29/11 deliberately:
the CMP has no offload fallback, so its only failure mode is a hard device OOM, while the
host easily absorbs rank 0's extra.
BONUS: layers 0-29 keep EVERY special structure on the Blackwell -- engram layers 1 and 14,
kv_source layers 2/8/14/20, all four indexer+compressor layers. The CMP gets plain MoE
layers 30-39, which is exactly what a 1x link can serve.
UNEVEN SPLIT IS SUPPORTED: VLLM_PP_LAYER_PARTITION (distributed/utils.py:143-154); len must
equal pp_size and sum must equal num_hidden_layers (40). The "realign the stage boundaries"
error at kv_cache_utils.py:1235 is in a MAMBA/MLA path (_pp_balanced_mamba_group_count) and
does not apply: DS4.1 has no mamba layers, and its compressor state caches (layers 2/8/14)
land on rank 0 alongside MLA layers either way.

### The sm_80 rebuild (in flight)
build_sm80.sh -> NEW tag dsv41-flash-exl3-sm80120, TORCH_CUDA_ARCH_LIST="8.0;12.0a".
Writes a NEW tag and never touches dsv41-flash-exl3-sm120, so the working image survives.
Gated: the build asserts via cuobjdump -lelf that BOTH sm_80 and sm_120 cubins exist and
refuses to docker commit otherwise -- a broken image must not be committed silently.
MY BUG, CAUGHT BEFORE SHIPPING: I wrote the script with `set -euo pipefail`, which would
abort before the bare `rc=$?` could run, making the "BUILD FAILED -> NOT committing" guard
unreachable and risking a commit of a broken image. Changed to `set -uo pipefail` with
explicit failure checks.
WHY IT IS NEEDED: rank 1 will hold 10 layers of exl3 experts and must EXECUTE them. The
current image's exllamav3_ext has 112 cubins, ALL sm_120. p10_boot.sh falls back to the
old image with a loud warning if the dual-arch build is absent.

## [BUILD-sm80] 2026-09-14T01:54Z — dual-arch image BUILT. The CMP can now run exl3.
cmd: TAG=dsv41-flash-exl3-sm80120 ARCHES="8.0;12.0a" bash /root/ds41/build_sm80.sh
compile: ~3.5 min, MAX_JOBS=24, load avg ~19 (healthy 24-way parallel build)

*** RESULT: CUBIN ARCHES: {'sm_120a': 112, 'sm_80': 112} ***
The P8b hypothesis is CONFIRMED: exllamav3 at pin 5be88657 compiles for Ampere. The image's
sm_120-only kernels were purely a build-flag artifact (TORCH_CUDA_ARCH_LIST=12.0a), never a
source limitation. 112 sm_80 cubins now exist alongside 112 sm_120a.
Verified in the FINAL COMMITTED IMAGE, not just the build log:
  docker run --rm dsv41-flash-exl3-sm80120 -> FINAL CUBIN ARCHES {'sm_120a':112,'sm_80':112}

MY BUG: THE GUARD REJECTED A GOOD BUILD.
  I asserted `c.get("sm_120")`, but nvcc emits the ARCH-SPECIFIC token `sm_120a` (from
  TORCH_CUDA_ARCH_LIST=12.0a), so the exact-key lookup returned None ->
  `AssertionError: NO sm_120 cubins -> the Blackwell would regress` -> refused to commit.
  The guard did exactly what it was built to do; the PREDICATE was wrong. Fixed to match by
  prefix (startswith "sm_80" / "sm_12") and the corrected script is on the box.
  LESSON: when asserting on tool output, match the shape the tool actually emits. An
  exact-equality check on an arch/version token is a rejection waiting to happen.

RECOVERY WITHOUT RECOMPILING:
  The inner script ran `set -euxo pipefail`, so the assertion aborted it BEFORE its final
    rm -rf /opt/exllamav3 /opt/vllm-exl3 /var/lib/apt/lists/*
    cp /opt/dsv41-patch/sitecustomize.py /usr/lib/python3.12/sitecustomize.py
  and, because the outer script uses `docker run --name` (no --rm), the container survived
  holding the full compile. So instead of burning another ~4 min:
    1. docker commit dsv41-img-build-sm80 -> dsv41-tmp:sm80build
    2. run those two skipped steps in a container from that tmp tag
    3. docker commit --change ENV/ENTRYPOINT -> dsv41-flash-exl3-sm80120
    4. re-verify cubins in the final image
  (The baked sitecustomize is irrelevant to our runs -- p10_boot.sh bind-mounts
   sitecustomize_ds41.py over it -- but I completed the step so the image matches sm120.)

DISK: 96G -> 93G free. Images: sm80120 24.2GB, sm120 22.6GB (UNTOUCHED, still the fallback),
tmp:sm80build 24.2GB (shares layers; retained for now as compile insurance, cleanup candidate).
The build container dsv41-img-build-sm80 is also retained.

## [P10] 2026-09-14T01:59Z — FIX F WORKS. Blocked by a v4.1 architectural constraint, not a budget.
cmd: /root/ds41/p10_boot.sh — image dsv41-flash-exl3-sm80120, VLLM_PP_LAYER_PARTITION=30,10,
     --cpu-offload-gb 70, sitecustomize_ds41.py.  rc=1 at 02:00:38 (42 s).

*** FIX F CONFIRMED — rank-aware offload works exactly as designed ***
  (Worker_PP1 pid=422) [p9] FIX F: PP rank 1 is the RESIDENT tier -> offload budget 70.00 GiB -> 0
  The PP rank resolved correctly at wrap_modules time (the lazy placement was right), and the
  CMP's budget was zeroed. Host RAM never left safe territory: avail 117 -> 103 G, shmem 4.1 G
  at the point of failure, versus 1 G / 116 G in P9b. The OOM mechanism is solved.
  P10 also correctly auto-selected the dual-arch image (IMAGE=dsv41-flash-exl3-sm80120).

FAILURE (rank 1, DeepseekV4DecoderLayer construction — NOT the offloader):
  attention.py:486
  NotImplementedError: Compressed-KV source language_model.model.layers.20.attn not found on
  this rank; PP splits inside a v4.1 kv-sharing group are not supported.

  NOTE: the traceback bottoms out at uva.py:61 `modules = [`. That is the enclosing generator
  frame, exactly as recorded in [P3]. I did NOT blame FIX F; read past it to attention.py:486.

*** WHY 30/10 WAS ILLEGAL, AND WHY THIS INVALIDATES THE WHOLE PP=2 PLAN ***
v4.1 shares compressed KV across layers: kv_source_layer_ids = [2, 8, 14, 20]. Every consumer
must sit on the same rank as its source, so a PP boundary may fall ONLY on a kv-group edge.
From config compress_ratios the groups are:
    {0-1}  {2-7}  {8-13}  {14-19}  {20-39}
The LAST GROUP IS 20 LAYERS AND INDIVISIBLE. Under PP=2 the second stage is layers k..39, so
it ALWAYS contains that whole group => the second stage can never hold fewer than 20 layers
(~98 GiB of experts). The CMP has 64 GiB (~58.9 at util 0.92). This is exactly what P9b showed
empirically: gpu1 reached 56,665 MiB on a 20-layer share and was STILL climbing.
Legal PP=2 boundaries are only k in {2, 8, 14, 20}. Consequences:
  - CMP as rank 1: >= 20 layers. Does not fit without host offload over a Gen2 x1 link.
  - CMP as rank 0 (needs CUDA_VISIBLE_DEVICES=1,0): k=8 -> 8 layers ~39 GiB FITS;
    k=14 -> 14 layers ~69 GiB does not. So k=8 is the only CMP-resident option, leaving
    32 layers (~157 GiB) on the Blackwell -> ~80 resident + ~77 offloaded over Gen5. Feasible
    on paper, but it puts layer 2's indexer/compressor and layer 1's engram on sm_80 hardware.

SECOND, STILL-OPEN QUESTION: CAN THE CMP RUN v4.1 ATTENTION AT ALL?
_select_dsv4_attn_cls (model.py:120-155) returns DeepseekV4FlashInferSM120Attention only when
device_capability.major == 12; every other arch falls through to DeepseekV4FlashMLAAttention.
FlashMLA is a Hopper path. If it has no sm_80 support then NO layer can run on the CMP,
regardless of the split, and the dual-arch exl3 kernels do not help because attention -- not
the MoE GEMM -- is what fails. P9b never reached a forward (host OOM first), so this has
never actually been tested. Being checked now; do not assume either way.

IMPLICATION IF THE CMP IS UNUSABLE: fall back to PP=1 on the Blackwell alone, which is the
configuration this box was explicitly prepared for -- 40 layers = 196.83 GiB of experts,
~80 GiB resident, ~117 GiB offloaded. That matches the 117.17 GiB figure computed earlier,
the reason RAM was raised to 142 GB, and /root/probe117.sh which pinned 117.2 GB successfully.
No kv-group constraint (no split), no sm_80 constraint, no x1 link in the hot path.

## [FINDING] 2026-09-14T02:05Z — THE CMP CANNOT HOST ANY v4.1 BACKBONE LAYER
Two independent constraints, both read from the image's own source. Together they rule the
CMP out of DeepSeek-V4.1-Flash entirely — this is architectural, not a tuning failure.

### 1. The kv-sharing groups are fixed, and the last one is indivisible
CONFIRMED FROM CODE, not inferred from compress_ratios (attention.py:284):
    self.kv_source_layer_id = max(s for s in self.kv_source_layers if s <= layer_id)
with kv_source_layer_ids = [2, 8, 14, 20]. So the groups are:
    {0-1} unconstrained (compress_ratio 0)  {2-7}  {8-13}  {14-19}  {20-39}
attention.py:477-487 raises NotImplementedError unless a consumer's source prefix is already
in _static_forward_context on the same rank. Legal PP=2 boundaries: k in {2, 8, 14, 20}.
The final group is 20 layers and INDIVISIBLE, so under PP=2 the second stage always carries
>= 20 layers (~98 GiB of experts) — more than the CMP's 64 GiB (~58.9 at util 0.92).
Only CMP-as-rank-0 with k=8 (8 layers, ~39 GiB) would have fit. Which brings us to:

### 2. There is NO v4.1 attention implementation the CMP can run
_select_dsv4_attn_cls (nvidia/model.py:120-155) returns DeepseekV4FlashInferSM120Attention
ONLY when device_capability.major == 12. Every other arch falls through to
DeepseekV4FlashMLAAttention — and in this image:
    python3 -c "import flash_mla"  ->  ModuleNotFoundError: No module named 'flash_mla'
So on the CMP (sm_80) the selected attention class has no backing library at all.

*** THE BLOCKER IS ATTENTION, NOT THE MoE GEMM. ***
This is why the dual-arch build does not rescue the topology. That build is correct and
verified ({'sm_120a': 112, 'sm_80': 112} in the committed image) and it genuinely does let the
CMP execute exl3 kernels — but exl3 only ever covered the expert/linear GEMMs. v4.1 attention
(sparse MLA + indexer + compressor) is Blackwell/Hopper-only code that exl3 never touched.
The sm_80 rebuild was still worth doing: it removes one real constraint permanently and is
reusable for models whose attention does run on Ampere (e.g. the GLM-5.3 PP work).

### Consequence: DS4.1 is a SINGLE-GPU model on this box
PP=1 on the Blackwell + host RAM. Arithmetic at util 0.97 and a 1 GiB KV budget:
    97887 MiB @0.97 = ~92.7 GiB usable
    non-expert ~9.1 GiB + KV 1 GiB -> ~82.6 GiB for experts, of 196.83 total
    => offload ~114 GiB
/root/probe117.sh already pinned 117.2 GB, so ~115 GiB is inside proven range — but it is the
tight part of this configuration, and it is exactly why RAM was raised to 142 GB.
Staged as /root/ds41/p11_boot.sh. FIX F deliberately does not fire (single rank = PP rank 0).

### What this means for the user's requested topology
The request was: Blackwell + CMP resident + DDR5 offload + engram on SSD. Three of the four
are achieved and proven. The CMP tier is NOT achievable for DeepSeek-V4.1-Flash with this
runtime, for the two reasons above. It remains valid for GLM-5.3-Flash, where PP=2 across
sm_80+sm_120 was already demonstrated (82.26 GiB / 50.39 GiB, 2026-09-12).

## [P11] 2026-09-14T02:06Z — single-GPU. OOM again. THE 3.0bpw PACK DOES NOT FIT THIS BOX.
cmd: PP=1, CUDA_VISIBLE_DEVICES=0, util 0.97, KV 1 GiB, --cpu-offload-gb 115, ctx 4096.
rc=1 at 02:08:33. Kernel: 02:08:22 Killed process 180147 (VLLM::EngineCor)
  shmem-rss:132289536kB = 126.2 GiB, anon-rss 2.68 GiB.
vLLM reported "Engine core initialization failed ... Failed core proc(s): {}" — the EMPTY
dict is the SIGKILL signature recorded in [P2]. Confirmed again: go to dmesg, not the traceback.

WHAT WORKED (worth keeping): all five loader fixes armed, and FIX F correctly stayed SILENT
because a single rank IS PP rank 0 — the predicted behaviour, so the rank gate is right in
both directions. No kv-group error (no PP split). No attention error.

*** THE ARITHMETIC, FROM MEASUREMENT NOT ESTIMATE ***
  declared offload budget            115.0 GiB
  actual peak shmem                  126.2 GiB   -> ~11 GiB OVERHEAD ABOVE THE BUDGET
  gpu0 peak                          94,492 MiB = 92.3 GiB (cap at util 0.97 = 94,950 MiB)
The same overhead appears in P9b: ~100 GiB authorised (2 x 50) -> 116 G shmem peak.
So budget is a FLOOR on host usage, not a ceiling: assume budget + ~11-15 GiB.

  pack decoder                       204.12 GiB
  max GPU residency @0.97            ~92.3 GiB
  => host must hold                  >= 111.8 GiB
  + measured overhead ~11            => ~123 GiB shmem
  + anon 2.7 + OS/baseline ~12       => ~138 GiB of 141 GiB total
That is inside the OOM noise floor, which is exactly what happened — at 02:08:21 the monitor
read avail=6G, and the kill landed one second later.

CONCLUSION: DeepSeek-V4.1-Flash EXL3 **3.0bpw** needs roughly 10-15 GiB MORE host RAM than
this VM has, on a single 96 GB GPU. And the CMP cannot absorb any of it, for the two
architectural reasons in [FINDING] (indivisible 20-layer kv group; no sm_80 v4.1 attention).
This is not a tuning failure and no further knob-turning closes it:
  - util 0.97 -> 0.99 buys ~2 GiB
  - KV is already at 1 GiB
  - offloading MORE increases host RAM, which is the exhausted resource, not the GPU
  - the wo_a bf16 premium is only ~1.8 GiB

THE THREE REAL OPTIONS (evidence-based, for the user to choose):
 1. Switch to a 2.0bpw pack (diffbot/...-2x-RTX-PRO-6000 or sfxnz@2.0bpw-mcg): decoder
    144.39 GiB -> ~92 GiB resident + ~52 GiB offload. Fits with large margin, and every fix
    built here carries over. Needs disk: the 3.0bpw pack (204 GiB) would have to go first.
 2. Raise VM RAM again (141 -> ~160 GiB). 3.0bpw then fits with ~20 GiB of headroom.
 3. Keep 3.0bpw and accept it will not serve on this hardware.

## [P12] 2026-09-14T02:11Z — VERDICT: 3.0bpw cannot serve on this box, and 1M context is out of reach
cmd: PP=1, util 0.99, --cpu-offload-gb 110, KV 1 GiB, drop_caches first, RAM-floor abort guard.
rc=137 at 02:13:09 — MY GUARD fired, not the OOM killer:
  *** RAM FLOOR HIT (avail=1G) -> aborting container before the OOM killer picks a victim ***
dmesg shows NO new kill (latest is still 02:08:22 from P11). The guard worked: controlled
abort, no collateral. P2b had taken redis-server down with it; this time nothing else died.

MEASURED (all three levers applied, each one helped, none was enough):
  drop_caches       pre-launch avail 118G (P11) -> 129G (P12)    +11 GiB
  util 0.97 -> 0.99  GPU cap 94,950 -> 96,908 MiB                +1.9 GiB
  budget 115 -> 110  host floor lowered                          -5 GiB
  peak: shmem 120.4 GiB at avail=1G, gpu0 had reached only 84,428 MiB (82.5 GiB)

*** WHY IT STILL FAILS — the offloader fills the HOST budget FIRST ***
gpu0 was still 12 GiB short of its own cap when host RAM ran out. wrap_modules pins its way
through the layers up to cpu_offload_gb before the GPU is full, so lowering the budget does
not hand the surplus to the GPU — the model simply needs more total memory than exists.
  pack decoder 204.12 GiB - GPU max 94.6 GiB = 109.5 GiB that MUST live in host RAM
  + ~10% pinning overhead (measured twice: 115 -> 126.2, and 110 -> 120.4) = ~120.5 GiB
  hard ceiling  = 141 total - ~20 GiB unreclaimable non-shmem = ~120-121 GiB
=> required ~120.5 vs available ~120.5. A dead heat, and it loses. There is no knob left:
   util is at 0.99, KV is at 1 GiB, and offloading MORE consumes the exhausted resource.

*** AND THE ACTUAL GOAL IS 1M CONTEXT, WHICH IS FURTHER OUT ***
KV at 1M is ~5.15 GiB (vs the 1 GiB used here), which takes ~4 GiB MORE off the GPU and adds
it to the host share: ~124.5 GiB required against a ~120.5 GiB ceiling. So even a lucky
ctx-4096 boot would not have led anywhere. 3.0bpw + 1M + this box is arithmetically dead.

*** WHAT WOULD WORK (measured, not guessed) ***
 1. 2.0bpw pack (diffbot/...-2x-RTX-PRO-6000 or sfxnz@2.0bpw-mcg), decoder 144.39 GiB:
      GPU 94.6 - ~7 non-expert - 5.15 KV(1M) = ~82.5 GiB of experts resident
      ~139 GiB experts total -> offload ~56.5 GiB -> shmem ~62 GiB of a ~120 GiB ceiling.
    Enormous headroom AND 1M context fits. Needs disk: 144.39 GiB against 93 G free, so the
    3.0bpw shards (204 GiB) would have to be deleted first.
 2. Raise VM RAM 141 -> ~160 GiB and keep 3.0bpw:
      1M needs ~114.3 GiB offload + 10% = ~125.7 GiB shmem + ~20 other = ~146 GiB. Fits in 160.
 3. Stop; restore glm53-sec.

EVERYTHING ELSE IS DONE AND PROVEN, and all of it carries over to a 2.0bpw pack unchanged:
the six-fix patch stack (622 non-routed exl3 modules incl. the wo_a BMM dequant and the
engram.wkv de-duplication), the engram-on-SSD wiring, the rank-aware offloader, and the
dual-arch sm_80+sm_120 image. The only thing that does NOT carry over is the CMP tier, which
is blocked by v4.1's architecture rather than by anything in this stack.

## [DECISION] 2026-09-14T02:15Z — user chose: raise VM RAM 141 -> ~160 GiB, keep the 3.0bpw pack
Nothing is deleted. Everything stays staged; a reboot does not disturb /root/ds41 or the images.

### Post-reboot plan (one variable at a time, as throughout)
 P13 = re-run p12_boot.sh UNCHANGED (ctx 4096, util 0.99, offload 110).
       Purpose: prove the load completes and the endpoint actually SERVES a token. P12 was
       ~0 GiB short; with ~19 GiB more RAM it should clear comfortably. Watch for the three
       things never yet observed: "[p9] wo_a reconstructed: 40 layers (0 ...) dropped 4",
       "Total CPU offloaded parameters", and "Uvicorn running".
 P14 = /root/ds41/p14_boot.sh, ctx 1,048,576. Only then change context.

### Arithmetic for 1M at 160 GiB (the config staged in p14_boot.sh)
  GPU 97887 MiB @0.99                       = 94.64 GiB
  non-expert resident                        ~9.14 GiB  (7.30 from the pack byte-split +
                                                         ~1.84 wo_a bf16 premium)
  KV at 1M                                   ~5.15 GiB  (DGX-Spark recipe measured 5.15 GiB
                                                         for 1,070,168 tokens; 6 GiB budgeted)
  workspace/activations                      ~2 GiB
  => experts resident ~78.35 of 196.83       -> offload ~118.5 GiB
  + 10% pinning overhead (measured twice)    -> ~130.3 GiB shmem
  + ~20 GiB unreclaimable non-shmem          -> ~150 GiB of 160 -> ~10 GiB margin.
  --cpu-offload-gb is set to 119, i.e. the MINIMUM needed, NOT a round number: the budget is
  a FLOOR on host usage (the offloader pins up to it before filling the GPU), so overshooting
  it wastes host RAM and starves the GPU. This is the single most counter-intuitive lesson of
  the whole run and it cost P9b, P11 and P12.

### Still true regardless of RAM (do not re-litigate)
The CMP cannot host any v4.1 backbone layer — see [FINDING]. More host RAM does not change
that; it is the indivisible 20-layer kv group plus the absence of any sm_80 v4.1 attention.
DS4.1 is a single-GPU model on this box. The CMP remains valid for GLM-5.3-Flash.

## [REBOOT] 2026-09-14T05:10Z — RAM raised: MemTotal 141G -> 157G. Everything survived.
The SSH timeout that preceded this was the REBOOT, not fail2ban. I suspected a ban first
(ai-vm-access records that heavy short-ssh polling gets the LAN IP banned, and I had been
polling hard), but the test distinguished them cleanly: ICMP answered, LAN ssh answered, and
`uptime` read "up 0 min". `gpu-wg` failing was consistent with the reboot too, since the VM
IS the WireGuard server. Hypothesis tested, not assumed.

Post-reboot state, all verified:
  MemTotal 157G, avail 146G, **Swap still 0** (so pinned memory remains unreclaimable)
  GPUs both back: cuda:0 RTX PRO 6000 97887 MiB, cuda:1 CMP 170HX 65536 MiB
  docker active; all 3 images intact (sm80120, sm120, tmp:sm80build)
  /root/ds41: 81 files, RUNLOG 870 lines, p10/p11/p12/p14 + sitecustomize_ds41.py present
  glm53-sec: disabled + inactive (unchanged, per instruction)
  disk 93G free
  NEW: /root/ds41/ds41_smoke.py — waits for :8111 then issues a real completion and reports
       TTFT/decode. No run has EVER served a token; "Uvicorn running" is not the finish line.

## [P13] 2026-09-14T05:11Z — re-run of p12_boot.sh UNCHANGED, now with 157G
Deliberately the SAME script, log and container names (p12_boot.sh / p12_boot.log / ds41-p12).
I did not rename them: a partial sed rename is exactly what mislabelled P3b earlier, and the
only thing that changed is the machine's RAM. One variable at a time.
Config: ctx 4096, util 0.99, --cpu-offload-gb 110, KV 1 GiB, drop_caches, RAM-floor abort.
P12 needed ~120.5 GiB shmem against a ~120.5 GiB ceiling and lost by a hair; with 146 GiB
available that is now ~25 GiB of margin.
WATCH FOR THE THREE THINGS NEVER YET OBSERVED:
  "[p9] wo_a reconstructed: 40 layers (0 slice-tensors not owned by this rank); dropped 4 ..."
  "Total CPU offloaded parameters: ~110"
  "Uvicorn running" -> then the smoke test's FINISH / COMPLETION_TOKENS / decode tok/s

## [P13] RESULT 2026-09-14T05:13Z — OOM at shmem 140.6 GiB. MY OVERHEAD MODEL WAS WRONG.
Same script, same budget (110), 16 GiB more RAM -> it died HARDER, not sooner-or-later:
  kernel 05:13:07 Killed process 4437 (VLLM::EngineCor) shmem-rss:147481536kB = **140.6 GiB**
  monitor: avail 146 -> 93 -> 67 -> 40 -> 11 -> 7 G ; shmem 0 -> 43.5 -> 70.6 -> 96.3 -> 124.4 -> 138.4 G
  P12 (141G RAM, same budget 110) peaked at shmem 120.4; P13 (157G RAM) reached 140.6.

*** WITHDRAWN: the "~10% pinning overhead" rule I recorded in [P12] and in memory. ***
It was induced from two numbers that were NOT comparable:
  - P12's 120.4 GiB was a MID-FLIGHT reading from a run my own guard aborted at avail=1G.
  - P11's 126.2 GiB was likewise the value at the moment of the kill.
Neither was a completed offload. **`Total CPU offloaded parameters` has NEVER printed in any
single-GPU run** — wrap_modules has never finished — so I never actually observed the budget
being honoured. I generalised a ratio from two truncated trajectories. That is the same error
class as [P3c] ("budget filled then broke"), which I also had to withdraw: reading a partial
trajectory as if it were a final state.

WHAT THE DATA ACTUALLY SHOWS: shmem climbs until RAM is exhausted, in every run, at every
budget (50/rank, 96/rank, 110, 115). More RAM simply moves the wall. So `--cpu-offload-gb` is
NOT bounding host memory the way I assumed, or something outside the pinned pool is also
consuming Shmem during load. I do not yet know which, and I will not guess again.

ALSO MY BUG: the RAM-floor guard did not fire. Floor was avail<4G polled every 10s; avail read
7G at 05:13:06 and the OOM killer struck at 05:13:07 — inside one poll interval. A guard whose
sampling period is longer than the failure it guards against is not a guard. Fixing: floor 12G,
poll every 3s.

NEXT (P15): the point is DIAGNOSIS, not another hopeful boot. ctx 4096, util 0.99,
--cpu-offload-gb 116 (sized so the GPU is not over-subscribed: 204.12 - 116 = 88.1 GiB of
weights + KV + workspace < 94.64 GiB usable), tight abort. The question it must answer is
narrow: **does shmem stop near the budget, or does it run away again?**
  - stops near ~116-128 GiB  -> the budget IS enforced; earlier runs simply over-subscribed
                               the GPU and the surplus spilled to host. Then raise it and serve.
  - runs away past ~140 GiB  -> the budget is not the controlling factor; instrument
                               UVAOffloader directly (as in [P3d-spy]) before any further boots.

## [P15] 2026-09-14T05:15Z — ANSWER: the budget does NOT bound host memory. Stop booting; instrument.
Diagnostic run: ctx 4096, util 0.99, --cpu-offload-gb 116, floor 12G polled every 3s.
  *** RAM FLOOR (avail=5G, shmem=132.4G) -> controlled abort ***
  *** ANSWER: shmem reached 132.4G against a 116G budget ***
The tightened guard WORKED: controlled abort, and dmesg shows NO new kill (latest is still
P13's 05:13:07). The 4G/10s guard that missed in P13 is fixed at 12G/3s.

RESULT: shmem overshot the budget by 16.4 GiB and was still climbing. Combined with P13
(budget 110 -> shmem 140.6), `--cpu-offload-gb` does NOT bound host memory here. Three
single-GPU runs at three budgets (110, 115, 116) all ran away; `Total CPU offloaded
parameters` has never printed once, so wrap_modules has never completed.

*** NEW EVIDENCE — the 3-second sampling caught something the 10-15s sampling never did ***
  05:16:40 avail=35G shmem=102.2G gpu0=89460MiB   <- GPU peak
  05:16:43 avail=30G shmem=108.4G gpu0=11156MiB   <- GPU COLLAPSES to 11 GB in one sample
  05:16:46 avail=20G shmem=115.4G gpu0=16106MiB
  05:16:49 avail=14G shmem=124.4G gpu0=21090MiB
  05:16:52 avail=5G  shmem=132.4G gpu0=31066MiB   <- refilling while shmem still grows
The GPU fills to ~89 GB, RELEASES almost all of it, then refills — while shmem rises
monotonically. That is the offloader moving already-materialised GPU tensors to pinned host
memory (uva.py `_maybe_offload_to_cpu`: `p.data.to("cpu")` per parameter, AFTER the module has
been built on the device). So host consumption is driven by the LOAD PATH, not by my budget:
every parameter is materialised on the GPU first and then copied out, and the budget only
decides when the offloader STOPS taking more — it cannot stop the weights that must still be
placed somewhere once the GPU is full.
This also explains the earlier device-0 allocator warning ("free: 19.6 MB of 102 GB") that I
logged as an open question in [P9]: the GPU really was momentarily full mid-load.

DECISION: no more boots on a hypothesis. Next step is the [P3d-spy] method — instrument
UVAOffloader to print, per module, cpu_offload_bytes vs cpu_offload_max_bytes alongside
/proc/meminfo Shmem, so the relationship between the budget, the pinned pool and Shmem is
OBSERVED rather than inferred. I have now been wrong twice about this subsystem by inferring
from trajectories ([P3c], and the withdrawn overhead model in [P12]); the cure is measurement.

## [P16-design] 2026-09-14T05:20Z — FIX G: instrument the offloader instead of guessing again
P16 is p15_boot.sh RE-RUN UNCHANGED (ctx 4096, util 0.99, offload 116, floor 12G/3s). The only
variable changed is the patch stack, which now carries FIX G. p15_boot.log was copied to
p15_boot.log.orig first, since the script truncates its own log and that log is the evidence
behind [P15].

### Why instrument
Three single-GPU runs at budgets 110 / 115 / 116 all drove host RAM to exhaustion, and
`Total CPU offloaded parameters` has never printed once — wrap_modules has never completed —
so the budget has never been OBSERVED to be honoured. I have now been wrong twice about this
exact subsystem by inferring from partial trajectories ([P3c], and the overhead model withdrawn
in [P13]). Same error class both times: reading a truncated trajectory as a final state.

### What FIX G prints
Wraps UVAOffloader._maybe_offload_to_cpu and, every 5 modules, logs the offloader's OWN
accounting beside the kernel's figures:
  [spy] mod=N offloaded=X/MAX GiB | Shmem=S MemAvail=A | gap=(S - X) GiB
**The gap column is the entire experiment:**
  gap flat and small -> the budget IS honoured and the overshoot is NOT the pinned pool; it
                        would then be weights that cannot fit the GPU spilling into host
                        memory, which matches the GPU collapse-and-refill seen in [P15].
  gap grows steadily -> the offloader under-counts its own allocations. uva.py:111-113 makes
                        TWO host copies per parameter — `p.data.to("cpu")` (pageable) and then
                        `cpu_data.pin_memory()` (a fresh pinned allocation plus copy) — while
                        only the final tensor's bytes are added to cpu_offload_bytes.

### Levers already located for whichever answer comes back (both real env vars, envs.py:304-305)
  VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 — drops the pin_memory() copy. On CUDA,
    get_accelerator_view_from_cpu_tensor does NOT pre-pin (only the XPU branch does), so a UVA
    view may still be obtainable from pageable memory.
  VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1 — falls back to functional_call on-demand movement
    (uva.py:124-150). Memory-light but moves weights per forward; likely far too slow here.
Also killed two stale ds41_smoke.py pollers from P13/P15 still inside their wait windows, which
would otherwise have fired against P16's endpoint and confused attribution.

## [P16] 2026-09-14T05:20Z — MEASURED: host RAM = 1.265 x the offloader's accounted bytes
FIX G worked. The spy, printing the offloader's own counter beside /proc/meminfo:
  mod= 3  offloaded= 4.76/116.00 GiB | Shmem=  6.1  MemAvail=132.1 | gap= 1.29
  mod= 5  offloaded=14.29/116.00 GiB | Shmem= 18.1  MemAvail=119.9 | gap= 3.80
  mod=10  offloaded=38.10/116.00 GiB | Shmem= 48.2  MemAvail= 90.2 | gap=10.08
  mod=15  offloaded=61.91/116.00 GiB | Shmem= 78.3  MemAvail= 59.7 | gap=16.36
  mod=20  offloaded=85.72/116.00 GiB | Shmem=108.4  MemAvail= 29.3 | gap=22.63
Shmem / offloaded = 1.281, 1.267, 1.265, 1.265, 1.265. **A CONSTANT 1.265.**

*** THE ANSWER: the gap GROWS, linearly. The offloader under-counts by 26.5%. ***
So the budget is NOT broken — it is simply never reached. It would cap at 116 GiB around
mod~27, but 116 x 1.265 = 146.7 GiB of Shmem, which exceeds RAM, so every run dies first.
Every previous "the budget does not bound host memory" reading was this multiplier in disguise.
Mechanism (uva.py:111-113): TWO host allocations per parameter —
    cpu_data = p.data.to(device="cpu")     # pageable copy
    cpu_data = cpu_data.pin_memory()       # a SECOND, pinned allocation + copy
    p.data   = get_accelerator_view_from_cpu_tensor(cpu_data)
    self.cpu_offload_bytes += p.data.numel() * p.data.element_size()   # counts ONE tensor
Per decoder layer the counter rises 4.76 GiB while host memory rises ~6.03 GiB.

*** THE OPERATIONAL LAW (use this for every future budget) ***
    host_bytes_required = cpu_offload_gb x 1.265
and therefore, for a safe Shmem ceiling C:   max cpu_offload_gb = C / 1.265.

### What that means for 3.0bpw at 157 GiB
  experts ~190.4 GiB (40 x 4.76) ; GPU holds 94.64 - ~9 non-expert - 1 KV = ~84.6 GiB
  => must offload ~105.8 GiB -> 105.8 x 1.265 = **133.8 GiB Shmem**
  ceiling ~136 GiB (abort came at Shmem 132.4 with avail 4G)
  => ~2 GiB of margin. Arithmetically possible, practically far too thin.

### P17: the single-variable test that could remove the multiplier
VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 (envs.py:304) drops the pin_memory() copy via
should_pin_memory(). On CUDA, get_accelerator_view_from_cpu_tensor does NOT pre-pin — only the
XPU branch does (torch_utils.py:907-919) — so a UVA view may still be obtainable from pageable
memory. If the multiplier collapses toward 1.0, the requirement falls ~134 -> ~106 GiB and the
margin becomes ~30 GiB. If instead the CUDA op rejects pageable memory, it fails fast and loudly,
which is also a clean answer. Same script, same budget, one env var changed.

## [P17] 2026-09-14T05:23Z — *** THE MEMORY WALL IS GONE. *** ratio 1.265 -> 1.000
ONE variable changed vs P16: VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1.
  mod= 3  offloaded=  4.76  Shmem=  4.8  ratio=1.008
  mod=10  offloaded= 38.10  Shmem= 38.1  ratio=1.000
  mod=25  offloaded=109.53  Shmem=109.6  ratio=1.001
  mod=30/35/40 offloaded=117.45 Shmem=117.5 ratio=1.000   <- plateau at the budget cap
The 26.5% overhead was EXACTLY the pin_memory() second copy (uva.py:112). Dropping it makes
host usage track the offloader's own accounting 1:1, which also validates FIX G as the right
instrument: the multiplier was measured, then removed, and the measurement predicted the result.

*** FIRST TIME IN ANY SINGLE-GPU RUN: ***
  (EngineCore) INFO 05:24:20 [uva.py:65] Total CPU offloaded parameters: 117.45
wrap_modules COMPLETED. In P9b/P11/P12/P13/P15/P16 it never once finished.
  gpu0 steady at 85,080 MiB; avail held 16-19G; shmem flat at 117.5G
  NO RAM-floor abort, NO OOM kill (dmesg still shows only P13's 05:13:07)
Budget 116 -> 117.45 actual: the per-parameter check is `>=` BEFORE adding, so the final
parameter overshoots by up to its own size. Size budgets accordingly.

### The operational law, corrected
    host_bytes = cpu_offload_gb x 1.265   WITH pinning (the default)
    host_bytes = cpu_offload_gb x 1.000   with VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
Pinning costs ~26.5% of host RAM here for no benefit we can use: UVA on CUDA gets its view via
get_cuda_view_from_cpu_tensor, which — unlike the XPU branch — does NOT require pinned memory
(torch_utils.py:907-919). It accepted pageable memory without complaint. Pinned memory would
only help DMA throughput; it was costing ~31 GiB of RAM at a 117 GiB budget.

### NEW failure, AFTER the wall — different problem, being diagnosed
05:24:20 offload completes -> 05:24:35 EngineCore traceback (core.py:1385) -> P17 EXIT rc=1.
Not OOM: avail was 16-19G and stable, and no kernel kill was recorded. `wo_a reconstructed`
had not printed yet, which is expected — wrap_modules runs during model CONSTRUCTION
(make_layers), and load_weights comes afterwards. So this fails between the two.

## [P17] ROOT CAUSE 2026-09-14T05:24Z — the head is UNQUANTIZED: ParallelLMHead never gets quant_config
Past the memory wall, load_weights now runs and fails on the very last module:
  vl_model.py:346 -> model.py:1115 -> models/utils.py:417
  ValueError: There is no module or parameter named 'lm_head.mul1' in
  DeepseekV41LLMForCausalLM. The available parameters belonging to lm_head (ParallelLMHead)
  are: {'lm_head.weight'}

DIAGNOSIS: the head module has ONLY `.weight`, so Exl3LinearMethod was never attached to it,
even though FIX A' declares "lm_head" and the run printed
  [p9] non_routed_exl3: 9 suffixes, bits=5, layer_bits={'lm_head': 6}
exl3.py:1683-1686 states the reason outright: "Embedding-family layers. vLLM only consults
quant_config for these when the model passes it (qwen4_exp needs quant_config= on
ParallelLMHead ...)". get_quant_method (exl3.py:1694-1698) is simply never called for a
ParallelLMHead built without quant_config, so _matches_non_routed_exl3 never gets a chance —
the declaration is correct but unreachable.

*** THIS IS THE FIX C FAILURE CLASS AGAIN ***, on the embedding family instead of a linear:
a constructor that does not forward quant_config, so a module the pack DID quantize is served
unquantized. FIX C solved it for compressor.fused_wkv_wgate and indexer.wk by injecting
quant_config by prefix; the head needs the same treatment.
It also closes out the original [P6] head problem properly: FIX A+B fixed the NAMES
(head.* -> lm_head.* via the VL mapper) but not the MODULE. The pack's head is exl3 at K=6
(trellis [320,8080,96]); nothing but an exl3 method can consume trellis/suh/svh/mul1.

NOTE the name in the error is `lm_head.mul1`, not `language_model.lm_head.mul1`: inside the
inner LLM module the prefix is plain `lm_head`, which is what any injection must match.
FIX H is being written against the actual constructor, not assumed.

## [P18-design] 2026-09-14T05:28Z — FIX H: give ParallelLMHead a quant_config
P18 = p17_boot.sh RE-RUN UNCHANGED (it already carries VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1,
ctx 4096, util 0.99, offload 116, floor 12G/3s). Only the patch stack changes. p17_boot.log was
copied to p17_boot.log.orig first — it holds the 1.000-ratio evidence and the script truncates it.

FIX H wraps ParallelLMHead.__init__ and injects the live quant_config when the prefix is
`lm_head` or `*.lm_head`. Verified before writing, not assumed:
  * the constructor DOES accept the kwarg — vocab_parallel_embedding.py:581
    `quant_config: QuantizationConfig | None = None`
    (FIX H still checks the signature at runtime with inspect and injects only if accepted, so
     it degrades to a no-op rather than a TypeError on a different vLLM build.)
  * geometry matches with no padding surprise: VocabParallelEmbedding calls create_weights with
    `num_embeddings_padded`, and vocab_size 129280 is already a multiple of both 64
    (DEFAULT_VOCAB_PADDING_SIZE) and 128 -> padded == real. exl3 will see
    in=5120, output_partition_sizes=[129280] -> in_tiles 320, out_tiles 8080, K=96/16=6,
    matching the pack's head trellis [320, 8080, 96] exactly. n_shards=1, so exl3's
    padded-geometry restriction is not engaged either way.
  * the error named `lm_head.mul1`, so inside the inner LLM the prefix is plain `lm_head`;
    FIX H matches both that and the `language_model.lm_head` form.

The patch stack is now eight fixes (A, A', C, D, E, F, G, H). Every one of them exists because
the Mia 3.0bpw pack quantized modules that stock vLLM serves unquantized, or because the
offloader's behaviour differs from its documentation. None is speculative: each was written
against a specific observed failure and its source line.

## [P18] 2026-09-14T05:27Z — *** THE MODEL LOADS. All eight fixes confirmed in one run. ***
p17_boot.sh + patch stack with FIX H. ctx 4096, util 0.99, offload 116, DISABLE_PIN_MEMORY=1.

EVERY FIX CONFIRMED WORKING SIMULTANEOUSLY:
  [p9] FIX H armed (ParallelLMHead accepts_quant_config=True)
  [p9] FIX H: quant_config -> ParallelLMHead 'language_model.lm_head'
  INFO 05:29:04 [uva.py:65] Total CPU offloaded parameters: 117.45     (shmem flat 117.5G, ratio 1.000)
  [p9] wo_a reconstructed: 40 layers (0 slice-tensors not owned by this rank);
       dropped 4 superseded fp8 engram.wkv tensors
  INFO 05:30:34 [default_loader.py:430] Loading weights took 89.17 seconds
  INFO 05:30:37 [model_runner.py:408] Model loading took 82.64 GiB memory and 144.12 seconds

*** `wo_a reconstructed` HAD NEVER PRINTED IN ANY RUN, AND THE NUMBERS ARE EXACTLY AS PREDICTED ***
  40 layers      = all of them (PP=1, so no layer belongs to another rank)
  0 not-owned    = correct for a single rank (was 640 under PP=2: 20 x 8 slices x 4 leaves)
  4 dropped      = 2 keys x 2 engram layers (1 and 14)
That is the whole P8 design validated end to end: the 9-suffix non_routed_exl3 declaration
covering 622 modules, the 2-shard fusions, scalar mul1 markers, the wo_a BMM dequant, the
engram de-duplication, and the exl3 head. load_weights RAN TO COMPLETION — the
`lm_head.mul1` ValueError that killed P17 is gone.

Memory behaved exactly as the P16/P17 measurements predicted: shmem pinned flat at 117.5 GiB
for the entire load, avail steady ~15-16G, gpu0 ~86.4 GiB, no OOM, no abort.

STATUS AT THIS ENTRY: past model loading, in post-load engine init; endpoint not yet answering.
RISK BEING WATCHED (not acted on): avail 15G sits only 3G above this script's own 12G abort
floor, and I cannot retune a running monitor. If warmup allocates a few more GiB the guard
will kill a run that might otherwise have succeeded. Now that shmem is known to plateau flat
at 117.5 GiB rather than climb, a lower floor (~6G) is safe for the next run.

## [P18] ROOT CAUSE 2026-09-14T05:30Z — MY OWN FIX C broke the forward path
The model LOADED (see above) and then died on the FIRST FORWARD, during profile_run:
  core.py:145 _initialize_kv_caches -> determine_available_memory -> profile_run
  -> _dummy_run -> execute_model -> attention.py:562 forward
  -> attention.py:793 _run_parallel_input_projections -> multi_stream_utils.py:107
  -> attention.py:777 compressor_kv_score:
        compressor.fused_wkv_wgate.weight.T,
  AttributeError: 'MergedColumnParallelLinear' object has no attribute 'weight'

*** THIS ONE IS MINE, NOT THE PACK'S. *** FIX C injected quant_config into
compressor.fused_wkv_wgate so the exl3 tensors would load. That worked — and it replaced the
module's `.weight` with trellis/suh/svh/mul1. But attention.py:777 does NOT call the module;
it reaches into `.weight` directly and transposes it for a raw matmul, bypassing forward() and
the quant method entirely. So FIX C fixed LOADING and broke EXECUTION.
Lesson: giving a module a quant_config changes its parameter surface. Any code that touches
that module's `.weight` directly must be found BEFORE the fix is shipped, not after.

FIX I (mirrors FIX D, which already solved exactly this shape of problem for wo_a):
 * drop "compressor.fused_wkv_wgate" from the non_routed_exl3 declaration (FIX A') AND from
   FIX C's injection list, so the module stays a plain dense linear with a real `.weight`;
 * dequantize compressor.wkv / compressor.wgate at load time and yield a single
   `...compressor.fused_wkv_wgate.weight`, concatenated [wkv; wgate] along dim 0 to match
   MergedColumnParallelLinear([head_dim, head_dim]); on layer 20 has_gate is False
   (compress_ratio == 1), so that layer yields wkv alone -> [512, 5120].
 * cost is trivial: 4 compressor layers x ~1024x5120 bf16 ~= 40 MB total.

BEFORE SHIPPING: sweeping the whole deepseek_v4_1 tree for direct `.weight` access on EVERY
module I declared exl3 (fused_wqa_wkv, wq_b, wo_b, shared_experts.*, indexer.wk, engram.wkv,
lm_head). indexer.wk is the immediate suspect since FIX C injects it too. Fixing these one
failed boot at a time would be several more cycles; one grep finds them all at once.

## [P19-design] 2026-09-14T05:35Z — FIX I: compressor stays dense, dequantized at load
THE SWEEP THAT BOUNDED THE PROBLEM (one grep instead of several failed boots):
searched the whole deepseek_v4_1 tree for direct `.weight` access on every module the
declaration makes exl3. On the nvidia path there is EXACTLY ONE:
    attention.py:777   compressor.fused_wkv_wgate.weight.T
Everything else is irrelevant: amd/rocm.py:565,652 (fused_wqa_wkv.weight, wo_b.weight) are on
the ROCm path we never take, and the "head.weight" hits are WeightsMapper rename STRINGS, not
attribute access. Critically, indexer.wk is NEVER read via `.weight`, so FIX C's indexer half
is safe and FIX I's blast radius is a single module.

FIX I, three coordinated changes:
 1. "compressor.fused_wkv_wgate" dropped from _P9_NON_ROUTED_MODULES (FIX A')
 2. _P9_NEEDS_QC reduced to ("indexer.wk",) (FIX C)
 3. _p9_transform now consumes layers.N.attn.compressor.{wkv,wgate}.{trellis,suh,svh,mul1}
    and emits one dense `...compressor.fused_wkv_wgate.weight`
So the module keeps a real `.weight` and attention.py:777 works untouched — no patch to the
model's forward path, mirroring how FIX D solved wo_a via deep_gemm_fp8_o_proj's bf16 branch.

GEOMETRY, read from the pack rather than assumed:
  compressor.wkv / wgate: suh[5120] svh[512] trellis[320,32,80] -> in 5120, out 512, K=5
  MergedColumnParallelLinear(5120, [head_dim, head_dim]) wants [1024, 5120], wkv rows FIRST
  (stacked_params_mapping: compressor.wkv -> shard 0, compressor.wgate -> shard 1)
  layers.20.attn.compressor.wgate is GENUINELY ABSENT (compress_ratio == 1 -> has_gate False),
  so layer 20 yields wkv alone -> [512, 5120].
Because that absence is only knowable once the key stream ends, compressor groups are flushed
at END OF STREAM, not eagerly on completion like wo_a's deterministic 8 slices.
Cost: 4 compressor layers x ~1024x5120 bf16 ~= 40 MB. Negligible.

MY PROCESS ERROR WORTH RECORDING: two Edit attempts failed with "string not found" because I
reconstructed the target text from memory instead of reading the file first (same class as the
build_sm80.sh indentation miss). Read the file, then edit it.

## [P19] 2026-09-14T05:37Z — *** THE ENDPOINT SERVES. *** First time in the whole bring-up.
FIX I worked exactly as designed:
  [p9] FIX C armed (indexer.wk only; compressor handled by FIX I)
  [p9] wo_a reconstructed: 40 layers; compressor fused: 4; 0 tensors not owned by this rank;
       dropped 4 superseded fp8 engram.wkv
  *** compressor fused: 4 *** — exactly the prediction (layers 2, 8, 14, 20; layer 20
  contributing wkv ALONE because compress_ratio == 1 -> has_gate False).
The AttributeError at attention.py:777 is gone: the module kept a real dense `.weight`, so the
model's own forward path was never patched.

FULL STARTUP, for the first time:
  INFO 05:39:11 [uva.py:65]          Total CPU offloaded parameters: 117.45
  INFO 05:40:41 [default_loader.py]  Loading weights took 89.59 seconds
  INFO 05:40:44 [model_runner.py]    Model loading took 82.66 GiB memory and 146.17 seconds
  INFO 05:41:03 [kv_cache_utils.py]  GPU KV cache size: 30,400 tokens
                                      Max concurrency for 4,096 tokens/request: 7.42x
  INFO  Application startup complete.        endpoint answered after 215 s
Steady state: shmem FLAT at 117.5 GiB, gpu0 ~88.0 GiB, avail 13 G, 47 C, 90.9 W. No abort,
no OOM. The 6G floor was the right call — a 12G floor would have aborted this successful run.

### NOT DECLARING VICTORY: the first generation came back EMPTY
  smoke: FINISH: stop | PROMPT_TOKENS: 18 | COMPLETION_TOKENS: 1 | TEXT: ''
One token, empty string, immediate EOS. Two very different explanations and I will not assume
which:
  (a) BENIGN — /v1/completions is the wrong shape for a chat/reasoning-tuned model served with
      --reasoning-parser deepseek_v41; a raw prompt with no chat template can legitimately emit
      EOS immediately.
  (b) SERIOUS — corrupted weights from one of the dequant/declaration fixes (wo_a's 320
      reconstructed slices, the compressor fusion, the exl3 head). Loading cleanly proves
      SHAPES are right; it proves nothing about VALUES.
Probing now with a proper /v1/chat/completions call (printing content AND reasoning_content,
since the reasoning parser may route text to the latter) plus a raw completion with min_tokens
forcing continuation. Until coherent text comes back, this is "loads and starts", not "works".

## [P19] VERIFIED 2026-09-14T05:42Z — *** CORRECT OUTPUT. DeepSeek-V4.1-Flash 3.0bpw SERVES. ***
The empty first generation was explanation (a) — the wrong API shape — NOT broken weights:
  /v1/chat/completions  "What is 17 multiplied by 23? Answer with the number only."
     -> content: '391'      (17 x 23 = 391, CORRECT)   20 tokens, 2.1 s, finish=stop
  /v1/completions       "The capital of France is"  (min_tokens=8, temperature=0)
     -> " Paris. The capital of France is Paris. The capital of France is Paris. ..."
        coherent; the repetition is just greedy decoding with forced continuation.
A raw /v1/completions prompt against a chat/reasoning-tuned model legitimately emits EOS at
once, which is exactly what the smoke test saw. Lesson: probe a chat model with the chat
endpoint before concluding anything about output quality.

*** THIS VALIDATES VALUES, NOT JUST SHAPES. *** Loading cleanly only proved the tensors fit;
correct arithmetic proves the reconstructions are numerically right — the 320 dequantized
wo_a slices (8 per layer x 40), the 4 fused compressors, and the exl3 head all produce
correct results. Nine fixes (A, A', C, D, E, F, G, H, I), zero patches to the model's forward.

RUNNING CONFIGURATION THAT WORKS (single GPU, 157 GB host):
  image dsv41-flash-exl3-sm80120 ; sitecustomize_ds41.py (709 lines)
  PP=1, CUDA_VISIBLE_DEVICES=0, --gpu-memory-utilization 0.99, --enforce-eager
  --offload-backend uva --cpu-offload-gb 116 --cpu-offload-params w13_*/w2_*
  **VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1**  <- without this nothing fits (1.265x)
  DSV41_ENGRAM_DISK=1 (engram streamed from SSD), --language-model-only, --block-size 64
  ctx 4096, --kv-cache-memory 1 GiB
  steady: shmem 117.5 GiB flat, gpu0 88.0 GiB, avail 13 G, 47 C, 91 W, load 215 s to ready

### NEXT PROBLEM: the KV arithmetic for 1M is NOT what I assumed
  GPU KV cache size: 30,400 tokens for --kv-cache-memory 1 GiB  ->  ~35 KB/token
That is ~40x the ~890 B/token figure I carried from the DGX-Spark recipe and wrote into
memory. At 35 KB/token, 1M tokens needs ~34.5 GiB of KV, which does NOT fit beside 88 GiB of
weights on a 94.64 GiB card. My note that "only 4 layers produce KV (kv_source_layer_ids
[2,8,14,20])" is the probable error — the indexer and sparse-SWA caches also consume per-token
memory. Measuring the real breakdown from the running server before sizing the 1M run; the
budget must come from this box's own numbers, not another deployment's.

## [KV] 2026-09-14T05:43Z — MEASURED: ~35.3 KB/token, so 1M does NOT fit the 3.0bpw pack
  INFO [gpu_worker.py:551] Initial free memory 94.46 GiB, reserved 1.0 GiB memory for KV Cache
       as specified by kv_cache_memory_bytes config and SKIPPED MEMORY PROFILING. This does not
       respect the gpu_memory_utilization config.
  INFO [kv_cache_utils.py:2321] GPU KV cache size: 30,400 tokens
=> 1,073,741,824 B / 30,400 = **35,320 B/token**.
KV format: fp8_ds_mla plus an MXFP4 indexer cache for the Lightning Indexer, BLHNC layout.

*** MY STORED FIGURE WAS WRONG BY ~40x. *** Memory said "~890 B/token, 5.15 GiB at 1M",
taken from the DGX-Spark recipe, together with "only 4 layers produce KV
(kv_source_layer_ids [2,8,14,20])". That did not survive contact with this box. Another
deployment's KV number is not transferable: it depends on kv_cache_dtype, the indexer cache
dtype, the sparse-SWA caches and the backend's layout. Measure it on the machine you serve on.

### 1M arithmetic on THIS box (3.0bpw)
  1,048,576 tokens x 35.32 KB = **35.3 GiB of KV**
  available: 94.46 GiB free at start - 82.66 GiB of weights = **11.8 GiB** -> ~350K tokens
  push offload to the host ceiling (~125 GiB; non-shmem overhead measures ~26.5 GiB of the
  157 total, so shmem cannot exceed ~125-130) -> GPU weights ~79.1 GiB
  -> KV headroom ~15.3 GiB -> **~455K tokens**
So 1M is NOT reachable with the 3.0bpw pack on a single 96 GB card, at any offload setting.
It needs the 2.0bpw pack (decoder 144.39 GiB): GPU weights ~55 GiB after a ~90 GiB offload,
leaving ~39 GiB for KV -> comfortably past 1M. That is the same conclusion [P12] reached from
the weight side, now confirmed independently from the KV side.

### P20: establish the real ceiling by MEASUREMENT
ctx 262,144 with --kv-cache-memory 9 GiB (predicts ~273K tokens at 35.32 KB), offload
unchanged at 116. Weights 82.66 + KV 9.0 = 91.7 GiB < 94.46 ✓. One variable changed from the
known-good P19 config, and it doubles as a second data point to confirm the KV law is linear.

## [KV] RETRACTED 2026-09-14T05:48Z — 1M IS REACHABLE. My 35.3 KB/token was an artifact.
P20 measured it directly at a long context:
  reserved 9.0 GiB -> **GPU KV cache size: 4,396,227 tokens** (max concurrency 16.77x at 262,144)
  => 9,663,676,416 / 4,396,227 = **~2,198 B/token**, NOT 35,320 B/token.

*** THE [KV] ENTRY ABOVE IS WRONG AND IS HEREBY WITHDRAWN. *** It concluded "1M does NOT fit
the 3.0bpw pack" from a single measurement taken at ctx 4096 (1 GiB -> 30,400 tokens). The two
readings disagree by 16x, and the ctx-4096 one does not generalise: at a small max_model_len
the per-group block granularity dominates the allocation, so the reported token capacity is
floor-limited rather than bytes-per-token limited. The long-context figure is the one that
governs long-context planning.
SAME ERROR CLASS AS [P12]'s withdrawn overhead model and [P3c]: generalising a law from a
measurement taken in the wrong regime. Measure in the regime you intend to run in.
(Note the DGX-Spark recipe's ~890 B/token was far closer to the truth than my 35.3 KB —
2,198 B/token is the same order. I over-corrected against it.)

### Consequence: 1M needs ~2.2 GiB of KV, and 9 GiB already holds 4.4M tokens
The weight-side conclusion in [P12] stands unchanged (3.0bpw needs ~157 GB of host RAM), but
the KV-side objection to 1M evaporates. Going straight to max-model-len 1,048,576 with KV
left at 9 GiB — one variable changed.
GPU headroom is now the binding constraint, not KV: gpu0 sits at 96,828 MiB against a
96,908 MiB cap (util 0.99), so KV must NOT be raised; weights 82.66 GiB + KV 9 GiB + ~3 GiB
of runtime workspace is already at the ceiling.

## [P20] 2026-09-14T05:44Z — SERVING AT 262,144 CONTEXT, 12.19 tok/s
  Total CPU offloaded parameters: 117.45 ; shmem flat 117.5 GiB ; avail 14 G
  [p9] wo_a reconstructed: 40 layers; compressor fused: 4; dropped 4
  Model loading took 82.66 GiB and 146.34 s ; endpoint ready after 195 s
  GPU KV cache size: 4,396,227 tokens ; Application startup complete
  /v1/models reports max_model_len = 262144
  smoke: FINISH stop, 22 tokens, 1.80 s, **12.19 tok/s**, coherent continuation
Decode is 6x the 2.03 tok/s seen at ctx 4096 (that figure was a 1-token EOS artifact, not a
real rate). gpu0 96,828 MiB, 48 C, 94.9 W — thermally fine, far below the 80 C guard.

## [P21] 2026-09-14T05:49Z — 1M: CUDA OOM on the KV buffer. I over-reserved KV by 5x.
  INFO [kv_cache_utils.py:2321] GPU KV cache size: 5,394,028 tokens,
       Maximum concurrency for 1,048,576 tokens per request: 5.14x
  worker/utils.py:429 allocate_kv_cache -> torch.zeros(buf_size, dtype=int8, device=cuda)
  torch.OutOfMemoryError: Tried to allocate 9.00 GiB. GPU 0 total capacity 95.01 GiB, of
  which 7.79 GiB is free; this process has 86.98 GiB in use (86.07 GiB by PyTorch).
Not a kernel OOM, not my RAM floor: a clean CUDA OOM on ONE contiguous 9 GiB allocation,
short by ~1.2 GiB. Everything before it was perfect — offload 117.45, wo_a 40 layers,
compressor fused 4, model loading 82.66 GiB in 144.98 s.

WHY 9 GiB FIT AT 256K BUT NOT AT 1M: the non-KV footprint grows with max_model_len (sparse
attention / Lightning-Indexer metadata buffers), so the process was already at 86.98 GiB
before the KV allocation, versus a 95.01 GiB capacity.

*** THE SIZING LINE HANDS ME THE EXACT NUMBER: ***
  9,663,676,416 B / 5,394,028 tokens = **~1,792 B/token AT 1M CONTEXT**
  a full 1,048,576-token window therefore needs only **~1.75 GiB**
I reserved 9 GiB — 5x more than required. With --max-num-seqs 1 there is no reason to reserve
concurrency headroom at all (the 5.14x concurrency it computed is wasted memory).
Note this ~1,792 B/token is close to the 2,198 B/token measured at 256K and to the DGX-Spark
recipe's ~890 B/token order — and nothing like the 35,320 B/token artifact from ctx 4096.
Three regimes, and only the long-context ones agree. The ctx-4096 reading was pure floor noise.

## [P22] plan — 1M with KV sized from measurement, not guesswork
--kv-cache-memory 9 GiB -> 3 GiB (3,221,225,472 B => ~1.8M tokens at 1,792 B/token, still
comfortably above the 1,048,576 window). Projected: 86.98 + 3.0 = ~90.0 GiB of 95.01 ->
~5 GiB of headroom. One variable changed from P21; everything else is the known-good P19/P20
configuration.

## [P22] 2026-09-14T05:54Z — *** 1,048,576 CONTEXT. SERVING. GOAL REACHED. ***
  INFO [kv_cache_utils.py:2321] GPU KV cache size: 1,797,989 tokens,
       Maximum concurrency for 1,048,576 tokens per request: **1.71x**
  INFO [uva.py:65] Total CPU offloaded parameters: 117.45
  [p9] wo_a reconstructed: 40 layers; compressor fused: 4; dropped 4
  INFO [model_runner.py:408] Model loading took 82.66 GiB memory and 145.66 seconds
  INFO:     Application startup complete.            endpoint ready after 195 s
  /v1/models -> **max_model_len = 1048576**
  smoke: FINISH length, 64 tokens, 4.77 s, **13.42 tok/s**, coherent on-topic prose
  steady: gpu0 93,670 MiB | 51 C | 168 W | shmem FLAT 117.5 GiB | avail 15 G
Thermals are far below the 80 C derate threshold, so the standing guard never engages.

### The user's requested topology, item by item
  6000 Pro (sm120) with partially fixed / partially streamed experts   ACHIEVED
      74.6 GiB of experts resident on the Blackwell, 117.45 GiB streamed over UVA
  1M KV cache                                                          ACHIEVED (1.71x margin)
  1xx GB DDR5 offload                                                  ACHIEVED (117.45 GiB)
  engram streamed from SSD                                             ACHIEVED (DSV41_ENGRAM_DISK=1)
  CMP 170HX fully resident                                             NOT POSSIBLE — see [FINDING]
      v4.1's last kv-sharing group is an indivisible 20 layers, and no v4.1 attention
      implementation exists for sm_80 (flash_mla absent; FlashInfer sparse MLA is sm90/sm120).
      Architectural, not a tuning failure. The dual-arch sm_80 exl3 image was still built and
      verified, and remains useful for models whose attention runs on Ampere (GLM-5.3).

### THE WORKING CONFIGURATION
  bash /root/ds41/p22_boot.sh   (image dsv41-flash-exl3-sm80120, sitecustomize_ds41.py 709 lines)
  PP=1, CUDA_VISIBLE_DEVICES=0, util 0.99, --enforce-eager, --block-size 64,
  --max-model-len 1048576 --max-num-seqs 1 --max-num-batched-tokens 2048,
  --kv-cache-dtype fp8 --kv-cache-memory 3221225472,
  --offload-backend uva --cpu-offload-gb 116 --cpu-offload-params w13_*/w2_*,
  VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1, DSV41_ENGRAM_DISK=1,
  --quantization exl3 --language-model-only --tokenizer-mode deepseek_v41

### THE TWO NUMBERS THAT DECIDED EVERYTHING
 1. pinning multiplier: host RAM = offload x 1.265 WITH pinning, x 1.000 without.
    Six runs died on this before it was measured (FIX G). It is the difference between fitting
    and not fitting on a 157 GB box.
 2. KV at 1M: ~1,792 B/token. Reserve ~1.75 GiB for a 1M window; I first asked for 9 GiB and
    OOM'd. With --max-num-seqs 1, concurrency headroom is pure waste.
Both were WRONG when estimated and RIGHT when measured, in the regime of interest.

### Still open / next
 * a genuine 1M FILL test (push ~1M real tokens through and measure TTFT + decode) — the
   capacity is proven, sustained throughput at depth is not. Analogous to the GLM 1M fill test.
 * decode is 13.4 tok/s at short context; expect it to fall as the window fills.
 * DSpark / MTP speculative decoding is untouched.

## [LONGCTX] 2026-09-14T05:59Z — validating throughput at DEPTH (capacity != performance)
Every request so far has been an 18-token prompt. The 1M capacity is proven (KV holds
1,797,989 tokens, 1.71x the window) but sustained prefill/decode at depth is NOT.
Staged /root/ds41/ds41_longctx.py and launched a 128K fill in the background
(/root/ds41/longctx_128k.log). Deliberately NOT issuing other requests while it runs:
--max-num-seqs 1 means anything else queues behind it and would confound the timing.

### OPEN QUESTION — an empty `content` on a short chat reply
A liveness probe ("Reply with exactly: ALIVE", max_tokens 12) came back with
`content = None`. The earlier arithmetic probe returned `content = '391'` with
`reasoning_content = None`, so content is clearly functional.
MOST LIKELY MY PROBE, NOT THE MODEL: that curl one-liner printed ONLY `content`, and with
--reasoning-parser deepseek_v41 a short reply can be routed to `reasoning_content` instead.
Same family as the earlier smoke-test artifact where /v1/completions on a chat-tuned model
emitted an immediate EOS and looked like a broken model.
NOT CONFIRMED EITHER WAY, and I am not calling the endpoint solid until it is: re-probe
printing the RAW response JSON (content, reasoning_content, finish_reason) once the fill test
finishes and the queue is free. If content is intermittently empty for real, that is a defect
worth knowing about before this is used for anything.

## [LONGCTX] RESULT 2026-09-14T06:07Z — 142,278-token request served. Prefill ~310 tok/s.
  target ~128,000 -> PROMPT_TOKENS: **142,278** ; COMPLETION_TOKENS: 32
  WALL: 457.8 s ; FINISH: length
  gpu0 95,832 MiB ; shmem FLAT 117.5 GiB ; avail 12 G ; peak 64 C (guard 80 C, never fired)
A real long-context request completed end to end with no error, no OOM, no thermal event.
That is the deepest this model has been exercised on this box.

*** DO NOT QUOTE THE 14,227 tok/s FIGURE. *** vLLM logged
  "Avg prompt throughput: 14227.3 tokens/s"
in a single 10 s window, which is an artifact of when its counter flushed at the end of the
prefill. The honest rate is the wall clock: 142,278 tokens / 457.8 s = **~311 tok/s sustained**.
Cross-check: chunked prefill at --max-num-batched-tokens 2048 means 70 chunks;
457.8 / 70 = 6.5 s per chunk = ~315 tok/s. The two agree, the log line does not.
CAUSE: every chunk streams routed experts from host RAM over PCIe (117.45 GiB offloaded), so
prefill is bandwidth-bound, not compute-bound.
EXTRAPOLATION: a full 1,048,576-token prompt would take roughly **~56 minutes** of prefill at
this rate. The 1M window is real and allocated; filling it is slow. Capacity and throughput
are different claims and should be reported separately.

### The empty-content question is NOT resolved — it got worse
This run: FINISH=length, and BOTH content AND reasoning_content are None, which kills my
earlier "the reasoning parser routed it to reasoning_content" hypothesis.
Pattern across all probes so far:
  finish=stop   (17x23, 256K continuation)  -> content PRESENT and correct
  finish=length (ALIVE max_tokens 12; this fill max_tokens 32) -> content None AND reasoning None
Consistent with the deepseek_v41 reasoning parser buffering an unterminated thinking block:
truncation by max_tokens leaves the block unclosed, so the parser emits neither field.
Testing that directly now the queue is free: the same question at max_tokens 256 (should
finish naturally -> content present) versus max_tokens 12 (should reproduce the None).
Until that comes back, the endpoint is "serves correct output when allowed to finish", with a
known rough edge on truncated replies. Not glossing over it.

## [LONGCTX] RESOLVED 2026-09-14T06:09Z — empty content is a TRUNCATION artifact, not a defect
Same prompt, only max_tokens differs. That is the whole experiment:
  max_tokens=256 -> finish=stop   content='The capital of France is Paris.'  reasoning=None
  max_tokens=12  -> finish=length content=None                              reasoning=None
So when the deepseek_v41 reasoning parser is cut off by max_tokens mid-block it emits NEITHER
field. Nothing is wrong with the model, the weights, or any of the nine fixes.
ALSO MY BUG: the message key is **`reasoning`**, not `reasoning_content`. Every earlier probe
printed `reasoning_content`, a field that does not exist in this response shape, so those Nones
were partly self-inflicted. Print `sorted(message.keys())` before trusting a field name.
PRACTICAL RULE: give chat requests an adequate max_tokens (finish=stop), or use
/v1/completions, or drop --reasoning-parser. A finish=length chat reply yields null content.
=> The endpoint is sound. The empty-content question raised in [LONGCTX] is CLOSED.

## [CMP-PROBE] 2026-09-14T07:2xZ — exl3 RUNS on sm_80, and FAST. The CMP is usable after all.
Two probes on cuda:1 (CMP 170HX, sm_80) while the 1M endpoint kept serving on cuda:0.

### 1. Correctness — /root/ds41/probe_sm80_exl3.py
Real pack tensors (layers.0.attn.wq_b, in 1280 -> out 32768, K=5):
  LinearEXL3 built ; reconstruct OK (1280,32768) finite
  exl3 forward OK (8,32768) ; max_abs_err 0.0039 ; **mean_rel_err 0.00057**
  VERDICT: PASS — exl3 executes CORRECTLY on sm_80.
The dual-arch build from [BUILD-sm80] is not merely present, it computes right answers.

### 2. Speed — /root/ds41/probe_sm80_speed.py (WARM, the 34.1 ms first call was JIT)
  rows=1   0.05 ms/call  -> **527 GB/s** effective weight read
  rows=8   0.07 ms/call  ->   359 GB/s
  rows=64  0.29 ms/call  ->    92 GB/s
  activation round trip (1 token, 5120 bf16 = 10 KiB): **0.082 ms**
Batch-1 decode is memory-bandwidth-bound, so the rows=1 figure is the one that governs.

*** THIS REVERSES THE PRACTICAL CONCLUSION OF [FINDING], WITHOUT CONTRADICTING IT. ***
[FINDING] said the CMP cannot host a LAYER — still true: the {20-39} kv-sharing group is
indivisible, and there is no sm_80 v4.1 ATTENTION. But experts are not attention. Routed
experts are exl3 GEMMs, and those demonstrably run on this card at 527 GB/s.
=> Use the CMP as an EXPERT RESIDENCY TIER, not a pipeline stage. Attention, indexer,
   compressor, engram and the head all stay on the Blackwell; only routed-expert weights move.

### Why this is strongly favourable
Those experts currently stream from HOST RAM over the Blackwell's Gen5 x16 link at ~42 GB/s.
Resident on the CMP they are read at ~527 GB/s — about **12x faster** — and the only new cost
is shipping activations over the CMP's Gen2 x1 link:
  0.082 ms/token round trip x ~12 layers = **~0.98 ms/token**, ~1% of a 77 ms/token decode.
This is the project's own rule paying off: ACTIVATIONS over slow links, never WEIGHTS.
Sizing: 63.4 GiB / ~4.92 GiB per layer = **~12 layers (~59 GiB)** on the CMP, which removes
roughly half of the current 117.45 GiB host offload.

### Interception points located (vllm_exl3/exl3.py)
  Exl3MoEMethod.create_weights (1832)  registers 10 params: w13_{trellis,suh,svh,mcg,mul1},
                                       w2_{trellis,suh,svh,mcg,mul1}
  process_weights_after_loading (1989) builds layer._exl3_inners / _exl3_ptrs / _exl3_fused_temps
  apply_exl3_experts(x, topk_ids, topk_weights, layer)  ->  pin_exl3_expert_map(layer, x2d.device)
                                       -> apply_exl3_fused_moe(x2d, ids, weights, layer, inners, ...)
FIX J must (a) move the 10 params of selected layers to cuda:1 BEFORE the pointer tables are
built, (b) ensure those tables are built for cuda:1, (c) marshal activations in apply, and
(d) exclude those layers from the host offloader so they are not ALSO UVA-mapped.
Checking now whether the tables bind to each tensor's device or to torch.cuda.current_device()
— that decides whether a device context manager is required.

## [FIX J] 2026-09-14T07:3xZ — design: CMP as an EXPERT tier (not a pipeline stage)
### Sizing, measured from the pack (not assumed)
  all 40 layers are MoE, 384 experts each, **4.762 GiB/layer**, 190.48 GiB total
  CMP 63.4 GiB - ~6 GiB headroom => **12 layers (57.1 GiB)** resident on the CMP
That removes ~57 of the 117.45 GiB currently streamed from host RAM.

### Why this does not contradict [FINDING]
[FINDING] stands: the CMP cannot host a LAYER (indivisible {20-39} kv group; no sm_80 v4.1
attention). FIX J never gives it a layer. Attention, indexer, compressor, engram and the head
all stay on the Blackwell; ONLY routed-expert weights move. Experts are exl3 GEMMs, proven on
sm_80 at mean_rel_err 0.00057 and ~527 GB/s.

### Two hooks, and one thing that needed NO patch
 J1 Exl3MoEMethod.create_weights -> move the 10 MoE params to cuda:1 while they are still
    EMPTY. Must be at create_weights, NOT process_weights_after_loading: the offloader runs
    during make_layers BEFORE weights load, so a later move would let 57 GiB land in host RAM
    first and then be copied off again.
    Device binding verified safe: build_exl3_fused_state:707 does
    `device = layer.w13_trellis.device` — from the TENSOR, not torch.cuda.current_device() —
    so _exl3_ptrs and _exl3_fused_temps follow the params to the CMP. pin_exl3_expert_map
    re-pins on a device change (its cache check includes `cached.device == device`).
 J3 Exl3MoEMethod.apply -> ship x / topk_weights / topk_ids to the CMP, result back.
 NO OFFLOADER PATCH: setting uva.py's own `_vllm_is_uva_offloaded` marker on the moved params
    makes its existing skip pass over them.
 REJECTED DESIGN: wrapping _maybe_offload_to_cpu with a per-MODULE device check. That is wrong
    — it is called per DECODER LAYER, and a decoder layer holds attention on cuda:0 and experts
    on cuda:1, so a per-module early return would also have skipped offloading the layer's other
    params. The marker is per-parameter and therefore correct.

### Expected effect
Those 12 layers' experts read at ~527 GB/s locally instead of ~42 GB/s over the Blackwell's
Gen5 link from host. Added cost: 0.082 ms/token round trip x 12 = ~0.98 ms/token (~1% of a
77 ms/token decode). Host offload should fall 117.45 -> ~60 GiB, which also buys back RAM.
Inert unless DS41_CMP_DEVICE is set, so every existing configuration is unaffected.

### Untestable by reading — these only surface at runtime
 * whether vLLM tolerates CUDA_VISIBLE_DEVICES=0,1 at TP=1/PP=1 without claiming device 1
 * whether the expert weight loader writes correctly into cuda:1 params
 * whether the fused exl3_moe kernel launches correctly with all its pointers on the CMP
 * whether P2P-disabled cross-device copies behave (they route through host; fine for 10 KiB)

## [P23] 2026-09-14T07:26Z — *** FIX J WORKS: THE CMP HOLDS AND COMPUTES 12 LAYERS OF EXPERTS ***
  [p9] FIX J armed: 12 layers -> cuda:1 (28,29,30,31,32,33,34,35,36,37,38,39)
  [p9] FIX J: layer N routed experts -> cuda:1 (10 params, 4.76 GiB)   x12  = 57.1 GiB
  **CMP = 59,922 MiB RESIDENT, util 32%, power 44 -> 66.6 W**   <- it computes, not just stores
  INFO [uva.py:65] Total CPU offloaded parameters: **60.31**  (was 117.45)
  avail **71-72 G** (was 15 G) ; shmem **60.4 G** (was 117.5 G)
  [p9] wo_a reconstructed: 40 layers; compressor fused: 4; dropped 4   <- rest of stack intact
  GPU KV cache size: 1,797,989 tokens ; max concurrency 1.71x at 1,048,576
  INFO: Application startup complete.        gpu0 93,596 MiB, 36% util, 52 C, 122 W
The fused exl3_moe kernel executes CROSS-DEVICE with no illegal access and no device mismatch.
Every number matched the derivation: 12 x 4.762 = 57.1 GiB placed, host offload 117.45 -> 60.31,
~56 GiB of DDR5 handed back.

### The CMP is now a real compute tier, which [FINDING] said was impossible for LAYERS
Both statements hold, and the distinction is the whole point:
  a LAYER on the CMP  -> impossible (indivisible {20-39} kv group; no sm_80 v4.1 attention)
  EXPERTS on the CMP  -> works (exl3 GEMMs, sm_80-capable, 527 GB/s local reads)
Attention, indexer, compressor, engram and the head all stayed on the Blackwell. Only routed
expert WEIGHTS moved, and only ACTIVATIONS cross the Gen2 x1 link.

### Cost paid, as predicted
Model loading 145.66 s -> **420.57 s**. The extra ~275 s is pushing 57 GiB across a 0.38 GB/s
H2D link; that is a one-time startup cost, not a serving cost.

### Two hooks, no forward-path patch, and one thing that needed no patch at all
J1 create_weights: move the 10 MoE params while still EMPTY (so the loader writes CPU->cuda:1
   directly and they never touch Blackwell VRAM) + set uva.py's own `_vllm_is_uva_offloaded`
   marker so the host offloader skips them. Device binding was already safe:
   build_exl3_fused_state:707 takes the device FROM THE TENSOR.
J3 apply: marshal x / topk_weights / topk_ids to the CMP, result back.
Inert unless DS41_CMP_DEVICE is set, so every earlier configuration is unaffected.

### NOT YET PROVEN — measuring now
The smoke line (FINISH stop, 1 token, TEXT '') is the KNOWN /v1/completions artifact, identical
to P19's; it says nothing about speed or correctness. Two things still to establish:
 * correctness with 12 of 40 layers computing on a different GPU — a silent numerical error
   there would still look like a healthy endpoint;
 * decode tok/s versus the P22 baseline of 13.42 tok/s. That is the number that decides whether
   FIX J is a win or merely a memory trade.

## [P23] VERIFIED 2026-09-14T07:35Z — cross-device MoE is CORRECT; decode gain is modest
### Correctness with 12 of 40 layers computing on the CMP
  "17 multiplied by 23"        -> '391'                                   CORRECT
  "capital of France / Japan"  -> 'Paris ... / Tokyo ...'                 CORRECT
A silent numerical error in the cross-device path would have looked like a healthy endpoint,
so this is the check that actually validates FIX J. It passes.

### Decode, stated honestly
  run 1: 128 tok in 9.15 s = 13.99 tok/s
  run 2: 128 tok in 8.63 s = 14.83 tok/s
  P22 baseline (no CMP, all experts host-streamed): 13.42 tok/s (single 64-token sample)
=> roughly **+4% to +10%**, which is INSIDE the noise band of this few samples.
MY OWN SCRIPT OVERSOLD IT: it printed "MEDIAN DECODE: 14.83" but with two samples
`sorted(r)[len(r)//2]` returns the HIGHER element — that is a max, not a median. Mislabelled
statistic, corrected here. Do not quote 14.83 as a median.

### So what did FIX J actually buy?
MEMORY, decisively: host offload 117.45 -> 60.31 GiB, avail 15 G -> 72 G, shmem 117.5 -> 60.4 G.
~57 GiB of DDR5 returned. That headroom is what buys larger KV, more context, or a higher-bpw
pack — none of which was affordable before.
DECODE, marginally. Expected: only 12 of 40 layers moved, attention and the other 28 layers are
unchanged, and the x1 activation round trip (~0.98 ms/token) eats part of the gain.

### The measurement that actually matters for the 1M goal
At 1M context PREFILL dominates (~56 min at 311 tok/s), and prefill is bandwidth-bound on
expert streaming — precisely what FIX J changes, swapping ~30% of expert reads from ~42 GB/s
host to ~527 GB/s local. Running the identical 128K fill (baseline: 142,278 tokens in 457.8 s
= ~311 tok/s) against P23 now. That number decides whether 1M is USABLE rather than merely
ALLOCATABLE.

## [P23] FILL 2026-09-14T07:35Z — prefill 311 -> 335 tok/s (+7.7%), and WHY it is not more
Identical 128K fill, same script, same prompt:
  P22 (no CMP): 142,278 tokens in 457.8 s = ~311 tok/s
  P23 (CMP)   : 142,278 tokens in **424.7 s = ~335 tok/s**   -> **+7.7%**
1M extrapolation: ~56 min -> **~52 min** of prefill.

### THE CARDS ALTERNATE — they do not overlap. This is the structural ceiling.
Per-card sampling during prefill (idx, util%, C, W):
  0, 100, 62, 179.11 | 1,  0, 51, 45.45
  0,   5, 62, 165.04 | 1, 95, 55, 69.36
  0,  10, 60, 137.20 | 1, 93, 54, 166.66
A single sequential forward runs layers 0-27 on the Blackwell, then 28-39 on the CMP, then
returns. There is NO pipelining: while the CMP computes, the Blackwell idles, and vice versa.
So the CMP's 527 GB/s advantage applies only to its 30% share of layers, and the x1 activation
hop serializes on top. That is why the gain is +7.7% and not ~+30%.
To do better one would need to OVERLAP the two - e.g. prefetch/compute on one card while the
other works, or split experts WITHIN a layer rather than by layer. Not attempted.

### Thermals under real dual-card load
  CMP peak 166.66 W (cap 200 W), 51-55 C   |  Blackwell peak ~255 W, 60-69 C
Both far below the 80 C derate. The guard never engaged.

### FIX J scorecard, honestly
  memory     : host offload 117.45 -> 60.31 GiB, avail 15 G -> 68-72 G   **decisive win**
  prefill    : 311 -> 335 tok/s                                          +7.7%
  decode     : 13.42 -> ~14 tok/s                                        +4-10%, noisy
  correctness: 391 / Paris / Tokyo                                       verified
  startup    : 145.7 s -> 420.6 s (57 GiB over a 0.38 GB/s link)         one-time cost
The headline result is that ~57 GiB of DDR5 came back, which is what buys headroom for a
higher-bpw pack or a larger KV budget. The throughput gain is real but modest, and bounded by
the alternation above.

## [P24] 2026-09-14T08:46Z — TOOL CALLING ENABLED. Client 400 fixed.
Client symptom:
  400 '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
A launch-flag gap, not a model or client problem: both flags are startup-only, so it needs a
restart. P24 = P23 + `--enable-auto-tool-choice --tool-call-parser deepseek_v41`. Nothing else.

### Finding the parser name WITHOUT guessing
Guessing between deepseek_v3 / v31 / v41 risks a startup failure or, worse, a server that
starts and silently MIS-PARSES every tool call. Sourced it authoritatively instead:
  vllm/tool_parsers/__init__.py lazy-registration table:
    "deepseek_v41": ("deepseekv41_engine_tool_parser", "DeepSeekV41EngineToolParser")
  (siblings: deepseek_v3, deepseek_v31, deepseek_v32, deepseek_v4 — a v41-specific parser exists)
Note the module path in this build is **vllm/tool_parsers/**, NOT the upstream
vllm.entrypoints.openai.tool_parsers — importing the latter raises ModuleNotFoundError, and my
first two attempts to enumerate parsers failed for exactly that reason. `vllm serve --help |
grep tool-call-parser` also returned nothing here; the registration table is the reliable source.
The parsers carry NO register_module decorator — registration is the table, not the file.

### Cross-checked against the pack's own chat template
  chat_template.jinja:24  `{%- set dsml = "｜DSML｜" -%}`
  emits: <｜DSML｜ calls> / <｜DSML｜ invoke name="..."> /
         <｜DSML｜ parameter name="..." string="true|false">...</｜DSML｜ parameter>
So the template does support tools (12 "tools", 8 "tool_calls" mentions) and the v41 parser is
the one written for that markup. Name and format agree.

### VERIFIED THROUGH THE PUBLIC ENDPOINT (Caddy + auth + parser + model in one shot)
  POST https://ds4.example.invalid:8443/v1/chat/completions  tools=[get_weather] tool_choice=auto
  -> HTTP 200 in 12.4 s
     finish_reason : tool_calls
     tool_calls    : 1
     name          : get_weather
     arguments     : {"city": "Reykjavik", "unit": "c"}     <- enum honoured
     content       : None
  VERDICT: PASS - tool call parsed (not raw <｜DSML｜> markup leaking into content)
The test deliberately distinguishes PARSED tool_calls from raw markup in `content`: the latter
would mean the flags took but the parser did not match the template — a different fix.

### Everything else survived the restart
  [p9] FIX J armed: 12 layers -> cuda:1 ; Total CPU offloaded parameters: 60.31
  wo_a reconstructed: 40 layers; compressor fused: 4 ; GPU KV cache size: 1,797,989 tokens
  gpu0 93,534 MiB 52 C | CMP 60,022 MiB 48 C
**/root/ds41/p24_boot.sh is now the canonical launcher** (p23 = no tool calls, p22 = no CMP tier).

### Correction (2026-09-22): the Blackwell H2D figure

Entries above use ~42 GB/s for host-to-device over the RTX PRO 6000's Gen5 x16 link. That
number did not reproduce. Measured idle-gated (service at `Running: 0` / `Waiting: 0`), three
trials per size:

    pinned   H2D   16-256 MiB transfers    56-57 GB/s
    pinned   H2D   1-2 GiB transfers       49-54 GB/s
    pinned   H2D   <=1 MiB transfers       29-32 GB/s   per-copy overhead dominates
    pinned   H2D   2-8 MiB transfers       32-52 GB/s   unstable run to run
    pageable H2D   any size                31-33 GB/s

No mode or size gives a stable 42; it is most plausibly one sample from the unstable 2-8 MiB
band. The FIX J reasoning is unaffected: 527 GB/s local exl3 weight reads against at most
~56 GB/s across the link is still roughly a 10x gap. The entries above are left as written.
