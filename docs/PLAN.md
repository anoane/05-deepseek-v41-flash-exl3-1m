# DeepSeek-V4.1-Flash EXL3 3.0bpw — bring-up plan
Target topology (user-specified):
  RTX PRO 6000 (sm120, 95.6 GiB) : partially resident + partially STREAMED experts + 1M KV cache
  CMP 170HX    (sm80,  64.0 GiB) : FULLY RESIDENT experts only (PCIe Gen2 x1 = 0.38 GB/s, never stream)
  DDR5                           : 1xx GB UVA expert offload
  SSD                            : Engram streamed (190 GiB, shards 47/48)
Goal: serving at **1M context**.

## Verified starting facts (measured this session — do not re-derive)
- pack: /root/workspace/DeepSeek-V4.1-Flash-EXL3-3.0bpw  204.12 GiB, 41 shards, 192,452 keys
  routed_experts 196.83 GiB (96.42%) | non-routed only 7.30 GiB
- engram: /root/workspace/DeepSeek-V4.1-Flash-engram  190 GiB (shards 47+48, 94.56 each)
- KV: native V4.1 FP4 ~890 B/token -> ~5.15 GiB at 1M. Do NOT force fp8 KV.
- RAM: MemTotal 142.0 GB, swap 0, overcommit_memory=1 (CommitLimit advisory).
  PROVEN: 117.2 GB pinned OK (8.5 GB left). 110.7 GiB is the safer target.
- PCIe measured under load: Blackwell Gen5 x16 H2D 42.15 / D2H 56.43 GB/s
                            CMP      Gen2 x1  H2D  0.38 / D2H  0.41 GB/s
- PP=2 across these two heterogeneous GPUs PROVEN to run under vLLM (2026-09-12, GLM AWQ).
- exllamav3 CANNOT load this (no DeepseekV41 in any ref). vLLM owns the graph; vllm-exl3 owns EXL3 experts.

## Known blocker: vLLM issue #56702 — V4.1 will not start on SM120
  (1) prefill SWA width ignores --language-model-only -> allocates 1152 cols -> no decode kernel
  (2) 3 sites treat sm120 like sm100 (128B blocks) but FlashInfer sm120 kernel is 64B
  (3) FlashInfer lacks a DSV4 prefill template for extra_page_block_size == 32  <- terminal
  Status: OPEN. diffbot recipe claims "sm_120 fixes for vLLM and FlashInfer" -> harvest those.

## Phases (stop at the first failure, fix, log, continue)
P1 RUNTIME
   1.1 pull vllm/vllm-openai:deepseekv41-flash-0909 (day-0 V4.1 image)
   1.2 fetch diffbot recipe (non-weight files only) -> sm120 vLLM + FlashInfer patches
   1.3 build overlay image ds41-exl3-sm120: + vllm-exl3 (pin d3cfd394) + exllamav3 (pin 5be88657)
P2 FIRST BOOT (minimal): --language-model-only, ctx 4096, NO offload, NO engram, TP1, eager.
   Success = DeepseekV41ForCausalLM instantiates and weights load.
P3 ENGRAM FROM SSD: DSV41_ENGRAM_DISK=1 / VLLM_ENGRAM_MODEL_DIR -> engram dir. Verify no RAM blowup.
P4 UVA EXPERT OFFLOAD to DDR5:
   VLLM_EXL3_REQUIRE_UVA_EXPERTS=1 --offload-backend uva --cpu-offload-gb N
   --cpu-offload-params w13_trellis w13_suh w13_svh w2_trellis w2_suh w2_svh
P5 SECOND GPU: --pipeline-parallel-size 2 so the CMP holds resident experts (activations only over x1).
   NOTE: PP may disable DSpark. Accept that; DSpark is optional.
P6 SCALE CONTEXT to 1M (--max-model-len 1048576), CUDA graphs on if it fits.
P7 VERIFY: generation correctness, throughput, VRAM/RAM peaks, 1M fill. Document.

## Rules for this run
- Every step + every fix goes in RUNLOG.md with the exact command and the exact error.
- Never stream weights to the CMP. Activations only.
- Watch RAM: swap is 0. Pinned allocations cannot be reclaimed.
- glm53-sec stays DOWN (user instruction) — it is disabled, do not re-enable.
