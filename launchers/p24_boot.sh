#!/usr/bin/env bash
# P24 — P23 plus tool calling. Two added flags, nothing else changed:
#   --enable-auto-tool-choice --tool-call-parser deepseek_v41
# Parser name is authoritative, from vllm/tool_parsers/__init__.py:
#   "deepseek_v41": ("deepseekv41_engine_tool_parser", "DeepSeekV41EngineToolParser")
# It matches the pack's chat template, which emits dsml = "｜DSML｜":
#   <｜DSML｜ calls> / <｜DSML｜ invoke name="..."> / <｜DSML｜ parameter name="..." string="...">
# Client symptom this fixes:
#   400 '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
set -u
R=/root/ds41/diffbot-recipe/recipe
PACK=/root/workspace/DeepSeek-V4.1-Flash-EXL3-3.0bpw
SC=/root/ds41/sitecustomize_ds41.py
LOG=/root/ds41/p24_boot.log
[ -f "$SC" ] || { echo "FATAL: missing patch stack $SC"; exit 1; }
IMG=dsv41-flash-exl3-sm80120
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: need the dual-arch image"; exit 1; }
: > $LOG
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
echo "P24 image=$IMG ctx=1048576 KV=3GiB offload=60 CMP=cuda:1 layers=28-39 +toolcalls $(date -Is)" >> $LOG
free -g | awk '/^Mem/{printf "  pre-launch MemTotal=%sG avail=%sG\n",$2,$7}' >> $LOG
docker rm -f ds41-p24 >/dev/null 2>&1
( for i in $(seq 1 9000); do
    AV=$(free -g | awk '/^Mem/{print $7}')
    SH=$(awk '/^Shmem:/{printf "%.1f", $2/1048576}' /proc/meminfo)
    read G0 G1 <<<"$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    read T0 T1 <<<"$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr '\n' ' ')"
    printf "  [mon %s] avail=%sG shmem=%sG | gpu0=%sMiB %sC | CMP=%sMiB %sC\n" \
      "$(date +%H:%M:%S)" "$AV" "$SH" "$G0" "$T0" "$G1" "$T1" >> "$LOG"
    if [ "${AV:-99}" -lt 6 ] 2>/dev/null; then
      echo "  *** RAM FLOOR (avail=${AV}G) -> controlled abort ***" >> "$LOG"
      docker kill ds41-p24 >/dev/null 2>&1; break
    fi
    sleep 3
  done ) & MON=$!
docker run --rm --name ds41-p24 --gpus all --network host --shm-size 8g \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=0,1 \
  -e DS41_CMP_DEVICE=cuda:1 -e DS41_CMP_LAYERS=28-39 \
  -e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 \
  -e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_DISK_THREADS=32 \
  -e VLLM_PLUGINS=vllm_exl3 -e VLLM_EXL3_REQUIRE_UVA_EXPERTS=1 \
  -v /root/workspace:/root/workspace:ro \
  -v "$SC":/usr/lib/python3.12/sitecustomize.py:ro \
  -v "$R/patches/flashinfer/sparse_mla_sm120_prefill.cu":/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/sparse_mla_sm120_prefill.cu:ro \
  -v "$R/patches/flashinfer/empty-aot":/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120:ro \
  -v /root/ds41/cache:/root/.cache \
  -e TORCH_CUDA_ARCH_LIST="8.0;12.0a" -e FLASHINFER_CUDA_ARCH_LIST=12.0f -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_P2P_DISABLE=1 -e NCCL_CUMEM_ENABLE=0 \
  --entrypoint vllm "$IMG" serve "$PACK" \
  --tensor-parallel-size 1 \
  --host 127.0.0.1 --port 8111 \
  --max-model-len 1048576 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --kv-cache-dtype fp8 --kv-cache-memory 3221225472 \
  --gpu-memory-utilization 0.99 \
  --offload-backend uva --cpu-offload-gb 60 \
  --cpu-offload-params w13_trellis w13_suh w13_svh w2_trellis w2_suh w2_svh \
  --kernel-config "{\"enable_flashinfer_autotune\":false,\"enable_jit_warmup\":false}" \
  --block-size 64 --quantization exl3 --language-model-only \
  --attention-config "{\"indexer_kv_dtype\":\"mxfp4\"}" \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 \
  --served-model-name ds41 --trust-remote-code --enforce-eager \
  >> $LOG 2>&1
rc=$?
kill $MON 2>/dev/null
echo "P24 EXIT rc=$rc $(date -Is)" >> $LOG
dmesg -T 2>/dev/null | grep -ai "Killed process" | tail -1 >> $LOG
