#!/usr/bin/env bash
# P22 — 1M context with KV sized from P21's own measurement. ONE variable vs p21: KV size.
#   --kv-cache-memory 9 GiB -> 3 GiB
# P21 reported: 9 GiB = 5,394,028 tokens at ctx 1,048,576 => ~1,792 B/token.
#   a full 1M window needs ~1.75 GiB; 3 GiB gives ~1.8M tokens.
#   P21 died allocating ONE contiguous 9 GiB buffer with only 7.79 GiB free (process already
#   at 86.98 GiB of 95.01). 86.98 + 3.0 = ~90.0 GiB -> ~5 GiB headroom.
# --max-num-seqs is 1, so reserving concurrency headroom (P21 computed 5.14x) is pure waste.
set -u
R=/root/ds41/diffbot-recipe/recipe
PACK=/root/workspace/DeepSeek-V4.1-Flash-EXL3-3.0bpw
SC=/root/ds41/sitecustomize_ds41.py
LOG=/root/ds41/p22_boot.log
[ -f "$SC" ] || { echo "FATAL: missing patch stack $SC"; exit 1; }
IMG=dsv41-flash-exl3-sm80120
docker image inspect "$IMG" >/dev/null 2>&1 || IMG=dsv41-flash-exl3-sm120
: > $LOG
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
echo "P22 image=$IMG ctx=1048576 KV=3GiB util=0.99 offload=116 DISABLE_PIN_MEMORY=1 $(date -Is)" >> $LOG
free -g | awk '/^Mem/{printf "  pre-launch MemTotal=%sG avail=%sG\n",$2,$7}' >> $LOG
docker rm -f ds41-p22 >/dev/null 2>&1
( for i in $(seq 1 9000); do
    AV=$(free -g | awk '/^Mem/{print $7}')
    SH=$(awk '/^Shmem:/{printf "%.1f", $2/1048576}' /proc/meminfo)
    G0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n 1p)
    T0=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sed -n 1p)
    printf "  [mon %s] avail=%sG | shmem=%sG | gpu0=%sMiB %sC\n" "$(date +%H:%M:%S)" "$AV" "$SH" "$G0" "$T0" >> "$LOG"
    if [ "${AV:-99}" -lt 6 ] 2>/dev/null; then
      echo "  *** RAM FLOOR (avail=${AV}G, shmem=${SH}G) -> controlled abort ***" >> "$LOG"
      docker kill ds41-p22 >/dev/null 2>&1
      break
    fi
    sleep 3
  done ) & MON=$!
docker run --rm --name ds41-p22 --gpus all --network host --shm-size 8g \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=0 \
  -e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 \
  -e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_DISK_THREADS=32 \
  -e VLLM_PLUGINS=vllm_exl3 -e VLLM_EXL3_REQUIRE_UVA_EXPERTS=1 \
  -v /root/workspace:/root/workspace:ro \
  -v "$SC":/usr/lib/python3.12/sitecustomize.py:ro \
  -v "$R/patches/flashinfer/sparse_mla_sm120_prefill.cu":/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/sparse_mla_sm120_prefill.cu:ro \
  -v "$R/patches/flashinfer/empty-aot":/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120:ro \
  -v /root/ds41/cache:/root/.cache \
  -e TORCH_CUDA_ARCH_LIST="12.0a" -e FLASHINFER_CUDA_ARCH_LIST=12.0f -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint vllm "$IMG" serve "$PACK" \
  --tensor-parallel-size 1 \
  --host 127.0.0.1 --port 8111 \
  --max-model-len 1048576 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --kv-cache-dtype fp8 --kv-cache-memory 3221225472 \
  --gpu-memory-utilization 0.99 \
  --offload-backend uva --cpu-offload-gb 116 \
  --cpu-offload-params w13_trellis w13_suh w13_svh w2_trellis w2_suh w2_svh \
  --kernel-config "{\"enable_flashinfer_autotune\":false,\"enable_jit_warmup\":false}" \
  --block-size 64 --quantization exl3 --language-model-only \
  --attention-config "{\"indexer_kv_dtype\":\"mxfp4\"}" \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --served-model-name ds41 --trust-remote-code --enforce-eager \
  >> $LOG 2>&1
rc=$?
kill $MON 2>/dev/null
echo "P22 EXIT rc=$rc $(date -Is)" >> $LOG
dmesg -T 2>/dev/null | grep -ai "Killed process" | tail -1 >> $LOG
