#!/usr/bin/env bash
# Rebuild the serving image with exllamav3 kernels for BOTH arches, so the CMP 170HX
# (sm_80) can execute exl3 modules and become the resident tier the topology needs.
#
# WHY: the current image's exllamav3_ext has 112 cubins, ALL sm_120 (cuobjdump -lelf),
# because it was built with TORCH_CUDA_ARCH_LIST=12.0a. But the plugin itself declares
# Exl3Config.get_min_capability() == 80 ("LinearEXL3 uses CUDA >= Ampere"), and at the
# pinned exllamav3 ref 5be88657 there are NO Hopper/Blackwell-only intrinsics
# (no wgmma / tcgen05 / cp.async.bulk / OMMA / e2m1 / blockscaled) and every
# __CUDA_ARCH__ guard is a downward branch (<750, ==860, >890, >=800). setup.py and
# CMakeLists carry no hardcoded gencode, so torch's cpp_extension just consumes
# TORCH_CUDA_ARCH_LIST.
#
# SAFETY: this writes a NEW tag and never touches dsv41-flash-exl3-sm120, so the
# working image survives if the dual-arch build fails or regresses.
#
# Mirrors build.sh's NOBUILDKIT path (docker run + docker commit); BuildKit hung on
# this host. MAX_JOBS is the real core count here, not the recipe's hardcoded 48.
# NOT `set -e`: with it, a failing `docker run` aborts the script before the bare
# `rc=$?` below can run, making the "BUILD FAILED -> NOT committing" guard
# unreachable and risking a commit of a broken image. Failures are checked explicitly.
set -uo pipefail

TAG=${TAG:-dsv41-flash-exl3-sm80120}
ARCHES=${ARCHES:-"8.0;12.0a"}
RECIPE=${RECIPE:-/root/ds41/diffbot-recipe/recipe/third_party/sfxnz-recipe}
REPO=${REPO:-/root/ds41/diffbot-recipe/recipe}
MAX_JOBS=${MAX_JOBS:-$(nproc)}
LOG=${LOG:-/root/ds41/build_sm80.log}

BASE=$(sed -n 's/^FROM //p' "$REPO/docker/Dockerfile.sm120")
VLLM_EXL3_REF=$(sed -n 's/^ARG VLLM_EXL3_REF=//p' "$REPO/docker/Dockerfile.sm120")
EXLLAMAV3_REF=$(sed -n 's/^ARG EXLLAMAV3_REF=//p' "$REPO/docker/Dockerfile.sm120")

echo "TAG=$TAG ARCHES=$ARCHES MAX_JOBS=$MAX_JOBS"
echo "BASE=$BASE"
echo "VLLM_EXL3_REF=$VLLM_EXL3_REF"
echo "EXLLAMAV3_REF=$EXLLAMAV3_REF"
[ -d "$RECIPE/docker/patch" ] || { echo "FATAL: no patch dir at $RECIPE/docker/patch"; exit 1; }

: > "$LOG"
C=dsv41-img-build-sm80
docker rm -f $C >/dev/null 2>&1 || true

docker run --name $C \
  -e VLLM_EXL3_REF="$VLLM_EXL3_REF" -e EXLLAMAV3_REF="$EXLLAMAV3_REF" \
  -e ARCHES="$ARCHES" -e MAX_JOBS="$MAX_JOBS" \
  --entrypoint bash -v "$RECIPE/docker/patch":/mnt/patch:ro "$BASE" -lc '
set -euxo pipefail
cp -r /mnt/patch /opt/dsv41-patch
python3 /opt/dsv41-patch/apply_engram_disk.py \
  --engram /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py \
  --weights /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py
apt-get update && apt-get install -y --no-install-recommends git libcusparse-dev-13-0 libcusolver-dev-13-0
git clone https://github.com/vcruz305/vllm-exl3.git /opt/vllm-exl3 && git -C /opt/vllm-exl3 checkout "$VLLM_EXL3_REF"
python3 /opt/dsv41-patch/fix_vllm_exl3_setup.py /opt/vllm-exl3/setup.py
VLLM_EXL3_NO_CUDA=1 python3 -m pip install --no-build-isolation --no-deps /opt/vllm-exl3
git clone https://github.com/turboderp-org/exllamav3.git /opt/exllamav3 && git -C /opt/exllamav3 checkout "$EXLLAMAV3_REF"
MAX_JOBS=$MAX_JOBS TORCH_CUDA_ARCH_LIST="$ARCHES" python3 -m pip install --no-build-isolation --no-deps /opt/exllamav3
python3 -c "import torch, exllamav3, exllamav3_ext; assert hasattr(exllamav3_ext, \"exl3_moe\")"
# PROVE both arches are present before committing (the whole point of this build).
python3 - <<PYEOF
import glob, subprocess, collections, sys
so = glob.glob("/usr/local/lib/python3.12/dist-packages/exllamav3_ext*.so")
assert so, "exllamav3_ext .so not found"
out = subprocess.run(["cuobjdump","-lelf",so[0]], capture_output=True, text=True).stdout
c = collections.Counter()
for line in out.splitlines():
    for tok in line.split("."):
        if tok.startswith("sm_"):
            c[tok] += 1
print("CUBIN ARCHES:", dict(c))
# Match by PREFIX, not exact key: nvcc emits the arch-specific token "sm_120a"
# (from TORCH_CUDA_ARCH_LIST=12.0a), so `c.get("sm_120")` is None and an exact
# check rejects a perfectly good build. That is exactly what happened on the
# first run: CUBIN ARCHES {'sm_120a': 112, 'sm_80': 112} was refused.
assert any(k.startswith("sm_80") for k in c), "NO sm_80 cubins -> the CMP still cannot run exl3"
assert any(k.startswith("sm_12") for k in c), "NO sm_12x cubins -> the Blackwell would regress"
PYEOF
rm -rf /opt/exllamav3 /opt/vllm-exl3 /var/lib/apt/lists/*
cp /opt/dsv41-patch/sitecustomize.py /usr/lib/python3.12/sitecustomize.py' >> "$LOG" 2>&1
rc=$?

if [ $rc -ne 0 ]; then
  echo "BUILD FAILED rc=$rc — NOT committing. See $LOG"
  tail -30 "$LOG"
  exit $rc
fi

docker commit --change 'ENV VLLM_PLUGINS=vllm_exl3' --change 'ENV DSV41_ENGRAM_DISK=1' \
  --change 'ENTRYPOINT ["vllm","serve"]' --change 'CMD []' $C "$TAG"
docker rm $C >/dev/null
echo "BUILT $TAG $(date -Is)"
grep -a "CUBIN ARCHES" "$LOG" || true
