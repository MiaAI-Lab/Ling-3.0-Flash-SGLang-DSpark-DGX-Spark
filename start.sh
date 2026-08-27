#!/usr/bin/env bash
# =============================================================================
#  start.sh — Ling-3.0-flash on one DGX Spark (GB10 / SM121, TP1)
#
#  Default: official GB10 MXFP4 + DSPARK cell from sgl-project/sglang#36364
#           (open cookbook PR, measured on a real GB10). Public LMSYS image —
#           no private GitHub clone required.
#
#  QUANT:
#    mxfp4 — inclusionAI/Ling-3.0-flash-fp4 + --moe-runner-backend flashinfer_mxfp4
#    int4  — inclusionAI/Ling-3.0-flash-int4 (previous proven Spark path)
#
#  Speculative decoding (SPEC_ALGO):
#    dspark — DSPARK with the external inclusionAI/Ling-3.0-flash-dspark draft
#             (1.36B, 5 layers; needs a recent SGLang → LMSYS Ling image).
#             Verify window 9 (block size 8) with the KDA ReplaySSM ring
#             pinned to --linear-replayssm-cache-len 32.
#    nextn  — built-in MTP layer (no extra checkpoint)
#    off    — no speculative decoding (high-throughput)
# =============================================================================
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
# External DSpark draft checkpoint for --speculative-algorithm DSPARK
# https://huggingface.co/inclusionAI/Ling-3.0-flash-dspark
DSPARK_MODEL_ID="inclusionAI/Ling-3.0-flash-dspark"
DSPARK_REVISION="${DSPARK_REVISION:-8e5d9988c9b09de13f1f7c9d999ff2bfa533a149}"
# Public images (new users need one of these — never requires inclusionAI/sglang_ling_v3):
#   1) Official LMSYS Ling-3.0 runtime:     lmsysorg/sglang:dev-Ling-3.0-flash
#   2) Spark-tuned (baked ling_v3_support): ghcr.io/miaai-lab/ling-3.0-flash-sglang-dgx-spark:ling_v3_support
#      INT4 + NEXTN/off only — predates DSPARK and FlashInfer MXFP4.
# Source rebuild only if you set IMAGE to a bare base (e.g. nvcr.io/nvidia/pytorch:26.01-py3)
# and FORCE_SOURCE_BUILD=1 (needs a public SGLANG_REPO mirror; official fork is often private).
DEFAULT_SPARK_IMAGE="ghcr.io/miaai-lab/ling-3.0-flash-sglang-dgx-spark:ling_v3_support"
DEFAULT_PUBLIC_IMAGE="lmsysorg/sglang:dev-Ling-3.0-flash"

# ---- Quantization (QUANT=mxfp4 default | int4) ------------------------------
# fp4 is accepted as an alias for mxfp4 (Hugging Face repo is *-fp4).
QUANT="${QUANT:-mxfp4}"
case "${QUANT}" in
  mxfp4|fp4)
    QUANT="mxfp4"
    MODEL_ID="inclusionAI/Ling-3.0-flash-fp4"
    MODEL_REVISION="${MODEL_REVISION:-3bae1cf4011b48475b2cc038fff283af49053ebc}"
    ;;
  int4)
    QUANT="int4"
    MODEL_ID="inclusionAI/Ling-3.0-flash-int4"
    MODEL_REVISION="${MODEL_REVISION:-7a27e9eb8179b2c2eb71eb214f0dab14ec6a63f2}"
    ;;
  *)
    echo "FATAL: QUANT must be 'mxfp4' or 'int4' (got: ${QUANT})"
    exit 1
    ;;
esac

# ---- Speculative decoding selection -----------------------------------------
# SPEC_ALGO=dspark (default) | nextn | off. ENABLE_NEXTN is honored as a legacy
# override (ENABLE_NEXTN=0 -> off, otherwise nextn) when SPEC_ALGO is unset.
if [[ -n "${SPEC_ALGO:-}" ]]; then
  :
elif [[ "${ENABLE_NEXTN:-}" == "0" ]]; then
  SPEC_ALGO="off"
elif [[ -n "${ENABLE_NEXTN:-}" ]]; then
  SPEC_ALGO="nextn"
else
  SPEC_ALGO="dspark"
fi
case "${SPEC_ALGO}" in
  dspark|nextn|off) ;;
  *)
    echo "FATAL: SPEC_ALGO must be 'dspark', 'nextn' or 'off' (got: ${SPEC_ALGO})"
    exit 1
    ;;
esac
# KDA ReplaySSM ring length for DSPARK: power of two, >= 2x the verify window
# (block size 8 -> window 9). Cookbook recipe pins 32.
REPLAYSSM_CACHE_LEN="${REPLAYSSM_CACHE_LEN:-32}"

# Official GB10 MXFP4 cell uses 0.85; INT4 stays at the safer 0.75 Spark default.
if [[ -z "${MEM_FRACTION_STATIC:-}" ]]; then
  if [[ "${QUANT}" == "mxfp4" ]]; then
    MEM_FRACTION_STATIC="0.85"
  else
    MEM_FRACTION_STATIC="0.75"
  fi
fi

if [[ -z "${IMAGE:-}" ]]; then
  # MXFP4 needs FlashInfer CUTLASS MXFP4; DSPARK needs dspark_components.
  # Both live in the LMSYS Ling image. GHCR ling_v3_support is INT4+NEXTN only.
  if [[ "${USE_LMSYS_IMAGE:-0}" == "1" || "${SPEC_ALGO}" == "dspark" || "${QUANT}" == "mxfp4" ]]; then
    IMAGE="${DEFAULT_PUBLIC_IMAGE}"
  else
    IMAGE="${DEFAULT_SPARK_IMAGE}"
  fi
fi
CONTAINER_NAME="${CONTAINER_NAME:-ling-3.0-flash}"
# Bind address for sglang serve / launch_server (--host). Docker uses --network host.
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
CTX="${CTX:-262144}"
WORK_DIR="$(pwd)"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
PID_FILE="${WORK_DIR}/.sglang.pid"
LOG_FILE="${WORK_DIR}/.sglang.log"
BOOTSTRAP_SCRIPT="/tmp/ling-bootstrap.sh"
FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${HOME}/.cache/flashinfer}"
# Host-side SGLang tree (optional). Only bind-mounted when already installed so an
# empty host dir does not hide a prebaked /opt/sglang-persist inside the image.
SGLANG_PERSIST_DIR="${SGLANG_PERSIST_DIR:-${WORK_DIR}/.sglang-persist}"
READY_URL="http://127.0.0.1:${PORT}/v1/models"
SGLANG_BRANCH="${SGLANG_BRANCH:-ling_v3_support}"
# Only used when FORCE_SOURCE_BUILD=1. inclusionAI/sglang_ling_v3 is often private.
SGLANG_REPO="${SGLANG_REPO:-https://github.com/inclusionAI/sglang_ling_v3.git}"
FORCE_SOURCE_BUILD="${FORCE_SOURCE_BUILD:-0}"

# ---- Argument parsing -------------------------------------------------------
DOWNLOAD_ONLY=false
case "${1:-}" in
  --download-only)
    DOWNLOAD_ONLY=true
    shift
    ;;
  -h|--help)
    echo "Usage: $0 [--download-only]"
    echo ""
    echo "  --download-only    Download the model to ~/.cache/huggingface/hub then exit"
    echo "                     without starting SGLang."
    echo ""
    echo "  Environment variables:"
    echo "    QUANT                  mxfp4 (default, official GB10 cell) | int4"
    echo "    SPEC_ALGO              Speculative decoding: dspark (default) | nextn | off"
    echo "                           dspark = DSPARK with the inclusionAI/Ling-3.0-flash-dspark"
    echo "                           draft (defaults to the LMSYS image, needs recent SGLang)"
    echo "    REPLAYSSM_CACHE_LEN    --linear-replayssm-cache-len for DSPARK (default: 32)"
    echo "    ENABLE_NEXTN           Legacy: 1 -> SPEC_ALGO=nextn, 0 -> SPEC_ALGO=off"
    echo "    PORT                   Server port (default: 8888)"
    echo "    HOST                   Bind address for SGLang (default: 0.0.0.0)"
    echo "    CTX                    Context length (default: 262144 / 256k)"
    echo "    MEM_FRACTION_STATIC    SGLang static mem fraction (default: 0.85 mxfp4 / 0.75 int4)"
    echo "    MAX_RUNNING_REQUESTS   (default: 6 concurrent)"
    echo "    MAX_MAMBA_CACHE_SIZE   (default: 32 on single Spark)"
    echo "    KV_CACHE_DTYPE         (default: fp8_e4m3; set empty to omit flag)"
    echo "    DOCKER_MEMORY          optional docker --memory (e.g. 100g); also sets --memory-swap"
    echo "    IMAGE                  Docker image (default: lmsysorg/sglang:dev-Ling-3.0-flash)"
    echo "    USE_LMSYS_IMAGE        set to 1 to force the LMSYS Ling image"
    echo "    MODEL_REVISION         Pin target snapshot (defaults per QUANT)"
    echo "    DSPARK_REVISION        Pin draft snapshot"
    echo "    HF_TOKEN               Hugging Face token for gated models"
    echo "    FORCE_SOURCE_BUILD     set to 1 to git-clone SGLANG_REPO into the container"
    echo "                           (not needed for public prebuilt images)"
    echo "    SGLANG_REPO            Git URL if FORCE_SOURCE_BUILD=1"
    echo "    GITHUB_TOKEN           Optional; private git clone / private GHCR pull"
    exit 0
    ;;
  "")
    ;;
  *)
    echo "Unknown argument: $1"
    echo "Usage: $0 [--download-only]"
    exit 1
    ;;
esac

# ---- Prerequisite checks ----------------------------------------------------
command -v docker >/dev/null 2>&1 || { echo "FATAL: docker is required"; exit 1; }
command -v curl   >/dev/null 2>&1 || { echo "FATAL: curl is required";   exit 1; }

# ---- Preflight checks (fail with a clear message, not a cryptic error) ------
# Docker daemon
if ! docker info >/dev/null 2>&1; then
  echo "FATAL: Docker daemon is not reachable (docker info failed)."
  exit 1
fi

# NVIDIA driver on the host
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "FATAL: No NVIDIA driver found (nvidia-smi -L failed)."
  echo "       Install the NVIDIA driver for this GPU before continuing."
  exit 1
fi

# NVIDIA container runtime (required for 'docker run --gpus all')
if ! command -v nvidia-container-runtime >/dev/null 2>&1 \
  && ! command -v nvidia-container-runtime-hook >/dev/null 2>&1 \
  && ! docker info 2>/dev/null | grep -qi 'nvidia'; then
  echo "FATAL: NVIDIA container runtime not found — '--gpus all' will fail."
  echo "       Install nvidia-container-toolkit, e.g.:"
  echo "         sudo apt-get install -y nvidia-container-toolkit"
  echo "         sudo systemctl restart docker"
  exit 1
fi

# Host memory (unified memory: model + compile + KV all share host RAM)
MEM_TOTAL_GIB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
if (( MEM_TOTAL_GIB < 80 )); then
  echo "FATAL: host has only ${MEM_TOTAL_GIB} GiB RAM; Ling-3.0-flash needs ~100+ GiB."
  exit 1
fi
if (( MEM_TOTAL_GIB < 110 )); then
  echo "WARN:  ${MEM_TOTAL_GIB} GiB RAM is below the DGX Spark class (~119 GiB)."
  echo "       Use conservative env (MEM_FRACTION_STATIC=0.70, MAX_MAMBA_CACHE_SIZE=16)."
fi

# Free disk for image (~20 GB) + weights + SGLang build
FREE_GIB=$(( $(df -Pk "${WORK_DIR}" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
if (( FREE_GIB < 40 )); then
  echo "FATAL: only ${FREE_GIB} GiB free on ${WORK_DIR} — need >= 40 GiB."
  exit 1
fi
if (( FREE_GIB < 100 )); then
  echo "WARN:  only ${FREE_GIB} GiB free on ${WORK_DIR} — keep an eye on disk usage."
fi

# Ports (host network: must be free on the host; not needed for --download-only)
if ! ${DOWNLOAD_ONLY}; then
  if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") >/dev/null 2>&1; then
    echo "FATAL: port ${PORT} is already in use — a server may already be running."
    echo "       Run ./stop.sh first, or pick another port via PORT=<port>."
    exit 1
  fi
  if (exec 3<>'/dev/tcp/127.0.0.1/2345') >/dev/null 2>&1; then
    echo "WARN:  internal dist port 2345 is in use; SGLang may fail to bind."
  fi
fi

# ---- Env exports (also passed to container) ---------------------------------
export HF_HOME
export HF_TOKEN="${HF_TOKEN:-}"
mkdir -p "${HF_HOME}" "${FLASHINFER_CACHE_DIR}"

is_hf_model_id() { [[ "${1}" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; }

hf_cache_repo_dir() { echo "${HF_HOME}/hub/models--${1//\//--}"; }

# ---- Model caching helpers --------------------------------------------------
model_is_fully_cached() {
  local cache_dir snapshot
  cache_dir="$(hf_cache_repo_dir "${1}")"
  [[ -d "${cache_dir}/snapshots" ]] || return 1
  for snapshot in "${cache_dir}"/snapshots/*/; do
    [[ -d "${snapshot}" ]] || continue
    [[ -f "${snapshot}/config.json" ]] || continue
    if [[ -f "${snapshot}/model.safetensors" ]] \
      || [[ -f "${snapshot}/model.safetensors.index.json" ]] \
      || compgen -G "${snapshot}/model-"*.safetensors >/dev/null \
      || [[ -f "${snapshot}/consolidated.safetensors" ]]; then
      return 0
    fi
  done
  return 1
}

snapshot_for() {
  local model_id="$1" revision="${2:-}"
  local cache_dir snapshot
  cache_dir="$(hf_cache_repo_dir "${model_id}")"
  if [[ -n "${revision}" && -d "${cache_dir}/snapshots/${revision}" ]]; then
    echo "${revision}"
    return 0
  fi
  for snapshot in "${cache_dir}"/snapshots/*/; do
    [[ -d "${snapshot}" ]] || continue
    basename "${snapshot}"
    return 0
  done
  return 1
}

download_model() {
  local model_id="$1" revision="${2:-}"
  echo ""
  echo "  >> Downloading ${model_id} …"
  echo "     (cache: ${HF_HOME})"
  [[ -n "${revision}" ]] && echo "     (revision: ${revision})"
  echo "     This can take a while for large models."

  if command -v hf >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" hf download "${model_id}" \
      ${revision:+--revision "${revision}"} \
      ${HF_TOKEN:+--token "${HF_TOKEN}"}
    return
  fi

  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" huggingface-cli download "${model_id}" \
      ${revision:+--revision "${revision}"} \
      ${HF_TOKEN:+--token "${HF_TOKEN}"}
    return
  fi

  # Fallback: download inside Docker
  docker run --rm \
    --entrypoint python3 \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    "${IMAGE}" \
    -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('${model_id}', revision='${revision}' or None, token=os.environ.get('HF_TOKEN') or None)
"
}

ensure_model() {
  local model_id="$1" label="$2" revision="${3:-}"
  if model_is_fully_cached "${model_id}" && { [[ -z "${revision}" ]] || [[ -d "$(hf_cache_repo_dir "${model_id}")/snapshots/${revision}" ]]; }; then
    echo "  [✓] ${label} (${model_id} @ ${revision:-cached}) is cached"
  else
    echo "  [↓] ${label} not cached — downloading …"
    download_model "${model_id}" "${revision}"
    if model_is_fully_cached "${model_id}"; then
      echo "  [✓] ${label} download complete"
    else
      echo "  [✗] ${label} download appears incomplete — check logs above"
      exit 1
    fi
  fi
}

# ---- Download model (idempotent) --------------------------------------------
echo "=============================================================================="
echo "  Ling-3.0-flash ${QUANT}  —  InclusionAI  —  1x DGX Spark"
echo "  $(date)"
echo "=============================================================================="
echo ""
echo "Checking model cache …"

ensure_model "${MODEL_ID}" "Ling-3.0-flash-${QUANT}" "${MODEL_REVISION}"
if [[ "${SPEC_ALGO}" == "dspark" ]]; then
  ensure_model "${DSPARK_MODEL_ID}" "Ling-3.0-flash-dspark (DSPARK draft)" "${DSPARK_REVISION}"
fi
echo ""

# ---- Early exit for download-only mode --------------------------------------
if ${DOWNLOAD_ONLY}; then
  echo "=============================================================================="
  echo "  Model is cached. Exiting (--download-only)."
  echo "=============================================================================="
  exit 0
fi

# ---- Bootstrap script -------------------------------------------------------
# Default path: use SGLang already in the image (or host-mounted persist).
# Source path: only when FORCE_SOURCE_BUILD=1 (private git is not required for normal runs).
# https://huggingface.co/inclusionAI/Ling-3.0-flash-int4
# https://docs.sglang.io/cookbook/autoregressive/InclusionAI/Ling-3.0-flash
cat > "${BOOTSTRAP_SCRIPT}" << BOOTSTRAP
#!/bin/bash
set -e

PERSIST="/opt/sglang-persist"
SGLANG_DIR="\${PERSIST}/sglang_ling_v3"
VENV="\${PERSIST}/venv"
SGLANG_BRANCH="${SGLANG_BRANCH}"
SGLANG_REPO="${SGLANG_REPO}"
FORCE_SOURCE_BUILD="${FORCE_SOURCE_BUILD}"
GITHUB_TOKEN="\${GITHUB_TOKEN:-}"

if [[ -n "\${PIP_CONSTRAINT:-}" ]]; then
  echo "[bootstrap] Clearing PIP_CONSTRAINT (\${PIP_CONSTRAINT})"
  unset PIP_CONSTRAINT
fi

export PATH="/root/.cargo/bin:\${PATH}"
mkdir -p "\${PERSIST}"

sglang_tree_ok() {
  [[ -d "\${SGLANG_DIR}/python/sglang" ]] || return 1
  [[ -f "\${SGLANG_DIR}/python/pyproject.toml" ]] || return 1
  return 0
}

activate_venv_if_present() {
  if [[ -x "\${VENV}/bin/python" ]]; then
    # shellcheck disable=SC1091
    source "\${VENV}/bin/activate"
    return 0
  fi
  return 1
}

launch_sglang() {
  echo "[bootstrap] Starting SGLang server (Ling-3.0-flash) …"
  if activate_venv_if_present; then
    exec python -m sglang.launch_server "\$@"
  fi
  if command -v sglang >/dev/null 2>&1; then
    # Official LMSYS images: preferred entrypoint is \`sglang serve\`
    exec sglang serve "\$@"
  fi
  if python3 -c "import sglang" 2>/dev/null; then
    exec python3 -m sglang.launch_server "\$@"
  fi
  echo "[bootstrap] FATAL: no SGLang runtime in this image."
  echo "  Use a prebuilt image, e.g.:"
  echo "    IMAGE=lmsysorg/sglang:dev-Ling-3.0-flash"
  echo "    IMAGE=ghcr.io/miaai-lab/ling-3.0-flash-sglang-dgx-spark:ling_v3_support"
  echo "  Or FORCE_SOURCE_BUILD=1 with a public SGLANG_REPO mirror."
  exit 1
}

clone_sglang() {
  local dest="\$1"
  local url="\${SGLANG_REPO}"
  if [[ -n "\${GITHUB_TOKEN}" && "\${url}" =~ ^https://github.com/ ]]; then
    url="https://x-access-token:\${GITHUB_TOKEN}@\${url#https://}"
  fi
  echo "[bootstrap] git clone -b \${SGLANG_BRANCH} \${SGLANG_REPO} …"
  if ! git clone -b "\${SGLANG_BRANCH}" "\${url}" "\${dest}"; then
    echo "[bootstrap] FATAL: could not clone \${SGLANG_REPO} (branch \${SGLANG_BRANCH})."
    echo "  inclusionAI/sglang_ling_v3 is often private (GitHub returns 'Repository not found')."
    echo "  Prefer a public prebuilt image instead of source build:"
    echo "    IMAGE=lmsysorg/sglang:dev-Ling-3.0-flash ./start.sh"
    echo "    IMAGE=ghcr.io/miaai-lab/ling-3.0-flash-sglang-dgx-spark:ling_v3_support ./start.sh"
    return 1
  fi
}

if [[ "\${FORCE_SOURCE_BUILD}" == "1" ]]; then
  echo "[bootstrap] FORCE_SOURCE_BUILD=1 — installing from \${SGLANG_REPO} …"
  if sglang_tree_ok && [[ -x "\${VENV}/bin/python" ]]; then
    echo "[bootstrap] Reusing existing source tree + venv"
  elif sglang_tree_ok; then
    echo "[bootstrap] Rebuilding venv from existing source (no clone) …"
    rm -rf "\${VENV}"
    cd "\${SGLANG_DIR}"
    command -v rustc &>/dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    python3 -m venv --system-site-packages "\${VENV}"
    # shellcheck disable=SC1091
    source "\${VENV}/bin/activate"
    pip install --upgrade pip
    pip install -e "python"
    { date; echo "branch=\${SGLANG_BRANCH}"; } > "\${SGLANG_DIR}/.installed"
  else
    rm -rf "\${SGLANG_DIR}.new" "\${VENV}"
    clone_sglang "\${SGLANG_DIR}.new"
    rm -rf "\${SGLANG_DIR}"
    mv "\${SGLANG_DIR}.new" "\${SGLANG_DIR}"
    cd "\${SGLANG_DIR}"
    command -v rustc &>/dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    python3 -m venv --system-site-packages "\${VENV}"
    # shellcheck disable=SC1091
    source "\${VENV}/bin/activate"
    pip install --upgrade pip
    pip install -e "python"
    { date; echo "branch=\${SGLANG_BRANCH}"; } > "\${SGLANG_DIR}/.installed"
  fi
else
  if sglang_tree_ok && [[ -x "\${VENV}/bin/python" ]]; then
    echo "[bootstrap] Using prebuilt/persisted SGLang at \${SGLANG_DIR}"
  else
    echo "[bootstrap] Using image-provided SGLang (no git clone)"
  fi
fi

launch_sglang "\$@"
BOOTSTRAP
chmod +x "${BOOTSTRAP_SCRIPT}"

# Host persist is only useful if it already has a working install. An empty
# bind-mount would hide the prebaked tree inside the image.
host_sglang_ok() {
  [[ -d "${SGLANG_PERSIST_DIR}/sglang_ling_v3/python/sglang" ]] \
    && [[ -x "${SGLANG_PERSIST_DIR}/venv/bin/python" ]]
}
SGLANG_VOLUME_ARGS=()
if host_sglang_ok; then
  echo "Using host SGLang persist: ${SGLANG_PERSIST_DIR}"
  SGLANG_VOLUME_ARGS+=(-v "${SGLANG_PERSIST_DIR}:/opt/sglang-persist")
else
  echo "Using image-baked SGLang (no host .sglang-persist mount)"
  mkdir -p "${SGLANG_PERSIST_DIR}"
fi

# ---- Container lifecycle ----------------------------------------------------
# Remove any existing container (running or stale) so we start fresh.
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Removing existing container ${CONTAINER_NAME} …"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi
# Legacy FP4 container name
if docker ps -a --format '{{.Names}}' | grep -qx "ling-3.0-flash-fp4"; then
  echo "Removing legacy container ling-3.0-flash-fp4 …"
  docker rm -f "ling-3.0-flash-fp4" >/dev/null
fi

echo "Starting SGLang server for ${MODEL_ID}"
echo "Quant: ${QUANT}"
echo "Image: ${IMAGE}"
echo "Spec decode: ${SPEC_ALGO}$( [[ "${SPEC_ALGO}" == "dspark" ]] && echo " (draft: ${DSPARK_MODEL_ID})")"
echo "Listening on ${HOST}:${PORT}"
echo "Context length: ${CTX}"
echo "Mem fraction: ${MEM_FRACTION_STATIC}"
echo ""

# Resolve snapshot path for the model (must be done before docker run)
HF_MODEL_CACHE="${HF_HOME}/hub/models--${MODEL_ID//\//--}"
SNAPSHOT_DIR="$(snapshot_for "${MODEL_ID}" "${MODEL_REVISION}" || true)"
if [[ -z "${SNAPSHOT_DIR}" ]]; then
  echo "FATAL: Could not find snapshot for ${MODEL_ID} in ${HF_MODEL_CACHE}/snapshots/"
  echo "       Run with --download-only first, or check your HF_HOME."
  exit 1
fi
if [[ "${SNAPSHOT_DIR}" != "${MODEL_REVISION}" ]]; then
  echo "WARN:  using snapshot ${SNAPSHOT_DIR}, not pinned revision ${MODEL_REVISION}"
fi
MODEL_PATH_IN_CONTAINER="/root/.cache/huggingface/hub/models--${MODEL_ID//\//--}/snapshots/${SNAPSHOT_DIR}"
echo "Model snapshot: ${HF_MODEL_CACHE}/snapshots/${SNAPSHOT_DIR}"

DSPARK_PATH_IN_CONTAINER=""
if [[ "${SPEC_ALGO}" == "dspark" ]]; then
  DSPARK_CACHE="${HF_HOME}/hub/models--${DSPARK_MODEL_ID//\//--}"
  DSPARK_SNAPSHOT_DIR="$(snapshot_for "${DSPARK_MODEL_ID}" "${DSPARK_REVISION}" || true)"
  if [[ -z "${DSPARK_SNAPSHOT_DIR}" ]]; then
    echo "FATAL: Could not find snapshot for ${DSPARK_MODEL_ID} in ${DSPARK_CACHE}/snapshots/"
    exit 1
  fi
  if [[ "${DSPARK_SNAPSHOT_DIR}" != "${DSPARK_REVISION}" ]]; then
    echo "WARN:  using DSPARK snapshot ${DSPARK_SNAPSHOT_DIR}, not pinned revision ${DSPARK_REVISION}"
  fi
  DSPARK_PATH_IN_CONTAINER="/root/.cache/huggingface/hub/models--${DSPARK_MODEL_ID//\//--}/snapshots/${DSPARK_SNAPSHOT_DIR}"
  echo "DSPARK draft snapshot: ${DSPARK_CACHE}/snapshots/${DSPARK_SNAPSHOT_DIR}"
fi
echo ""

echo "Pulling ${IMAGE} ..."
if ! docker pull "${IMAGE}" 2>&1; then
  echo "FATAL: failed to pull ${IMAGE}"
  echo "  Public alternatives:"
  echo "    IMAGE=lmsysorg/sglang:dev-Ling-3.0-flash ./start.sh"
  echo "    IMAGE=ghcr.io/miaai-lab/ling-3.0-flash-sglang-dgx-spark:ling_v3_support ./start.sh"
  echo "  Private GHCR: docker login ghcr.io -u USER --password-stdin  (then re-pull)"
  # Fall back to official public LMSYS image if the Spark GHCR tag is missing.
  if [[ "${IMAGE}" == "${DEFAULT_SPARK_IMAGE}" ]]; then
    echo "  Retrying with public LMSYS image: ${DEFAULT_PUBLIC_IMAGE}"
    IMAGE="${DEFAULT_PUBLIC_IMAGE}"
    if ! docker pull "${IMAGE}" 2>&1; then
      echo "FATAL: also failed to pull ${IMAGE}"
      exit 1
    fi
  else
    exit 1
  fi
fi
echo ""

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching SGLang container (${MODEL_ID})
EOF

# Optional flags (Spark-safe defaults; spec decode picked via SPEC_ALGO)
EXTRA_ARGS=()
if [[ "${QUANT}" == "mxfp4" ]]; then
  # Official GB10 cell (sgl-project/sglang#36364): FlashInfer CUTLASS MXFP4 MoE.
  EXTRA_ARGS+=(--moe-runner-backend flashinfer_mxfp4)
fi
if [[ -n "${KV_CACHE_DTYPE:-fp8_e4m3}" ]]; then
  EXTRA_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}")
fi
case "${SPEC_ALGO}" in
  dspark)
    # Cookbook recipe: DSPARK + external draft + KDA ReplaySSM verify path.
    # Draft block size (8) auto-infers from the checkpoint.
    EXTRA_ARGS+=(
      --speculative-algorithm DSPARK
      --speculative-draft-model-path "${DSPARK_PATH_IN_CONTAINER}"
      --enable-linear-replayssm-spec
      --linear-replayssm-cache-len "${REPLAYSSM_CACHE_LEN}"
    )
    ;;
  nextn)
    EXTRA_ARGS+=(--speculative-algorithm NEXTN)
    ;;
  off)
    # No speculative decoding
    ;;
esac
# Split reasoning into the reasoning field. Ling's chat template defaults
# thinking OFF; without this the ling3 parser dumps <think> into content.
# Set DEFAULT_CHAT_TEMPLATE_KWARGS= empty to omit (older images may lack the flag).
if [[ -z "${DEFAULT_CHAT_TEMPLATE_KWARGS+x}" ]]; then
  DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true,"thinking":true}'
fi
if [[ -n "${DEFAULT_CHAT_TEMPLATE_KWARGS}" ]]; then
  EXTRA_ARGS+=(--default-chat-template-kwargs "${DEFAULT_CHAT_TEMPLATE_KWARGS}")
fi

# Optional host-safety memory cap (unified memory: prefer container death over host OOM)
DOCKER_MEM_ARGS=()
if [[ -n "${DOCKER_MEMORY:-}" ]]; then
  echo "Docker memory limit: ${DOCKER_MEMORY} (memory-swap matched)"
  DOCKER_MEM_ARGS+=(--memory "${DOCKER_MEMORY}" --memory-swap "${DOCKER_MEMORY}")
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --user root \
  --network host \
  --shm-size=32g \
  --ulimit memlock=-1:-1 \
  --cap-add=IPC_LOCK \
  --ipc host \
  --gpus all \
  --workdir /workspace \
  --entrypoint /bootstrap.sh \
  "${DOCKER_MEM_ARGS[@]}" \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=1 \
  -e SGLANG_ENABLE_SPEC_V2=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e GITHUB_TOKEN="${GITHUB_TOKEN:-}" \
  -v "${BOOTSTRAP_SCRIPT}:/bootstrap.sh:ro" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${FLASHINFER_CACHE_DIR}:/root/.cache/flashinfer" \
  "${SGLANG_VOLUME_ARGS[@]}" \
  -v "${WORK_DIR}:/workspace" \
  "${IMAGE}" \
  --model-path "${MODEL_PATH_IN_CONTAINER}" \
    --trust-remote-code \
    --nnodes 1 \
    --dist-init-addr "127.0.0.1:2345" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp-size 1 \
    --ep-size 1 \
    --random-seed 308534008 \
    --max-running-requests "${MAX_RUNNING_REQUESTS:-6}" \
    --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE:-32}" \
    --chunked-prefill-size 8192 \
    --allow-auto-truncate \
    --context-length "${CTX}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --tool-call-parser ling3 \
    --reasoning-parser ling3 \
    --enable-fp32-lm-head \
    --disable-shared-experts-fusion \
    "${EXTRA_ARGS[@]}" \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id:0:12})"
echo "Log: ${LOG_FILE}"
echo ""

# ---- Wait for readiness -----------------------------------------------------
log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]]; then
    kill "${log_follow_pid}" 2>/dev/null || true
    wait "${log_follow_pid}" 2>/dev/null || true
    log_follow_pid=""
  fi
}
trap cleanup EXIT INT TERM

echo "Waiting for HTTP readiness at ${READY_URL}"
echo "--- container logs ---"

docker logs -f "${CONTAINER_NAME}" 2>&1 &
log_follow_pid=$!

while ! curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo ""
    echo "SGLang container exited before becoming ready"
    exit 1
  fi
  sleep 2
done

cleanup

echo ""
echo "=============================================================================="
echo "  SGLang is ready!"
echo "  Model: ${MODEL_ID} (${QUANT})"
if [[ "${HOST}" == "0.0.0.0" || "${HOST}" == "::" ]]; then
  echo "  OpenAI-compatible endpoint:  http://127.0.0.1:${PORT}/v1  (bound on ${HOST})"
else
  echo "  OpenAI-compatible endpoint:  http://${HOST}:${PORT}/v1"
fi
echo ""
echo "  Recommended client params:"
echo "    temperature=0.6  top_p=0.95  top_k=20"
echo '    chat_template_kwargs: {"enable_thinking": true}  (server default is on)'
case "${SPEC_ALGO}" in
  dspark)
    echo "  Spec decode: DSPARK with ${DSPARK_MODEL_ID} (ReplaySSM ring ${REPLAYSSM_CACHE_LEN})"
    ;;
  nextn)
    echo "  Spec decode: NEXTN (built-in MTP); set SPEC_ALGO=dspark or off to change"
    ;;
  off)
    echo "  Spec decode: off; set SPEC_ALGO=dspark or nextn to enable"
    ;;
esac
echo "=============================================================================="
