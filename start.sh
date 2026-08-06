#!/usr/bin/env bash
# =============================================================================
#  start.sh — inclusionAI/Ling-3.0-flash-int4 on DGX Spark (GB10 / SM121)
#
#  Default: public prebuilt Docker image — no private GitHub clone required.
#  Spark: tp-size 1, conservative mem defaults. No flashinfer_mxfp4 / FP4.
# =============================================================================
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
MODEL_ID="inclusionAI/Ling-3.0-flash-int4"
# Public images (new users need one of these — never requires inclusionAI/sglang_ling_v3):
#   1) Spark-tuned (baked ling_v3_support): ghcr.io/miaai-lab/ling-3.0-flash-sglang-dgx-spark:ling_v3_support
#   2) Official LMSYS Ling-3.0 runtime:     lmsysorg/sglang:dev-Ling-3.0-flash
# Source rebuild only if you set IMAGE to a bare base (e.g. nvcr.io/nvidia/pytorch:26.01-py3)
# and FORCE_SOURCE_BUILD=1 (needs a public SGLANG_REPO mirror; official fork is often private).
# Always default to a public registry image. Spark-tuned GHCR is preferred on
# aarch64 when available; LMSYS is the universal public fallback.
DEFAULT_SPARK_IMAGE="ghcr.io/miaai-lab/ling-3.0-flash-sglang-dgx-spark:ling_v3_support"
DEFAULT_PUBLIC_IMAGE="lmsysorg/sglang:dev-Ling-3.0-flash"
if [[ -z "${IMAGE:-}" ]]; then
  # Default: Spark GHCR (public; baked ling_v3_support for INT4). LMSYS is fallback
  # if the pull fails, or set USE_LMSYS_IMAGE=1 / IMAGE=lmsysorg/sglang:dev-Ling-3.0-flash.
  if [[ "${USE_LMSYS_IMAGE:-0}" == "1" ]]; then
    IMAGE="${DEFAULT_PUBLIC_IMAGE}"
  else
    IMAGE="${DEFAULT_SPARK_IMAGE}"
  fi
fi
CONTAINER_NAME="ling-3.0-flash-int4"
# Bind address for sglang serve / launch_server (--host). Docker uses --network host.
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
CTX="${CTX:-262144}"
WORK_DIR="$(pwd)"
HF_HOME="${HOME}/.cache/huggingface"
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
    echo "    PORT                   Server port (default: 8888)"
    echo "    HOST                   Bind address for SGLang (default: 0.0.0.0)"
    echo "    CTX                    Context length (default: 262144 / 256k)"
    echo "    MEM_FRACTION_STATIC    SGLang static mem fraction (default: 0.75)"
    echo "    MAX_RUNNING_REQUESTS   (default: 6 concurrent)"
    echo "    MAX_MAMBA_CACHE_SIZE   (default: 32 on single Spark)"
    echo "    KV_CACHE_DTYPE         (default: fp8_e4m3; set empty to omit flag)"
    echo "    ENABLE_NEXTN           set to 1 to pass --speculative-algorithm NEXTN"
    echo "    DOCKER_MEMORY          optional docker --memory (e.g. 100g); also sets --memory-swap"
    echo "    IMAGE                  Docker image (default: ghcr.io/miaai-lab/ling-3.0-flash-sglang-dgx-spark:ling_v3_support)"
    echo "    USE_LMSYS_IMAGE        set to 1 for lmsysorg/sglang:dev-Ling-3.0-flash instead"
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
  echo "FATAL: host has only ${MEM_TOTAL_GIB} GiB RAM; Ling-3.0-flash-int4 needs ~100+ GiB."
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

# Ports (host network: must be free on the host)
if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") >/dev/null 2>&1; then
  echo "FATAL: port ${PORT} is already in use — a server may already be running."
  echo "       Run ./stop.sh first, or pick another port via PORT=<port>."
  exit 1
fi
if (exec 3<>'/dev/tcp/127.0.0.1/2345') >/dev/null 2>&1; then
  echo "WARN:  internal dist port 2345 is in use; SGLang may fail to bind."
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

download_model() {
  local model_id="$1"
  echo ""
  echo "  >> Downloading ${model_id} …"
  echo "     (cache: ${HF_HOME})"
  echo "     This can take a while for large models."

  if command -v hf >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" hf download "${model_id}" \
      ${HF_TOKEN:+--token "${HF_TOKEN}"}
    return
  fi

  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" huggingface-cli download "${model_id}" \
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
snapshot_download('${model_id}', token=os.environ.get('HF_TOKEN') or None)
"
}

ensure_model() {
  local model_id="$1" label="$2"
  if model_is_fully_cached "${model_id}"; then
    echo "  [✓] ${label} (${model_id}) is cached"
  else
    echo "  [↓] ${label} not cached — downloading …"
    download_model "${model_id}"
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
echo "  Ling-3.0-flash-int4  —  InclusionAI"
echo "  $(date)"
echo "=============================================================================="
echo ""
echo "Checking model cache …"

ensure_model "${MODEL_ID}" "Ling-3.0-flash-int4"
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
  echo "[bootstrap] Starting SGLang server (Ling-3.0-flash-int4) …"
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
echo "Image: ${IMAGE}"
echo "Listening on ${HOST}:${PORT}"
echo "Context length: ${CTX}"
echo ""

# Resolve snapshot path for the model (must be done before docker run)
HF_MODEL_CACHE="${HF_HOME}/hub/models--${MODEL_ID//\//--}"
SNAPSHOT_DIR="$(ls "${HF_MODEL_CACHE}/snapshots/" 2>/dev/null | head -1)"
if [[ -z "${SNAPSHOT_DIR}" ]]; then
  echo "FATAL: Could not find snapshot for ${MODEL_ID} in ${HF_MODEL_CACHE}/snapshots/"
  echo "       Run with --download-only first, or check your HF_HOME."
  exit 1
fi
MODEL_PATH_IN_CONTAINER="/root/.cache/huggingface/hub/models--${MODEL_ID//\//--}/snapshots/${SNAPSHOT_DIR}"
echo "Model snapshot: ${HF_MODEL_CACHE}/snapshots/${SNAPSHOT_DIR}"
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

# Optional flags (Spark-safe defaults; card recommends NEXTN but it is opt-in here)
EXTRA_ARGS=()
if [[ -n "${KV_CACHE_DTYPE:-fp8_e4m3}" ]]; then
  EXTRA_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}")
fi
if [[ "${ENABLE_NEXTN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--speculative-algorithm NEXTN)
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
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.75}" \
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
echo "  Model: ${MODEL_ID}"
if [[ "${HOST}" == "0.0.0.0" || "${HOST}" == "::" ]]; then
  echo "  OpenAI-compatible endpoint:  http://127.0.0.1:${PORT}/v1  (bound on ${HOST})"
else
  echo "  OpenAI-compatible endpoint:  http://${HOST}:${PORT}/v1"
fi
echo ""
echo "  Recommended client params:"
echo "    temperature=0.6  top_p=0.95  top_k=20"
echo '    chat_template_kwargs: {"enable_thinking": true}'
echo "  NEXTN: set ENABLE_NEXTN=1 to enable --speculative-algorithm NEXTN"
echo "=============================================================================="
