#!/usr/bin/env bash
# =============================================================================
#  start.sh — inclusionAI/Ling-3.0-flash-int4 on DGX Spark (GB10 / SM121)
#
#  Official INT4 recipe: branch ling_v3_support (no flashinfer_mxfp4).
#  Spark: tp-size 1, conservative mem defaults.
# =============================================================================
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
MODEL_ID="inclusionAI/Ling-3.0-flash-int4"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
CONTAINER_NAME="ling-3.0-flash-int4"
HOST="0.0.0.0"
PORT="${PORT:-8888}"
CTX="${CTX:-8192}"
WORK_DIR="$(pwd)"
HF_HOME="${HOME}/.cache/huggingface"
PID_FILE="${WORK_DIR}/.sglang.pid"
LOG_FILE="${WORK_DIR}/.sglang.log"
BOOTSTRAP_SCRIPT="/tmp/ling-bootstrap.sh"
FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${HOME}/.cache/flashinfer}"
# Persist SGLang + venv across container recreates (avoids 10+ min reinstall)
SGLANG_PERSIST_DIR="${SGLANG_PERSIST_DIR:-${WORK_DIR}/.sglang-persist}"
READY_URL="http://127.0.0.1:${PORT}/v1/models"
SGLANG_BRANCH="ling_v3_support"

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
    echo "    CTX                    Context length (default: 8192; raise when mem allows)"
    echo "    MEM_FRACTION_STATIC    SGLang static mem fraction (default: 0.75)"
    echo "    MAX_RUNNING_REQUESTS   (default: 1 on single Spark)"
    echo "    MAX_MAMBA_CACHE_SIZE   (default: 32 on single Spark)"
    echo "    KV_CACHE_DTYPE         (default: fp8_e4m3; set empty to omit flag)"
    echo "    ENABLE_NEXTN           set to 1 to pass --speculative-algorithm NEXTN"
    echo "    DOCKER_MEMORY          optional docker --memory (e.g. 100g); also sets --memory-swap"
    echo "    IMAGE                  Docker image (default: nvcr.io/nvidia/pytorch:26.01-py3)"
    echo "    HF_TOKEN               Hugging Face token for gated models"
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
# Official INT4 card: branch ling_v3_support (no MXFP4 / flashinfer_mxfp4).
# https://huggingface.co/inclusionAI/Ling-3.0-flash-int4
cat > "${BOOTSTRAP_SCRIPT}" << 'BOOTSTRAP'
#!/bin/bash
set -e

PERSIST="/opt/sglang-persist"
SGLANG_DIR="${PERSIST}/sglang_ling_v3"
VENV="${PERSIST}/venv"
SGLANG_BRANCH="ling_v3_support"

# NGC PyTorch pins some packages via PIP_CONSTRAINT; clear for clean installs.
if [[ -n "${PIP_CONSTRAINT:-}" ]]; then
  echo "[bootstrap] Clearing PIP_CONSTRAINT (${PIP_CONSTRAINT})"
  unset PIP_CONSTRAINT
fi

export PATH="/root/.cargo/bin:${PATH}"
mkdir -p "${PERSIST}"

if [[ -f "${SGLANG_DIR}/.installed" ]] \
  && grep -qx "branch=${SGLANG_BRANCH}" "${SGLANG_DIR}/.installed" 2>/dev/null \
  && [[ -x "${VENV}/bin/python" ]]; then
  echo "[bootstrap] Reusing persisted install at ${SGLANG_DIR} (branch ${SGLANG_BRANCH})"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
else
  echo "[bootstrap] Building persisted install under ${PERSIST} (branch ${SGLANG_BRANCH}) …"
  rm -rf "${SGLANG_DIR}" "${VENV}"
  git clone -b "${SGLANG_BRANCH}" https://github.com/inclusionAI/sglang_ling_v3.git "${SGLANG_DIR}"
  cd "${SGLANG_DIR}"

  if ! command -v rustc &>/dev/null; then
    echo "[bootstrap] Installing Rust …"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  fi

  echo "[bootstrap] Creating venv + installing SGLang …"
  python3 -m venv --system-site-packages "${VENV}"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  pip install --upgrade pip
  pip install -e "python"

  {
    date
    echo "branch=${SGLANG_BRANCH}"
  } > "${SGLANG_DIR}/.installed"
  echo "[bootstrap] SGLang installed (persisted)"
fi

echo "[bootstrap] Starting SGLang server (Ling-3.0-flash-int4) …"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
exec python -m sglang.launch_server "$@"
BOOTSTRAP
chmod +x "${BOOTSTRAP_SCRIPT}"
mkdir -p "${SGLANG_PERSIST_DIR}"

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
docker pull "${IMAGE}" 2>&1 || { echo "Failed to pull image"; exit 1; }
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
  -v "${BOOTSTRAP_SCRIPT}:/bootstrap.sh:ro" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${FLASHINFER_CACHE_DIR}:/root/.cache/flashinfer" \
  -v "${SGLANG_PERSIST_DIR}:/opt/sglang-persist" \
  -v "${WORK_DIR}:/workspace" \
  "${IMAGE}" \
  --model-path "${MODEL_PATH_IN_CONTAINER}" \
    --trust-remote-code \
    --nnodes 1 \
    --dist-init-addr "127.0.0.1:2345" \
    --port "${PORT}" \
    --tp-size 1 \
    --ep-size 1 \
    --random-seed 308534008 \
    --max-running-requests "${MAX_RUNNING_REQUESTS:-1}" \
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
echo "  OpenAI-compatible endpoint:  http://${HOST}:${PORT}/v1"
echo ""
echo "  Recommended client params:"
echo "    temperature=0.6  top_p=0.95  top_k=20"
echo '    chat_template_kwargs: {"enable_thinking": true}'
echo "  NEXTN: set ENABLE_NEXTN=1 to enable --speculative-algorithm NEXTN"
echo "=============================================================================="
