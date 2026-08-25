#!/usr/bin/env bash
# =============================================================================
#  stop.sh — Stop and remove the Ling-3.0-flash-int4 SGLang container
#
#  Works regardless of speculative-decoding mode (DSPARK draft / NEXTN / off):
#  it simply stops and removes the container started by start.sh.
#
#  Cleaned up:
#    - the Docker container (any state: running, exited, dead)
#    - .sglang.pid  (container ID written by start.sh)
#    - .sglang.log  (launch line written by start.sh)
#    - /tmp/ling-bootstrap.sh (container bootstrap script)
#
#  Not touched (safe to keep across restarts):
#    - model weights in $HF_HOME
#    - .sglang-persist/ (optional host-side SGLang install)
# =============================================================================
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
# Keep in sync with start.sh. Override if you launched with a custom name.
CONTAINER_NAME="${CONTAINER_NAME:-ling-3.0-flash-int4}"
WORK_DIR="$(pwd)"
PID_FILE="${WORK_DIR}/.sglang.pid"
LOG_FILE="${WORK_DIR}/.sglang.log"
BOOTSTRAP_SCRIPT="/tmp/ling-bootstrap.sh"

# ---- Argument parsing -------------------------------------------------------
FORCE=false
case "${1:-}" in
  -f|--force)
    FORCE=true
    shift
    ;;
  -h|--help)
    echo "Usage: $0 [-f|--force]"
    echo ""
    echo "  Stops and removes the ${CONTAINER_NAME} Docker container"
    echo "  (Ling-3.0-flash-int4 served with SGLang, any spec-decode mode:"
    echo "   DSPARK draft / NEXTN / off)."
    echo ""
    echo "  -f, --force    Skip confirmation prompt"
    echo ""
    echo "  Environment:"
    echo "    CONTAINER_NAME  Container to stop (default: ling-3.0-flash-int4)"
    exit 0
    ;;
  "")
    ;;
  *)
    echo "Unknown argument: $1"
    echo "Usage: $0 [-f|--force]"
    exit 1
    ;;
esac

# ---- Prerequisite checks ----------------------------------------------------
command -v docker >/dev/null 2>&1 || { echo "FATAL: docker is required"; exit 1; }

# ---- Check container exists -------------------------------------------------
if ! docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} not found — nothing to stop."
  # Clean up stale artifacts even if the container is gone
  rm -f "${PID_FILE}" "${LOG_FILE}" "${BOOTSTRAP_SCRIPT}"
  exit 0
fi

# ---- Show container status --------------------------------------------------
STATUS=$(docker inspect -f '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo "unknown")
echo "Container: ${CONTAINER_NAME}"
echo "Status:    ${STATUS}"

# Show which spec-decode mode the container was launched with (best effort)
SPEC_MODE=$(docker inspect -f '{{join .Args " "}}' "${CONTAINER_NAME}" 2>/dev/null \
  | grep -oE '\-\-speculative-algorithm [A-Z0-9_]+' || true)
if [[ -n "${SPEC_MODE}" ]]; then
  echo "Spec:      ${SPEC_MODE#--speculative-algorithm }"
fi
echo ""

# ---- Confirmation -----------------------------------------------------------
if [[ "${FORCE}" != "true" ]]; then
  read -r -p "Stop and remove this container? [y/N] " answer
  case "${answer}" in
    [yY]|[yY][eE][sS])
      ;;
    *)
      echo "Aborted."
      exit 0
      ;;
  esac
fi

# ---- Stop container ---------------------------------------------------------
echo ""
echo "Stopping ${CONTAINER_NAME} …"
docker stop "${CONTAINER_NAME}" 2>/dev/null || true

# ---- Remove container -------------------------------------------------------
echo "Removing ${CONTAINER_NAME} …"
docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true

# ---- Clean up PID file, launch log, and bootstrap script --------------------
rm -f "${PID_FILE}" "${LOG_FILE}" "${BOOTSTRAP_SCRIPT}"

echo ""
echo "=============================================================================="
echo "  ${CONTAINER_NAME} stopped and removed."
echo "  Model weights stay cached in ~/.cache/huggingface — restart with ./start.sh"
echo "=============================================================================="
