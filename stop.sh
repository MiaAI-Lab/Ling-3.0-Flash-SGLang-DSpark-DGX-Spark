#!/usr/bin/env bash
# =============================================================================
#  stop.sh — Stop and remove the Ling-3.0-flash-int4 SGLang container
# =============================================================================
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
CONTAINER_NAME="${CONTAINER_NAME:-ling-3.0-flash-int4}"
WORK_DIR="$(pwd)"
PID_FILE="${WORK_DIR}/.sglang.pid"
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
    echo "  Stops and removes the ${CONTAINER_NAME} Docker container."
    echo ""
    echo "  -f, --force    Skip confirmation prompt"
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
  # Clean up stale PID file
  rm -f "${PID_FILE}"
  exit 0
fi

# ---- Show container status --------------------------------------------------
STATUS=$(docker inspect -f '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo "unknown")
echo "Container: ${CONTAINER_NAME}"
echo "Status:    ${STATUS}"
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

# ---- Clean up PID file ------------------------------------------------------
rm -f "${PID_FILE}"

# ---- Clean up bootstrap script ----------------------------------------------
rm -f "${BOOTSTRAP_SCRIPT}"

echo ""
echo "=============================================================================="
echo "  ${CONTAINER_NAME} stopped and removed."
echo "=============================================================================="
