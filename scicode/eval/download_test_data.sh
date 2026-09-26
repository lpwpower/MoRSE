#!/usr/bin/env bash
# ============================================================================
# Download SciCode's numerical ground-truth `test_data.h5` (~1 GB).
#
# This is the official SciCode artifact used by the step/general evaluator
# (the subprocess in `scicode/src/scicode` reads it as `eval/data/test_data.h5`).
# It is NOT redistributed here; this script fetches it from the upstream
# SciCode project's published location.
#
# CANONICAL SOURCE (from the official SciCode repo README + eval/scripts):
#   https://github.com/scicode-bench/SciCode
#   "Download the numeric test results and save them as ./eval/data/test_data.h5"
#   The upstream link is a public Google Drive *folder* containing test_data.h5:
#     https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR
#
# If the upstream link ever changes, update SCICODE_GDRIVE_FOLDER_URL below to
# the new official location (do NOT point it at any private mirror).
# ============================================================================
set -euo pipefail

# --- Configuration (override via environment if needed) ---------------------

# Official upstream Google Drive *folder* holding test_data.h5 (~1 GB).
# Source of truth: scicode-bench/SciCode README + eval/scripts/README.md.
SCICODE_GDRIVE_FOLDER_URL="${SCICODE_GDRIVE_FOLDER_URL:-https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR}"

# Destination (kept in sync with the evaluator's expected path).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${SCRIPT_DIR}/data"
DEST_FILE="${DEST_DIR}/test_data.h5"

# Expected size is roughly 1 GB; this is only a sanity floor (bytes).
MIN_BYTES="${MIN_BYTES:-900000000}"

# --- Run --------------------------------------------------------------------

mkdir -p "${DEST_DIR}"

if [[ -f "${DEST_FILE}" ]]; then
  existing_bytes="$(stat -c%s "${DEST_FILE}" 2>/dev/null || stat -f%z "${DEST_FILE}")"
  if [[ "${existing_bytes}" -ge "${MIN_BYTES}" ]]; then
    echo "[skip] ${DEST_FILE} already present (${existing_bytes} bytes)."
    exit 0
  fi
  echo "[warn] ${DEST_FILE} exists but is smaller than expected (${existing_bytes} bytes); re-downloading."
fi

if ! command -v gdown >/dev/null 2>&1; then
  echo "[error] 'gdown' is required to fetch from Google Drive. Install it with:" >&2
  echo "          pip install gdown" >&2
  echo "        Alternatively, manually download test_data.h5 from:" >&2
  echo "          ${SCICODE_GDRIVE_FOLDER_URL}" >&2
  echo "        and place it at: ${DEST_FILE}" >&2
  exit 1
fi

echo "[info] Downloading SciCode test_data.h5 (~1 GB) from the official folder:"
echo "       ${SCICODE_GDRIVE_FOLDER_URL}"
echo "       -> ${DEST_FILE}"

# Download the whole folder into DEST_DIR, then place test_data.h5 at DEST_FILE.
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

gdown --folder "${SCICODE_GDRIVE_FOLDER_URL}" -O "${tmp_dir}"

found="$(find "${tmp_dir}" -name 'test_data.h5' -print -quit)"
if [[ -z "${found}" ]]; then
  echo "[error] test_data.h5 not found in the downloaded folder. Inspect: ${tmp_dir}" >&2
  echo "        The upstream layout may have changed; verify ${SCICODE_GDRIVE_FOLDER_URL}" >&2
  exit 1
fi

mv -f "${found}" "${DEST_FILE}"

final_bytes="$(stat -c%s "${DEST_FILE}" 2>/dev/null || stat -f%z "${DEST_FILE}")"
if [[ "${final_bytes}" -lt "${MIN_BYTES}" ]]; then
  echo "[error] Downloaded file is only ${final_bytes} bytes (< ${MIN_BYTES}); likely incomplete." >&2
  exit 1
fi

echo "[ok] Saved ${DEST_FILE} (${final_bytes} bytes)."
