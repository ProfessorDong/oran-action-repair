#!/usr/bin/env bash
# Download DeepSig RadioML 2016.10A from the public Zenodo mirror.
# License: CC BY-NC-SA 4.0 (DeepSig).  Respect the licence terms.
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")"/.. && pwd)/sim/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

URL="https://zenodo.org/api/records/18397070/files/RML2016.10a.tar.bz2/content"
ARCHIVE="RML2016.10a.tar.bz2"
PKL="RML2016.10a_dict_optimized.pkl"

if [[ -f "$PKL" ]]; then
    echo "[radioml] $PKL already present; skipping download."
    exit 0
fi

echo "[radioml] fetching $URL"
wget --tries=3 --timeout=120 --show-progress -O "$ARCHIVE" "$URL"

echo "[radioml] extracting"
tar xjf "$ARCHIVE"
rm -f "$ARCHIVE"

ls -lh "$PKL"
echo "[radioml] done."
