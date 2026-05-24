#!/usr/bin/env bash
# Download Qwen2.5-1.5B-Instruct (Apache-2.0) from the ModelScope mirror.
# The HuggingFace public endpoint rate-limits anonymous downloads to a
# few KB/s for files of this size; ModelScope serves at 10-30 MB/s.
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "$0")"/.. && pwd)/sim/models/Qwen2.5-1.5B-Instruct"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

BASE="https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct/resolve/master"
FILES=(
  config.json
  generation_config.json
  tokenizer.json
  tokenizer_config.json
  merges.txt
  vocab.json
  model.safetensors
)

for f in "${FILES[@]}"; do
  if [[ -s "$f" ]]; then
    echo "[qwen] $f already present; skipping."
    continue
  fi
  echo "[qwen] fetching $f"
  wget --tries=3 --timeout=120 --show-progress -O "$f.part" "$BASE/$f"
  mv "$f.part" "$f"
done

ls -lh
echo "[qwen] done."
