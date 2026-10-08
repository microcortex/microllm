#!/usr/bin/env bash
# MicroCortex — nightly flash + verify (un solo comando, todo el output a .scratch/nightly.log)
# Uso: bash ~/Documents/microllm/recipes/000-llama2c-esp32-s3/nightly.sh
set -u
LOG="$HOME/Documents/microllm/.scratch/nightly.log"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "=== MicroCortex nightly $(date) ==="

echo "=== [0/4] Activando entorno ==="
export PATH="$HOME/esp/shadow:$PATH"
. "$HOME/esp/esp-idf/export.sh" || { echo "❌ export.sh falló"; exit 1; }

cd "$HOME/Documents/microllm/recipes/000-llama2c-esp32-s3/firmware" || exit 1

echo "=== [1/4] BUILD ==="
idf.py build || { echo "❌ BUILD FAILED"; exit 1; }

echo "=== [2/4] FLASH FIRMWARE ==="
sg dialout -c "idf.py -p /dev/ttyACM0 flash" || { echo "❌ FW FLASH FAILED"; exit 1; }

echo "=== [3/4] FLASH MODEL (¡verificar Hash of data verified!) ==="
sg dialout -c "esptool.py --chip esp32s3 -p /dev/ttyACM0 --baud 921600 write_flash 0x190000 ../train/out/model_Q8.jny" || { echo "❌ MODEL FLASH FAILED"; exit 1; }

echo "=== [4/4] SERIAL CAPTURE 30s (idf.py monitor, se corta solo) ==="
sg dialout -c "timeout 30 idf.py -p /dev/ttyACM0 monitor" || true

echo "=== DONE — log completo en $LOG ==="
