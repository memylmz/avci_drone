#!/usr/bin/env bash
# Hibrit takip modülünü bu bilgisayara (macOS, Apple Silicon) kurar.
# Kullanım (hibrit_takip klasörünün içinden):  bash kurulum_mac.sh
# Tekrar çalıştırmak güvenli: hazır olan adımları atlar.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"          # .../Avcı Drone/hibrit_takip
ROOT="$(dirname "$HERE")"                       # .../Avcı Drone
VENV="$ROOT/.venv"
OSTRACK="$HERE/OSTrack"
CKPT="$OSTRACK/output/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar"
CKPT_ID="1dySfPrSk-knAEo0lTXqZ1kMk3BCvVofV"   # resmi OSTrack Google Drive dosyası

echo "== 1/6 Python seçiliyor"
PY=""
for c in python3.13 python3.12 python3.11 \
         /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 \
         /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import lzma, ssl" 2>/dev/null; then PY="$(command -v "$c")"; break; fi
done
[ -n "$PY" ] || { echo "Uygun Python (3.11-3.13) bulunamadı. python.org'dan Python 3.13 kurun."; exit 1; }
echo "   $PY ($("$PY" --version))"

echo "== 2/6 Sanal ortam: $VENV"
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip

echo "== 3/6 Paketler kuruluyor (ilk seferde birkaç dakika sürer)"
python -m pip install -q torch torchvision ultralytics opencv-python numpy easydict timm \
    jpeg4py lmdb pyyaml pandas matplotlib pytest gdown

echo "== 4/6 OSTrack"
[ -d "$OSTRACK" ] || git clone -q --depth 1 https://github.com/botaoye/OSTrack.git "$OSTRACK"
python "$HERE/colab/prepare_ostrack.py" --ostrack-root "$OSTRACK" --save-dir "$OSTRACK/output"

echo "== 5/6 OSTrack ağırlığı (~350 MB)"
if [ -f "$CKPT" ] && [ "$(stat -f%z "$CKPT")" -gt 100000000 ]; then
  echo "   zaten var"
else
  gdown "$CKPT_ID" -O "$CKPT" || true
  if [ ! -f "$CKPT" ] || [ "$(stat -f%z "$CKPT")" -lt 100000000 ]; then
    echo "   OTOMATİK İNDİRME BAŞARISIZ. Tarayıcıda açın ve indirin:"
    echo "   https://drive.google.com/file/d/$CKPT_ID/view"
    echo "   İnen OSTrack_ep0300.pth.tar dosyasını şuraya koyun:"
    echo "   $CKPT"
    exit 1
  fi
fi

echo "== 6/6 Testler"
(cd "$HERE" && python -m pytest tests -q)
python - <<EOF
import torch
print("Apple GPU (MPS):", "VAR" if torch.backends.mps.is_available() else "YOK (CPU kullanılacak)")
EOF

echo
echo "Kurulum tamam. Çalıştırmak için:"
echo "  source \"$VENV/bin/activate\""
echo "  cd \"$HERE\""
echo "  python run_on_video.py --video VIDEO.mp4 --weights MODEL.pt --ostrack-root OSTrack --show"
