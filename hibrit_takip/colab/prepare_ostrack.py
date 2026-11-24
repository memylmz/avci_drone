"""OSTrack reposunu güncel Colab ortamına (PyTorch 2.x, NumPy 2.x, yeni timm) hazırlar.

Yapılanlar (hepsi idempotent, tekrar çalıştırmak güvenli):
  1. ostrack_patch.diff: track() dönüşüne best_score ekler
  2. torch._six importlarını kaldırır (PyTorch 2.x'te yok)
  3. np.bool / np.int / np.float kullanımlarını düzeltir (NumPy 1.24+'da yok)
  4. torch.load(..., weights_only=False) — PyTorch 2.6+ varsayılanı checkpoint'i reddediyor
  5. visdom importunu isteğe bağlı yapar (sadece debug görselleştirmede lazım)
  6. tensorboard importunu isteğe bağlı yapar
  7. .cuda() çağrılarını seçilebilir cihaza çevirir: CUDA, Apple GPU (MPS) veya CPU.
     Cihaz OSTRACK_DEVICE ortam değişkeniyle seçilir (auto/cuda/mps/cpu).
  8. lib/test/evaluation/local.py ve lib/train/admin/local.py dosyalarını oluşturur

Kullanım:
    python prepare_ostrack.py --ostrack-root /content/OSTrack --save-dir /content/ostrack_output
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, "..", "ostrack_patch.diff")


def edit(path: str, fn) -> bool:
    """Dosyayı fn ile dönüştürür; değiştiyse True."""
    with open(path, encoding="utf-8") as f:
        old = f.read()
    new = fn(old)
    if new != old:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
        return True
    return False


def apply_score_patch(root: str) -> None:
    trk = os.path.join(root, "lib/test/tracker/ostrack.py")
    with open(trk, encoding="utf-8") as f:
        if "best_score" in f.read():
            print("[ok] skor yaması zaten uygulanmış")
            return
    r = subprocess.run(["git", "apply", os.path.abspath(PATCH)], cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[HATA] ostrack_patch.diff uygulanamadı:\n{r.stderr}\nREADME'deki elle düzeltme adımını yapın.")
    print("[+] skor yaması uygulandı")


def fix_compat(root: str) -> None:
    # 2) torch._six
    loader = os.path.join(root, "lib/train/data/loader.py")
    if edit(loader, lambda s: s.replace("from torch._six import string_classes", "string_classes = str")
                               .replace("from torch._six import int_classes", "int_classes = int")):
        print("[+] torch._six kaldırıldı (loader.py)")

    # 3) NumPy eski tip takma adları
    np_re = re.compile(r"\bnp\.(bool|int|float)\b(?!_|\d)")
    for dirpath, _, files in os.walk(os.path.join(root, "lib")):
        for fn in files:
            if fn.endswith(".py"):
                p = os.path.join(dirpath, fn)
                if edit(p, lambda s: np_re.sub(lambda m: {"bool": "np.bool_", "int": "np.int64",
                                                           "float": "np.float64"}[m.group(1)], s)):
                    print(f"[+] NumPy takma adları düzeltildi: {os.path.relpath(p, root)}")

    # 4) torch.load weights_only
    trk = os.path.join(root, "lib/test/tracker/ostrack.py")
    if edit(trk, lambda s: s.replace("map_location='cpu')['net']", "map_location='cpu', weights_only=False)['net']")):
        print("[+] torch.load weights_only=False")

    # 5) visdom isteğe bağlı
    base = os.path.join(root, "lib/test/tracker/basetracker.py")
    if edit(base, lambda s: s if "Visdom = None" in s else s.replace(
            "from lib.vis.visdom_cus import Visdom\n",
            "try:\n    from lib.vis.visdom_cus import Visdom\nexcept ImportError:  # Colab: visdom gerekmiyor\n"
            "    Visdom = None\n")):
        print("[+] visdom importu isteğe bağlı yapıldı")

    # 6) tensorboard isteğe bağlı (sadece eğitimde lazım ama import zinciri test'te de çekiyor)
    tb = os.path.join(root, "lib/train/admin/tensorboard.py")
    if edit(tb, lambda s: (s if "SummaryWriter = None" in s else s.replace(
            "    from tensorboardX import SummaryWriter\n",
            "    try:\n        from tensorboardX import SummaryWriter\n    except ImportError:\n"
            "        SummaryWriter = None  # inference için gerekmiyor\n")).replace(
            "    print('WARNING: You are using tensorboardX instead sis you have a too old pytorch version.')\n",
            "    pass  # (yanıltıcı tensorboardX uyarısı kaldırıldı)\n")):
        print("[+] tensorboard importu isteğe bağlı yapıldı")

    # 7) Cihaz bağımsızlığı (GPU'suz bilgisayar / Apple Silicon için)
    dev_mod = os.path.join(root, "lib/utils/device.py")
    if not os.path.isfile(dev_mod):
        with open(dev_mod, "w", encoding="utf-8") as f:
            f.write(DEVICE_MODULE)
        print("[+] lib/utils/device.py oluşturuldu")
    for rel in ("lib/test/tracker/ostrack.py", "lib/test/tracker/data_utils.py", "lib/models/layers/head.py"):
        p = os.path.join(root, rel)

        def to_device(s: str) -> str:
            if "_OST_DEVICE" in s:
                return s
            s = s.replace(".cuda()", ".to(_OST_DEVICE)")
            return "from lib.utils.device import OST_DEVICE as _OST_DEVICE\n" + s

        if edit(p, to_device):
            print(f"[+] .cuda() -> seçilebilir cihaz: {rel}")


DEVICE_MODULE = '''"""OSTrack çalışma cihazı (hibrit_takip tarafından eklendi).

OSTRACK_DEVICE ortam değişkeni: auto (varsayılan) | cuda | mps | cpu
auto: CUDA varsa CUDA, yoksa Apple GPU (MPS), o da yoksa CPU.
"""
import os

import torch


def _pick() -> torch.device:
    d = os.environ.get("OSTRACK_DEVICE", "auto").lower()
    if d != "auto":
        return torch.device(d)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


OST_DEVICE = _pick()
'''


def create_local_files(root: str, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    r = subprocess.run([sys.executable, "tracking/create_default_local_file.py",
                        "--workspace_dir", root, "--data_dir", os.path.join(root, "data"),
                        "--save_dir", save_dir], cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[HATA] local.py oluşturulamadı:\n{r.stderr}")
    print(f"[+] local.py oluşturuldu (save_dir={save_dir})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ostrack-root", default="/content/OSTrack")
    ap.add_argument("--save-dir", default="/content/ostrack_output")
    ap.add_argument("--param", default="vitb_256_mae_ce_32x4_ep300")
    a = ap.parse_args()
    root, save_dir = os.path.abspath(a.ostrack_root), os.path.abspath(a.save_dir)

    apply_score_patch(root)
    fix_compat(root)
    create_local_files(root, save_dir)

    ckpt = os.path.join(save_dir, "checkpoints/train/ostrack", a.param, "OSTrack_ep0300.pth.tar")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    if os.path.isfile(ckpt):
        print(f"[ok] checkpoint bulundu: {ckpt}")
    else:
        print(f"[!] checkpoint YOK. Şuraya koyun: {ckpt}")


if __name__ == "__main__":
    main()
