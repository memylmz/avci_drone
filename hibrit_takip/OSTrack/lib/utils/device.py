"""OSTrack çalışma cihazı (hibrit_takip tarafından eklendi).

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
