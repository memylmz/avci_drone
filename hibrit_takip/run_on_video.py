"""Hibrit takipçiyi bir video dosyası üzerinde çalıştıran demo.

Yerelde (Mac'te Apple GPU otomatik kullanılır):
    python run_on_video.py --video ucus.mp4 --weights best.pt --ostrack-root ../OSTrack --show

Colab'da (ekran yok; sonuç videoya ve CSV'ye yazılır):
    !python run_on_video.py --video /content/drive/MyDrive/avci/ucus.mp4 \
        --weights /content/drive/MyDrive/avci/best.pt --ostrack-root /content/OSTrack \
        --out-video /content/sonuc.mp4 --csv /content/sonuc.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

from hybrid_tracker import Detection, HybridTracker, OSTrackWrapper, TrackerParams

BOX_COLOR_BGR = (0, 0, 255)   # #FF0000 (OpenCV BGR sırası)
BOX_THICKNESS = 2             # şartname: en fazla 3 px
METHOD_COLORS = {"YOLO": (255, 120, 0), "OSTrack": (0, 255, 0), "OSTrack+YOLO": (255, 255, 0),
                 "Kalman": (0, 165, 255), "YOLO+Kalman": (255, 0, 255)}
STATE_COLORS = {"ARAMA": (200, 200, 200), "ADAY": (0, 255, 255), "TAKIP": (0, 255, 0), "KAYIP": (0, 165, 255)}


def in_colab() -> bool:
    """Colab ortamında mı çalışıyoruz?"""
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def has_display() -> bool:
    """cv2.imshow kullanılabilir mi? (Colab ve sunucularda kullanılamaz)"""
    if in_colab():
        return False
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def auto_yolo_device() -> str:
    """NVIDIA GPU > Apple GPU (MPS) > CPU."""
    import torch
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class UltralyticsDetector:
    """Ultralytics YOLO modelini Detector arayüzüne uyarlar (xyxy -> xywh)."""

    def __init__(self, weights: str, imgsz: int = 640, classes: Optional[List[int]] = None,
                 device: Optional[str] = None, half: bool = False) -> None:
        from ultralytics import YOLO  # isteğe bağlı bağımlılık
        self.model = YOLO(weights)
        self.imgsz, self.classes, self.device, self.half = imgsz, classes, device, half

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        # Eşiği düşük tutuyoruz; asıl eşik TrackerParams.det_conf ile uygulanıyor
        kw = {"half": True} if self.half else {}  # yeni ultralytics'te half=False bile uyarı basıyor
        res = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=0.05, classes=self.classes,
                                 device=self.device, verbose=False, **kw)[0]
        out: List[Detection] = []
        for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            out.append(((x1, y1, x2 - x1, y2 - y1), float(c)))
        return out


def draw(frame: np.ndarray, out: dict) -> None:
    """Kutuyu, durumu ve skoru kare üzerine çizer."""
    if out["box"] is not None:
        x, y, w, h = (int(round(v)) for v in out["box"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR_BGR, BOX_THICKNESS)
    if out["predicted_center"] is not None:
        cx, cy = (int(round(v)) for v in out["predicted_center"])
        cv2.drawMarker(frame, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 10, 1)
    text = f"#{out['frame_idx']} {out['state']} skor={out['score']:.2f}"
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                STATE_COLORS.get(out["state"], (255, 255, 255)), 2, cv2.LINE_AA)
    # O an kutuyu üreten yöntem
    method = out.get("method", "")
    mtext = f"yontem: {method}"
    cv2.putText(frame, mtext, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, mtext, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                METHOD_COLORS.get(method, (255, 255, 255)), 2, cv2.LINE_AA)


def to_h264(path: str) -> Optional[str]:
    """mp4v videoyu tarayıcıda/Colab'da oynatılabilir H.264'e çevirir (ffmpeg gerekir)."""
    if not shutil.which("ffmpeg"):
        return None
    out = os.path.splitext(path)[0] + "_h264.mp4"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out])
    return out if r.returncode == 0 else None


def process_video(video: str, tracker: HybridTracker, csv_path: str,
                  out_video: Optional[str] = None, show: bool = False,
                  max_frames: Optional[int] = None, progress_every: int = 100) -> int:
    """Videoyu kare kare işler, CSV'ye yazar. İşlenen kare sayısını döndürür."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    writer = None
    n = 0
    t0 = time.time()
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s", "state", "x", "y", "w", "h", "score", "method"])
        while max_frames is None or n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            # Video zamanı; canlı sistemde sunucu saatiyle eşleşmiş zaman verilecek
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            out = tracker.step(frame, timestamp=ts)
            box = out["box"] or ("", "", "", "")
            wr.writerow([out["frame_idx"], f"{ts:.3f}", out["state"],
                         *[f"{v:.1f}" if v != "" else "" for v in box], f"{out['score']:.4f}", out["method"]])
            draw(frame, out)
            if out_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame)
            if show:
                cv2.imshow("hibrit takip", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            n += 1
            if progress_every and n % progress_every == 0:
                el = time.time() - t0
                print(f"  {n}/{total if total > 0 else '?'} kare, {n / el:.1f} fps, durum={out['state']}, yontem={out['method']}",
                      flush=True)
    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    el = time.time() - t0
    print(f"{n} kare {el:.1f} sn'de işlendi ({n / max(el, 1e-6):.1f} fps)")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO + OSTrack + Kalman hibrit takip demosu")
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True, help="YOLO ağırlık dosyası (ultralytics)")
    ap.add_argument("--ostrack-root", required=True, help="OSTrack reposunun kök dizini")
    ap.add_argument("--ostrack-param", default="vitb_256_mae_ce_32x4_ep300")
    ap.add_argument("--classes", type=int, nargs="*", default=None, help="YOLO sınıf filtresi")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None,
                    help="YOLO cihazı: '0' (NVIDIA GPU), 'mps' (Apple GPU) veya 'cpu'. Boşsa otomatik")
    ap.add_argument("--ostrack-device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                    help="OSTrack cihazı. auto: CUDA > Apple GPU (MPS) > CPU")
    ap.add_argument("--half", action="store_true", help="YOLO FP16 (sadece NVIDIA GPU'da)")
    ap.add_argument("--csv", default="sonuclar.csv")
    ap.add_argument("--out-video", default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="Hızlı deneme için kare sınırı")
    ap.add_argument("--show", action="store_true", help="Pencere aç (Colab'da çalışmaz, otomatik kapanır)")
    ap.add_argument("--log-level", default="INFO")
    # Parametre ayarları
    for name, default in vars(TrackerParams()).items():
        if isinstance(default, bool):
            continue
        ap.add_argument(f"--{name.replace('_', '-')}", type=type(default), default=default)
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    params = TrackerParams(**{k: getattr(args, k) for k in vars(TrackerParams()) if hasattr(args, k)})

    show = args.show and has_display()
    if args.show and not show:
        print("Ekran yok (Colab/sunucu): pencere açılmayacak, sonucu --out-video ile kaydedin.")
    if not args.out_video and not show:
        print("Uyarı: --out-video verilmedi, sadece CSV yazılacak.")

    sot = OSTrackWrapper(args.ostrack_root, param_name=args.ostrack_param, device=args.ostrack_device)
    yolo_device = args.device or auto_yolo_device()
    half = args.half and yolo_device not in ("cpu", "mps")
    print(f"Cihazlar: YOLO={yolo_device}, OSTrack={sot.device}")
    detector = UltralyticsDetector(args.weights, imgsz=args.imgsz, classes=args.classes,
                                   device=yolo_device, half=half)
    tracker = HybridTracker(detector, sot, params)
    process_video(args.video, tracker, args.csv, args.out_video, show=show, max_frames=args.max_frames)
    print(f"CSV: {args.csv}")
    if args.out_video:
        h264 = to_h264(args.out_video)
        print(f"Video: {h264 or args.out_video}")


if __name__ == "__main__":
    main()
