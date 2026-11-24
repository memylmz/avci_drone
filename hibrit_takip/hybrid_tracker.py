"""YOLO + OSTrack + Kalman hibrit hedef takip modülü.

Bu modül yalnızca "hedef nerede ve ne kadar eminim" sorusunu cevaplar.
Şartname kilit kuralları (AV bölgesi, boyut eşiği, 10 sn'de 5 sn, 200 ms tolerans)
bu modülün çıktısını kullanan ayrı bir katmandadır.

Kutu formatı her yerde: (x, y, w, h) — sol-üst köşe + genişlik/yükseklik, piksel.

Durum makinesi:
    ARAMA -> ADAY -> TAKIP -> KAYIP -> (TAKIP | ARAMA)
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("hybrid_tracker")

Box = Tuple[float, float, float, float]          # (x, y, w, h)
Detection = Tuple[Box, float]                     # (kutu, güven)
Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Arayüzler
# ---------------------------------------------------------------------------

class Detector(Protocol):
    """Tespit modeli arayüzü (YOLO buraya bağlanacak)."""

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Karedeki tespitleri [(box_xywh, conf), ...] olarak döndürür."""
        ...


class SingleObjectTracker(Protocol):
    """Tek hedef takipçi arayüzü (OSTrack veya test için mock)."""

    def init(self, frame_bgr: np.ndarray, box: Box) -> None:
        """Takipçiyi verilen kutuyla (yeniden) başlatır."""
        ...

    def update(self, frame_bgr: np.ndarray) -> Tuple[Box, float]:
        """Yeni karede (kutu, skor) döndürür."""
        ...


# Ego-motion uzantı noktası: kare -> (dx, dy). Sabit sahnenin görüntüde kaç piksel
# kaydığını döndürmeli (ör. optik akış / homografi ile). Şimdilik implementasyon yok.
EgoMotionEstimator = Callable[[np.ndarray], Tuple[float, float]]


# ---------------------------------------------------------------------------
# Parametreler ve durumlar
# ---------------------------------------------------------------------------

@dataclass
class TrackerParams:
    """Hibrit takipçi parametreleri."""

    # ADAY doğrulama
    confirm_frames: int = 5          # kaç ardışık tutarlı tespitle hedef doğrulanır
    confirm_dist: float = 40.0       # ardışık tespit merkezleri arası maks. mesafe (px)
    det_conf: float = 0.5            # YOLO güven eşiği
    # TAKIP
    verify_every: int = 30 # kaç karede bir YOLO ile doğrulama yapılır
    reinit_iou: float = 0.5          # YOLO-OSTrack IoU'su bunu geçerse OSTrack yeniden başlatılır
    track_score_min: float = 0.35    # OSTrack skor eşiği
    low_score_patience: int = 3      # art arda kaç düşük skorlu karede KAYIP'a geçilir
    # KAYIP
    lost_max_frames: int = 15        # KAYIP'ta en fazla kaç kare beklenir
    lost_search_scale: float = 2.0   # arama yarıçapı = büyük kenar * bu katsayı
    lost_search_min: float = 20.0    # arama yarıçapı alt sınırı (px)
    # Kalman gürültüleri (piksel^2 / kare)
    kf_q_pos: float = 1.0            # konum/boyut süreç gürültüsü
    kf_q_vel: float = 10.0           # hız süreç gürültüsü (ego-motion yüzünden yüksek tutuldu)
    kf_r: float = 4.0                # ölçüm gürültüsü
    # Çıktı
    output_smoothed: bool = True     # True: Kalman'dan geçmiş kutu, False: ham OSTrack/YOLO kutusu


class TrackState(str, Enum):
    """Takip durumları."""

    ARAMA = "ARAMA"
    ADAY = "ADAY"
    TAKIP = "TAKIP"
    KAYIP = "KAYIP"


# ---------------------------------------------------------------------------
# Yardımcı geometri fonksiyonları
# ---------------------------------------------------------------------------

def box_center(box: Box) -> Point:
    """Kutunun merkezini döndürür."""
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def center_dist(a: Box, b: Box) -> float:
    """İki kutunun merkezleri arası Öklid mesafesi."""
    (ax, ay), (bx, by) = box_center(a), box_center(b)
    return math.hypot(ax - bx, ay - by)


def iou(a: Box, b: Box) -> float:
    """İki xywh kutunun IoU değeri."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def clip_box(box: Box, width: int, height: int) -> Box:
    """Kutuyu kare sınırlarına kırpar (en az 1x1 px bırakır)."""
    x, y, w, h = box
    x1 = min(max(x, 0.0), width - 1.0)
    y1 = min(max(y, 0.0), height - 1.0)
    x2 = min(max(x + w, x1 + 1.0), float(width))
    y2 = min(max(y + h, y1 + 1.0), float(height))
    return (x1, y1, x2 - x1, y2 - y1)


# ---------------------------------------------------------------------------
# Kalman filtresi
# ---------------------------------------------------------------------------

class KalmanBoxFilter:
    """Sabit hız modelli kutu filtresi.

    Durum: [cx, cy, w, h, vx, vy, vw, vh], ölçüm: [cx, cy, w, h].
    """

    def __init__(self, q_pos: float = 1.0, q_vel: float = 10.0, r: float = 4.0) -> None:
        self.q_pos, self.q_vel, self.r = q_pos, q_vel, r
        self.kf: Optional[cv2.KalmanFilter] = None

    @property
    def initialized(self) -> bool:
        return self.kf is not None

    def init(self, box: Box) -> None:
        """Filtreyi verilen kutuyla, sıfır hızla başlatır."""
        kf = cv2.KalmanFilter(8, 4)
        F = np.eye(8, dtype=np.float32)
        for i in range(4):
            F[i, i + 4] = 1.0  # dt = 1 kare
        kf.transitionMatrix = F
        kf.measurementMatrix = np.hstack([np.eye(4), np.zeros((4, 4))]).astype(np.float32)
        kf.processNoiseCov = np.diag([self.q_pos] * 4 + [self.q_vel] * 4).astype(np.float32)
        kf.measurementNoiseCov = (np.eye(4) * self.r).astype(np.float32)
        # Başlangıçta hız hakkında bilgi yok -> hız belirsizliği büyük
        kf.errorCovPost = np.diag([10.0] * 4 + [100.0] * 4).astype(np.float32)
        cx, cy = box_center(box)
        kf.statePost = np.array([[cx], [cy], [box[2]], [box[3]], [0], [0], [0], [0]], dtype=np.float32)
        self.kf = kf

    def reset(self) -> None:
        self.kf = None

    def apply_camera_motion(self, dx: float, dy: float) -> None:
        """EGO-MOTION UZANTI NOKTASI: predict() öncesi kamera hareketini telafi eder.

        (dx, dy): sabit sahnenin bu karede görüntüde kaydığı miktar (px).
        Şimdilik sadece konumu kaydırıyoruz; ileride homografi/ölçek de eklenebilir.
        """
        if self.kf is None or (dx == 0 and dy == 0):
            return
        s = self.kf.statePost.copy()
        s[0, 0] += dx
        s[1, 0] += dy
        self.kf.statePost = s

    def predict(self) -> Box:
        """Bir kare ileri tahmin yapar, tahmini kutuyu döndürür."""
        assert self.kf is not None, "Kalman başlatılmadı"
        s = self.kf.predict()
        return self._state_to_box(s)

    def correct(self, box: Box) -> Box:
        """Ölçümle günceller, düzeltilmiş kutuyu döndürür."""
        assert self.kf is not None, "Kalman başlatılmadı"
        cx, cy = box_center(box)
        m = np.array([[cx], [cy], [box[2]], [box[3]]], dtype=np.float32)
        s = self.kf.correct(m)
        return self._state_to_box(s)

    def current_box(self) -> Box:
        """Son durum (statePost) kutusu."""
        assert self.kf is not None
        return self._state_to_box(self.kf.statePost)

    @staticmethod
    def _state_to_box(s: np.ndarray) -> Box:
        cx, cy = float(s[0, 0]), float(s[1, 0])
        w, h = max(float(s[2, 0]), 1.0), max(float(s[3, 0]), 1.0)  # negatif boyut olmasın
        return (cx - w / 2.0, cy - h / 2.0, w, h)


# ---------------------------------------------------------------------------
# OSTrack sarmalayıcı
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
OSTRACK_CKPT_DRIVE_ID = "1dySfPrSk-knAEo0lTXqZ1kMk3BCvVofV"  # vitb_256_mae_ce_32x4_ep300 resmi ağırlığı


def _is_ostrack_root(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "lib", "test", "evaluation", "tracker.py"))


def find_ostrack_root(given: Optional[str] = None) -> str:
    """OSTrack klasörünü bulur. Verilen yol yanlışsa bilinen konumlara bakar:
    verilen yol (çalışma dizinine ve bu dosyaya göre), hibrit_takip/OSTrack, Avcı Drone/OSTrack."""
    cands: List[str] = []
    if given:
        g = os.path.expanduser(given)
        cands += [os.path.abspath(g), os.path.abspath(os.path.join(_HERE, g))]
    cands += [os.path.join(_HERE, "OSTrack"), os.path.join(os.path.dirname(_HERE), "OSTrack")]
    for c in cands:
        if _is_ostrack_root(c):
            if given and os.path.abspath(c) != cands[0]:
                logger.warning("'%s' yolunda OSTrack yok; %s kullanılıyor.", given, c)
            return os.path.abspath(c)
    tried = "\n  ".join(dict.fromkeys(cands))
    raise FileNotFoundError(f"OSTrack klasörü bulunamadı. Bakılan yerler:\n  {tried}\n"
                            "--ostrack-root ile OSTrack klasörünün yolunu verin.")


def ensure_ostrack_local_files(root: str) -> None:
    """OSTrack'in local.py dosyaları mutlak yol içerir. Eksikse ya da klasör taşınmışsa yeniden üretir."""
    local = os.path.join(root, "lib", "test", "evaluation", "local.py")
    if os.path.isfile(local):
        with open(local, encoding="utf-8") as f:
            if f"'{root}'" in f.read():
                return
    import subprocess
    r = subprocess.run([sys.executable, os.path.join("tracking", "create_default_local_file.py"),
                        "--workspace_dir", root, "--data_dir", os.path.join(root, "data"),
                        "--save_dir", os.path.join(root, "output")],
                       cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"OSTrack local.py oluşturulamadı:\n{r.stderr}")
    logger.info("OSTrack local.py dosyaları bu klasöre göre yeniden oluşturuldu.")


def check_ostrack_checkpoint(root: str, tracker_name: str, param_name: str) -> None:
    """Ağırlık dosyası yoksa nereye konacağını söyleyen anlaşılır bir hata verir."""
    ckpt_dir = os.path.join(root, "output", "checkpoints", "train", tracker_name, param_name)
    if os.path.isdir(ckpt_dir) and any(f.endswith(".pth.tar") for f in os.listdir(ckpt_dir)):
        return
    target = os.path.join(ckpt_dir, "OSTrack_ep0300.pth.tar")
    raise FileNotFoundError(
        f"OSTrack ağırlık dosyası yok. İndirip şuraya koyun:\n  {target}\n"
        f"İndirmek için:  gdown {OSTRACK_CKPT_DRIVE_ID} -O \"{target}\"")

class OSTrackWrapper:
    """botaoye/OSTrack için init/update sarmalayıcı.

    OSTrack RGB bekler; OpenCV BGR verir. Dönüşüm burada yapılır.
    Skor için `ostrack_patch.diff` uygulanmış olmalı; yoksa 1.0 döner ve uyarı loglanır.
    """

    def __init__(
        self,
        ostrack_root: Optional[str] = None,
        tracker_name: str = "ostrack",
        param_name: str = "vitb_256_mae_ce_32x4_ep300",
        device: str = "auto",
    ) -> None:
        """device: auto | cuda | mps (Apple GPU) | cpu. Cihaz seçimi için OSTrack'e
        prepare_ostrack.py uygulanmış olmalı; uygulanmamışsa OSTrack sadece CUDA ile çalışır."""
        # MPS'te desteklenmeyen nadir işlemler CPU'ya düşsün (torch importundan önce ayarlanmalı)
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ["OSTRACK_DEVICE"] = device
        root = find_ostrack_root(ostrack_root)
        logger.info("OSTrack klasörü: %s", root)
        ensure_ostrack_local_files(root)
        check_ostrack_checkpoint(root, tracker_name, param_name)
        if root not in sys.path:
            sys.path.insert(0, root)
        # OSTrack bağımlılığı sadece bu sınıf kullanılırsa yüklenir
        try:
            from lib.utils.device import OST_DEVICE  # type: ignore
        except ImportError:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("OSTrack'e prepare_ostrack.py uygulanmamış; bu haliyle sadece CUDA GPU ile "
                                   "çalışır. Önce: python hibrit_takip/colab/prepare_ostrack.py --ostrack-root ...")
            OST_DEVICE = "cuda"
        self.device = str(OST_DEVICE)
        logger.info("OSTrack cihazı: %s", self.device)
        if self.device == "cpu":
            logger.warning("OSTrack CPU'da çalışıyor; GPU'ya göre yavaş (M2'de ~13 fps, MPS'te ~25 fps).")
        import warnings
        # OSTrack eski timm yollarını kullanıyor; zararsız FutureWarning'leri gizle
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\..*")
        from lib.test.evaluation.tracker import Tracker  # type: ignore

        t = Tracker(tracker_name, param_name, "video")
        params = t.get_parameters()
        params.debug = 0
        params.tracker_name = tracker_name
        params.param_name = param_name
        self._tracker = t.create_tracker(params)
        self._warned_no_score = False

    def init(self, frame_bgr: np.ndarray, box: Box) -> None:
        """Template'i verilen kutuyla başlatır (yeniden başlatma için de kullanılır)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._tracker.initialize(rgb, {"init_bbox": [float(v) for v in box]})

    def update(self, frame_bgr: np.ndarray) -> Tuple[Box, float]:
        """Yeni karede (kutu, skor) döndürür."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = self._tracker.track(rgb)
        box = tuple(float(v) for v in out["target_bbox"])
        score = out.get("best_score")
        if score is None:
            if not self._warned_no_score:
                logger.warning("OSTrack skor döndürmüyor; ostrack_patch.diff uygulanmamış. Skor=1.0 kabul ediliyor.")
                self._warned_no_score = True
            score = 1.0
        return box, float(score)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Hibrit takipçi
# ---------------------------------------------------------------------------

# O karede çıktı kutusunu üreten yöntem (step() çıktısındaki "method" alanı)
METHOD_YOLO = "YOLO"                  # kutu YOLO tespitinden (arama/aday/yeniden başlatma)
METHOD_OSTRACK = "OSTrack"            # kutu OSTrack ölçümünden (Kalman ile yumuşatılmış)
METHOD_OSTRACK_YOLO = "OSTrack+YOLO"  # periyodik doğrulamada ikisi uyuştu, template tazelendi
METHOD_KALMAN = "Kalman"              # ölçüm yok/güvenilmez; kutu Kalman tahmini
METHOD_YOLO_KALMAN = "YOLO+Kalman"    # KAYIP: YOLO, Kalman tahmini çevresinde arıyor

class HybridTracker:
    """YOLO (tespit) + OSTrack (takip) + Kalman (yumuşatma/tahmin) durum makinesi."""

    def __init__(
        self,
        detector: Detector,
        tracker: SingleObjectTracker,
        params: Optional[TrackerParams] = None,
        ego_motion: Optional[EgoMotionEstimator] = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.p = params or TrackerParams()
        self.ego_motion = ego_motion
        self.kalman = KalmanBoxFilter(self.p.kf_q_pos, self.p.kf_q_vel, self.p.kf_r)
        # Durum geçiş geçmişi: (kare, eski, yeni, sebep) — test ve analiz için
        self.history: List[Tuple[int, TrackState, TrackState, str]] = []
        self.frame_idx = -1
        self.reset()

    # ----- genel -----

    def reset(self) -> None:
        """Her şeyi sıfırlar ve ARAMA durumuna döner (frame_idx korunur)."""
        self.state = TrackState.ARAMA
        self.kalman.reset()
        self._cand_box: Optional[Box] = None
        self._cand_conf = 0.0
        self._cand_count = 0
        self._track_frames = 0
        self._low_count = 0
        self._lost_count = 0
        self._last_score = 0.0
        self._method = METHOD_YOLO

    def _transition(self, new: TrackState, reason: str) -> None:
        old = self.state
        if old == new:
            return
        logger.info("kare %d: %s -> %s (%s)", self.frame_idx, old.value, new.value, reason)
        self.history.append((self.frame_idx, old, new, reason))
        self.state = new

    def _detect(self, frame: np.ndarray) -> List[Detection]:
        """Güven eşiğini geçen tespitleri döndürür."""
        dets = self.detector.detect(frame) or []
        return [(tuple(float(v) for v in b), float(c)) for b, c in dets if c >= self.p.det_conf]  # type: ignore[misc]

    def _search_radius(self, box: Box) -> float:
        return max(self.p.lost_search_scale * max(box[2], box[3]), self.p.lost_search_min)

    def _nearest(self, dets: Sequence[Detection], ref: Box, max_dist: float) -> Optional[Detection]:
        """ref kutusunun merkezine max_dist içinde en yakın tespit."""
        best, best_d = None, max_dist
        for d in dets:
            dist = center_dist(d[0], ref)
            if dist <= best_d:
                best, best_d = d, dist
        return best

    def _start_tracking(self, frame: np.ndarray, box: Box, reason: str) -> None:
        self.tracker.init(frame, box)
        self.kalman.init(box)
        self._track_frames = 0
        self._low_count = 0
        self._lost_count = 0
        self._transition(TrackState.TAKIP, reason)

    # ----- ana adım -----

    def step(
        self,
        frame_bgr: np.ndarray,
        timestamp: Optional[float] = None,
        camera_motion: Optional[Tuple[float, float]] = None,
    ) -> dict:
        """Bir kare işler ve takip çıktısını döndürür.

        Args:
            frame_bgr: BGR kare.
            timestamp: Dışarıdan zaman damgası (sunucu saatiyle eşleşme için). None ise time.time().
            camera_motion: (dx, dy) kamera hareketi telafisi. None ise ego_motion tahmincisi
                (varsa) kullanılır.
        """
        self.frame_idx += 1
        ts = time.time() if timestamp is None else timestamp
        H, W = frame_bgr.shape[:2]

        box: Optional[Box] = None
        score = 0.0
        pred_center: Optional[Point] = None
        self._method = METHOD_YOLO  # işleyiciler gerekirse değiştirir

        if self.state == TrackState.ARAMA:
            box, score = self._step_search(frame_bgr)
        elif self.state == TrackState.ADAY:
            box, score = self._step_candidate(frame_bgr)
        else:
            # TAKIP / KAYIP: Kalman predict'ten önce ego-motion telafisi
            if camera_motion is None and self.ego_motion is not None:
                camera_motion = self.ego_motion(frame_bgr)
            if camera_motion is not None:
                self.kalman.apply_camera_motion(*camera_motion)
            pred = self.kalman.predict()
            pred_center = box_center(pred)
            if self.state == TrackState.TAKIP:
                box, score = self._step_track(frame_bgr, pred)
            else:
                box, score = self._step_lost(frame_bgr, pred)

        if box is not None:
            box = clip_box(box, W, H)
        if self.state not in (TrackState.TAKIP, TrackState.KAYIP):
            pred_center = None

        return {
            "state": self.state.value,
            "box": box,
            "score": float(score),
            "predicted_center": pred_center,
            "method": self._method,
            "frame_idx": self.frame_idx,
            "timestamp": ts,
        }

    # ----- durum işleyicileri -----

    def _step_search(self, frame: np.ndarray) -> Tuple[Optional[Box], float]:
        dets = self._detect(frame)
        if not dets:
            return None, 0.0
        b, c = max(dets, key=lambda d: d[1])
        self._cand_box, self._cand_conf, self._cand_count = b, c, 1
        self._transition(TrackState.ADAY, f"tespit conf={c:.2f}")
        return self._maybe_confirm(frame)

    def _step_candidate(self, frame: np.ndarray) -> Tuple[Optional[Box], float]:
        dets = self._detect(frame)
        if not dets:
            self._cand_box, self._cand_count = None, 0
            self._transition(TrackState.ARAMA, "aday doğrulanamadı: tespit yok")
            return None, 0.0
        assert self._cand_box is not None
        near = self._nearest(dets, self._cand_box, self.p.confirm_dist)
        if near is not None:
            self._cand_box, self._cand_conf = near
            self._cand_count += 1
        else:
            # Tutarsızlık: sayaç sıfırlanır, en güvenli tespit yeni aday olur
            self._cand_box, self._cand_conf = max(dets, key=lambda d: d[1])
            self._cand_count = 1
            logger.debug("kare %d: aday tutarsız, sayaç sıfırlandı", self.frame_idx)
        return self._maybe_confirm(frame)

    def _maybe_confirm(self, frame: np.ndarray) -> Tuple[Optional[Box], float]:
        assert self._cand_box is not None
        if self._cand_count >= self.p.confirm_frames:
            box, conf = self._cand_box, self._cand_conf
            self._start_tracking(frame, box, f"{self._cand_count} kare tutarlı tespit")
            self._last_score = conf
            return box, conf
        return self._cand_box, self._cand_conf

    def _step_track(self, frame: np.ndarray, pred: Box) -> Tuple[Optional[Box], float]:
        self._track_frames += 1
        trk_box, trk_score = self.tracker.update(frame)
        low = trk_score < self.p.track_score_min
        periodic = self.p.verify_every > 0 and self._track_frames % self.p.verify_every == 0

        meas: Optional[Box] = trk_box
        score = trk_score
        self._method = METHOD_OSTRACK

        if low or periodic:
            dets = self._detect(frame)
            det = self._nearest(dets, pred, self._search_radius(pred))
            if det is not None and iou(det[0], trk_box) >= self.p.reinit_iou:
                # YOLO ve OSTrack uyuşuyor: template tazelenir, kutu sıkılaşır
                self.tracker.init(frame, det[0])
                meas, score = det[0], max(trk_score, det[1])
                self._low_count = 0
                self._method = METHOD_OSTRACK_YOLO
                logger.debug("kare %d: OSTrack YOLO ile tazelendi (IoU>=%.2f)", self.frame_idx, self.p.reinit_iou)
            elif low and det is not None:
                # OSTrack kaymış ama YOLO Kalman tahmininin yakınında hedefi görüyor
                self.tracker.init(frame, det[0])
                meas, score = det[0], det[1]
                self._low_count = 0
                self._method = METHOD_YOLO
                logger.info("kare %d: OSTrack kaydı (skor=%.2f), YOLO kutusuyla yeniden başlatıldı",
                            self.frame_idx, trk_score)
            elif low:
                meas = None  # bozuk ölçümle Kalman'ı kirletme
                self._low_count += 1
            else:
                self._low_count = 0
        else:
            self._low_count = 0

        if meas is None:
            self._method = METHOD_KALMAN
            if self._low_count >= self.p.low_score_patience:
                self._lost_count = 0
                self._transition(TrackState.KAYIP,
                                 f"{self._low_count} kare düşük skor (son={trk_score:.2f}), YOLO uyuşmadı")
                self._last_score = trk_score
                return None, trk_score
            # Sabır süresi içinde: Kalman tahmini çıktı olarak verilir
            self._last_score = trk_score
            return pred, trk_score

        corrected = self.kalman.correct(meas)
        self._last_score = score
        return (corrected if self.p.output_smoothed else meas), score

    def _step_lost(self, frame: np.ndarray, pred: Box) -> Tuple[Optional[Box], float]:
        self._lost_count += 1
        self._method = METHOD_YOLO_KALMAN
        dets = self._detect(frame)
        det = self._nearest(dets, pred, self._search_radius(pred))
        if det is not None:
            self._start_tracking(frame, det[0], f"KAYIP'ta {self._lost_count}. karede yeniden bulundu")
            self._method = METHOD_YOLO
            self._last_score = det[1]
            return det[0], det[1]
        if self._lost_count >= self.p.lost_max_frames:
            self._transition(TrackState.ARAMA, f"{self._lost_count} kare boyunca bulunamadı")
            self.reset()
        return None, 0.0
