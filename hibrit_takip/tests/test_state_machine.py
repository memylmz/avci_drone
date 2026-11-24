"""Durum makinesi birim testleri (OSTrack/YOLO gerektirmez)."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hybrid_tracker import (  # noqa: E402
    Box,
    Detection,
    HybridTracker,
    KalmanBoxFilter,
    TrackerParams,
    TrackState,
    box_center,
    iou,
)

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
TARGET: Box = (300.0, 200.0, 20.0, 12.0)


# ---------------------------------------------------------------------------
# Sahte (mock) dedektör ve takipçi
# ---------------------------------------------------------------------------

class MockDetector:
    """Kare indeksine göre önceden yazılmış tespitleri döndürür."""

    def __init__(self, script: Optional[Dict[int, List[Detection]]] = None,
                 default: Optional[List[Detection]] = None) -> None:
        self.script = script or {}
        self.default = default or []
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        out = self.script.get(self.calls, self.default)
        self.calls += 1
        return list(out)


class MockTracker:
    """update() çağrı sırasına göre (kutu, skor) döndürür; init çağrılarını kaydeder."""

    def __init__(self, default: Tuple[Box, float] = (TARGET, 0.9),
                 script: Optional[Dict[int, Tuple[Box, float]]] = None) -> None:
        self.default = default
        self.script = script or {}
        self.updates = 0
        self.inits: List[Box] = []

    def init(self, frame_bgr: np.ndarray, box: Box) -> None:
        self.inits.append(box)

    def update(self, frame_bgr: np.ndarray) -> Tuple[Box, float]:
        out = self.script.get(self.updates, self.default)
        self.updates += 1
        return out


def run(ht: HybridTracker, n: int) -> List[dict]:
    return [ht.step(FRAME, timestamp=float(i)) for i in range(n)]


def make(det: MockDetector, trk: MockTracker, **kw) -> HybridTracker:
    return HybridTracker(det, trk, TrackerParams(**kw))


# ---------------------------------------------------------------------------
# Testler
# ---------------------------------------------------------------------------

def test_no_detection_stays_in_search():
    ht = make(MockDetector(), MockTracker())
    outs = run(ht, 10)
    assert all(o["state"] == "ARAMA" and o["box"] is None for o in outs)
    assert ht.history == []


def test_low_conf_detection_ignored():
    ht = make(MockDetector(default=[(TARGET, 0.3)]), MockTracker())
    assert run(ht, 5)[-1]["state"] == "ARAMA"


def test_candidate_confirms_after_confirm_frames():
    trk = MockTracker()
    ht = make(MockDetector(default=[(TARGET, 0.8)]), trk, confirm_frames=5)
    outs = run(ht, 6)
    states = [o["state"] for o in outs]
    # 1. kare ARAMA->ADAY, 5. karede (indeks 4) doğrulanır
    assert states[:4] == ["ADAY"] * 4
    assert states[4] == "TAKIP"
    assert trk.inits == [TARGET]
    assert outs[5]["predicted_center"] is not None
    assert [(h[1], h[2]) for h in ht.history] == [
        (TrackState.ARAMA, TrackState.ADAY), (TrackState.ADAY, TrackState.TAKIP)]


def test_single_frame_detection_not_accepted():
    det = MockDetector(script={0: [(TARGET, 0.9)]})
    ht = make(det, MockTracker())
    outs = run(ht, 3)
    assert [o["state"] for o in outs] == ["ADAY", "ARAMA", "ARAMA"]


def test_inconsistent_candidate_resets_counter():
    far: Box = (100.0, 100.0, 20.0, 12.0)
    script = {0: [(TARGET, 0.9)], 1: [(TARGET, 0.9)], 2: [(far, 0.9)]}
    ht = make(MockDetector(script=script, default=[(far, 0.9)]), MockTracker(), confirm_frames=3)
    states = [o["state"] for o in run(ht, 5)]
    # 3. karede sıçrama -> sayaç 1'e iner; far ile 3 kare tamamlanınca (indeks 4) TAKIP
    assert states == ["ADAY", "ADAY", "ADAY", "ADAY", "TAKIP"]


def test_low_score_patience_then_lost():
    trk = MockTracker(default=(TARGET, 0.1))
    # Doğrulamadan sonra YOLO hiçbir şey görmüyor
    det = MockDetector(script={i: [(TARGET, 0.9)] for i in range(3)})
    ht = make(det, trk, confirm_frames=3, low_score_patience=3)
    states = [o["state"] for o in run(ht, 7)]
    assert states[2] == "TAKIP"
    assert states[3:5] == ["TAKIP", "TAKIP"]  # sabır süresi
    assert states[5] == "KAYIP"


def test_single_bad_frame_does_not_drop():
    trk = MockTracker(script={2: (TARGET, 0.05)})
    det = MockDetector(script={i: [(TARGET, 0.9)] for i in range(3)})
    ht = make(det, trk, confirm_frames=3)
    outs = run(ht, 10)
    assert all(o["state"] == "TAKIP" for o in outs[2:])


def test_lost_recovers_when_detection_near_prediction():
    trk = MockTracker(default=(TARGET, 0.1))
    script = {i: [(TARGET, 0.9)] for i in range(3)}
    script[8] = [((305.0, 202.0, 20.0, 12.0), 0.9)]  # KAYIP sırasında yakında tekrar görünür
    det = MockDetector(script=script)
    ht = make(det, trk, confirm_frames=3, low_score_patience=3)
    outs = run(ht, 9)
    assert outs[5]["state"] == "KAYIP"
    assert outs[5]["box"] is None and outs[5]["predicted_center"] is not None
    assert outs[8]["state"] == "TAKIP"
    assert len(trk.inits) == 2


def test_lost_ignores_far_detection_and_times_out():
    trk = MockTracker(default=(TARGET, 0.1))
    script = {i: [(TARGET, 0.9)] for i in range(3)}
    det = MockDetector(script=script, default=[])
    ht = make(det, trk, confirm_frames=3, low_score_patience=3, lost_max_frames=4)
    # KAYIP sırasında uzak tespit -> yok sayılmalı
    for i in range(6, 9):
        det.script[i] = [((10.0, 10.0, 20.0, 12.0), 0.9)]
    states = [o["state"] for o in run(ht, 11)]
    # 5. karede KAYIP'a geçilir, ardından 4 kare beklenir, 4.'sünde ARAMA
    assert states[5:10] == ["KAYIP"] * 4 + ["ARAMA"]
    assert ht.kalman.initialized is False


def test_periodic_verify_reinitializes_on_iou_match():
    trk = MockTracker(default=(TARGET, 0.9))
    tight: Box = (301.0, 201.0, 19.0, 11.0)
    assert iou(tight, TARGET) >= 0.5
    det = MockDetector(default=[(tight, 0.9)])
    ht = make(det, trk, confirm_frames=1, verify_every=5)
    run(ht, 11)  # doğrulama sonrası 10 takip karesi -> 2 periyodik doğrulama
    assert len(trk.inits) == 1 + 2
    assert trk.inits[-1] == tight


def test_low_score_with_iou_match_does_not_count_as_bad():
    trk = MockTracker(default=(TARGET, 0.1))
    det = MockDetector(default=[(TARGET, 0.9)])
    ht = make(det, trk, confirm_frames=1, low_score_patience=2)
    outs = run(ht, 10)
    assert all(o["state"] == "TAKIP" for o in outs)


def test_external_timestamp_and_frame_idx():
    ht = make(MockDetector(), MockTracker())
    o = ht.step(FRAME, timestamp=123.456)
    assert o["timestamp"] == 123.456 and o["frame_idx"] == 0
    assert ht.step(FRAME)["frame_idx"] == 1


def test_camera_motion_shifts_prediction():
    trk = MockTracker(default=(TARGET, 0.1))
    det = MockDetector(script={0: [(TARGET, 0.9)]})
    ht = make(det, trk, confirm_frames=1, low_score_patience=1, lost_max_frames=50)
    ht.step(FRAME)          # TAKIP
    ht.step(FRAME)          # düşük skor -> KAYIP
    base = ht.step(FRAME)["predicted_center"]
    shifted = ht.step(FRAME, camera_motion=(30.0, -10.0))["predicted_center"]
    assert shifted[0] - base[0] == pytest.approx(30.0, abs=1.0)
    assert shifted[1] - base[1] == pytest.approx(-10.0, abs=1.0)


def test_kalman_tracks_constant_velocity():
    kf = KalmanBoxFilter()
    kf.init((0.0, 0.0, 10.0, 10.0))
    for t in range(1, 30):
        kf.predict()
        kf.correct((5.0 * t, 0.0, 10.0, 10.0))
    pred = kf.predict()
    assert box_center(pred)[0] == pytest.approx(5.0 * 30 + 5.0, abs=2.0)


def test_method_field_reports_active_source():
    """Çıktıdaki 'method' alanı o karede kutuyu üreten yöntemi gösterir."""
    # 0-2: YOLO ile doğrulama; sonra OSTrack iyi; 3. update'ten itibaren skor düşük
    trk = MockTracker(script={i: (TARGET, 0.1) for i in range(3, 20)})
    det = MockDetector(script={i: [(TARGET, 0.9)] for i in range(3)})
    ht = make(det, trk, confirm_frames=3, low_score_patience=3, verify_every=0)
    outs = run(ht, 9)
    methods = [o["method"] for o in outs]
    states = [o["state"] for o in outs]
    assert methods[:3] == ["YOLO"] * 3          # ARAMA/ADAY: YOLO
    assert methods[3:6] == ["OSTrack"] * 3      # TAKIP, skor iyi
    assert methods[6:8] == ["Kalman", "Kalman"]  # düşük skor, YOLO yok: Kalman tahmini
    assert states[8] == "KAYIP" and methods[8] == "Kalman"  # KAYIP'a geçiş karesi


def test_method_yolo_kalman_in_lost_and_yolo_on_recovery():
    trk = MockTracker(default=(TARGET, 0.1))
    script = {i: [(TARGET, 0.9)] for i in range(3)}
    ht = make(MockDetector(script=script), trk, confirm_frames=3, low_score_patience=1)
    outs = run(ht, 6)
    assert outs[3]["state"] == "KAYIP"
    assert outs[4]["method"] == "YOLO+Kalman"
    # YOLO tekrar gördüğünde yeniden başlatma karesi YOLO olarak işaretlenir
    ht.detector.default = [(TARGET, 0.9)]
    o = ht.step(FRAME)
    assert o["state"] == "TAKIP" and o["method"] == "YOLO"


def test_method_ostrack_yolo_on_periodic_refresh():
    det = MockDetector(default=[(TARGET, 0.9)])
    ht = make(det, MockTracker(), confirm_frames=1, verify_every=2)
    outs = run(ht, 5)
    assert [o["method"] for o in outs[1:5]] == ["OSTrack", "OSTrack+YOLO", "OSTrack", "OSTrack+YOLO"]
