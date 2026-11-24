# Hibrit Hedef Takip (YOLO + OSTrack + Kalman)

Bu modül yer istasyonunda çalışır ve her kare için "hedef nerede, ne kadar eminim" bilgisini üretir.
Şartname kilit kuralları (AV bölgesi, boyut eşiği, 10 sn'de 5 sn, 200 ms) bu modülün **dışında**dır.

```
ARAMA ──tespit──▶ ADAY ──confirm_frames tutarlı──▶ TAKIP ──düşük skor × patience──▶ KAYIP
  ▲                 │ tespit yok                    ▲                                 │
  └─────────────────┘                               └──── tahmin çevresinde bulundu ──┤
  ▲                                                                                   │
  └──────────────────────────── lost_max_frames aşıldı ───────────────────────────────┘
```

**Algoritmanın ayrıntılı anlatımı:** [docs/algoritma.pdf](docs/algoritma.pdf) (yeniden üretmek için: `python docs/make_algoritma_pdf.py`).

| Dosya | İçerik |
|---|---|
| `hybrid_tracker.py` | Arayüzler, `TrackerParams`, `KalmanBoxFilter`, `OSTrackWrapper`, `HybridTracker` |
| `ostrack_patch.diff` | OSTrack `track()` fonksiyonuna `best_score` ekler |
| `run_on_video.py` | Video üzerinde demo: çizim + CSV |
| `tests/test_state_machine.py` | Mock dedektör/takipçi ile durum geçişi testleri |

Kutu formatı her yerde `(x, y, w, h)`: sol-üst köşe + genişlik/yükseklik (OSTrack ile aynı).

## Google Colab ile çalıştırma

OSTrack kodu GPU'yu (`.cuda()`) sabit kullanır. Bilgisayarda ekran kartı yoksa Colab kullanın.
Drive'a bir şey yüklemek **zorunlu değil**. OSTrack ağırlığı da otomatik indirilir.

1. Bilgisayarda 3 dosya hazırla: `hibrit_takip.zip` (proje klasörünün bir üstünde), YOLO modelin (`.pt`) ve videon.
2. https://colab.research.google.com → **Dosya → Not defteri yükle** → `hibrit_takip/colab/avci_takip_colab.ipynb`.
3. **Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU**.
4. Hücreleri sırayla çalıştır. 2. hücre "Dosya seç" düğmesiyle sırayla zip'i, modeli ve videoyu ister.
5. 6. hücrede `EK_AYAR` satırına kendi `--imgsz` (ve gerekirse `--classes`) değerini yaz.
6. 7. hücre sonuç CSV'sini ve videoyu bilgisayarına indirir.

Notebook varsayılan olarak Drive kullanır (`USE_DRIVE = True`): dosyaları `MyDrive/avci_drone/` klasörüne koy.
OSTrack ağırlığı ilk çalıştırmada oraya otomatik indirilir. Drive istemezsen `USE_DRIVE = False` yap, dosyalar yükleme penceresiyle istenir.

OSTrack ağırlığı otomatik inmezse (Google indirme kotası): https://drive.google.com/file/d/1dySfPrSk-knAEo0lTXqZ1kMk3BCvVofV/view
adresinden elle indir ve notebook'taki "SADECE otomatik indirme başarısız olduysa" hücresiyle yükle.

Notlar:
- Colab'da ekran olmadığı için `cv2.imshow` kullanılmaz. Sonuç videoya yazılır, sonra tarayıcıda
  oynayabilmesi için ffmpeg ile H.264'e çevrilir (`*_h264.mp4`).
- Oturum kapanınca `/content` silinir (yüklediğin dosyalar dahil). Yeni oturumda hücreleri baştan çalıştır.
  Drive kullanıyorsan dosyalar kalır.
- Kodu bilgisayarda değiştirdiysen zip'i yeniden oluşturup tekrar yükle.
- Ücretsiz Colab'da GPU süresi sınırlı. Parametre denemelerini `--max-frames 300` ile kısa tutun.

`colab/prepare_ostrack.py`, orijinal OSTrack'i güncel Colab ortamına (PyTorch 2.x, NumPy 2.x, timm 1.x) uyarlar:
skor yamasını uygular, `torch._six`, `np.bool`/`np.int` ve `torch.load(weights_only)` sorunlarını giderir,
visdom ve tensorboard'u isteğe bağlı yapar ve `local.py` dosyalarını oluşturur. Tekrar çalıştırmak güvenlidir.

## Kendi bilgisayarında çalıştırma (Mac, Apple Silicon)

Ekran kartı (NVIDIA) gerekmez. OSTrack ve YOLO Mac'in kendi GPU'sunda (MPS) çalışır, o da yoksa CPU'da.

**Kurulum (bir kez):** Terminal'de:

```bash
cd ~/Desktop/Teknofest/"Avcı Drone"/hibrit_takip
bash kurulum_mac.sh
```

Script şunları yapar:
- `Avcı Drone/.venv` sanal ortamını kurar.
- Paketleri yükler.
- OSTrack'i `Avcı Drone/OSTrack` klasörüne indirip hazırlar.
- OSTrack ağırlığını indirir.
- Testleri çalıştırır.

Tekrar çalıştırmak güvenli, hazır olan adımları atlar.

**Her çalıştırmada:**

```bash
cd ~/Desktop/Teknofest/"Avcı Drone"/hibrit_takip
source ../.venv/bin/activate
python run_on_video.py --video VIDEON.mp4 --weights MODELIN.pt --ostrack-root ../OSTrack --show --out-video sonuc.mp4 --csv sonuc.csv
```

- `--show`: canlı pencere açar. `q` veya `Esc` ile kapanır.
- İlk denemede `--max-frames 300` ekle.
- Cihaz seçimi otomatiktir. Elle seçmek için: `--device mps|cpu` (YOLO) ve `--ostrack-device mps|cpu` (OSTrack).

### NVIDIA GPU'lu Linux/Windows bilgisayar

```bash
pip install torch torchvision ultralytics opencv-python easydict timm jpeg4py lmdb pyyaml gdown pytest
git clone https://github.com/botaoye/OSTrack.git
python hibrit_takip/colab/prepare_ostrack.py --ostrack-root OSTrack --save-dir OSTrack/output
gdown 1dySfPrSk-knAEo0lTXqZ1kMk3BCvVofV -O OSTrack/output/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar
```

Patch uygulanmazsa (satır numaraları sürüme göre değişebilir) `lib/test/tracker/ostrack.py` içinde elle yapın:
`response = self.output_window * pred_score_map` satırından sonra `best_score = float(response.max())` ekleyin
ve `track()`'in döndürdüğü her iki sözlüğe `"best_score": best_score` koyun.
Patch'siz çalışır ama skor hep 1.0 olur ve uyarı loglanır; bu durumda düşük skor → KAYIP mantığı devre dışı kalır.

## Çalıştırma

```bash
python run_on_video.py --video ucus.mp4 --weights best.pt --ostrack-root ../OSTrack --csv sonuc.csv --out-video sonuc.mp4
```

- `--show`: canlı pencere açar (sadece ekranlı bilgisayarda; `q`/`Esc` ile çıkılır). Varsayılan: kapalı.
- `--device 0 --half`: YOLO'yu NVIDIA GPU'da FP16 çalıştırır. `--max-frames N`: hızlı deneme.
- `--classes 0`: YOLO sınıf filtresi.
- Bütün `TrackerParams` alanları komut satırından değiştirilebilir: `--confirm-frames 4 --track-score-min 0.3`.
- CSV sütunları: `frame, time_s, state, x, y, w, h, score, method` (kutu yoksa boş).
- Video üzerinde ikinci satırda o karede kutuyu üreten **yöntem** yazar (aşağıdaki tablo).

Testler:

```bash
python -m pytest tests -q
```

## Kod içinden kullanım

```python
from hybrid_tracker import HybridTracker, OSTrackWrapper, TrackerParams

ht = HybridTracker(yolo_detector, OSTrackWrapper("~/OSTrack"), TrackerParams(det_conf=0.4))
out = ht.step(frame_bgr, timestamp=sunucu_zamani, camera_motion=(dx, dy))
# out = {"state", "box", "score", "predicted_center", "method", "frame_idx", "timestamp"}
```

Dedektör `detect(frame_bgr) -> [((x, y, w, h), conf), ...]` metodunu sağlayan herhangi bir nesne olabilir.

**Ego-motion:** `step(..., camera_motion=(dx, dy))` ile ya da `HybridTracker(..., ego_motion=fn)` ile
(fn: kare → (dx, dy)) verilebilir. Kalman `predict()`'ten hemen önce konuma uygulanır.
(dx, dy) = sabit sahnenin bu karede görüntüde kaydığı piksel miktarı. Şimdilik tahminci yok.

## Spesifikasyonda netleştirdiğim noktalar

- **Düşük skorlu kare Kalman'a verilmez.** Sabır süresi (`low_score_patience`) boyunca çıktı kutusu Kalman tahminidir.
  Bozuk OSTrack kutusu filtreyi kirletip KAYIP'taki aramayı da yanlış yere kaydırmasın diye.
- **OSTrack kaymış ama YOLO hedefi görüyorsa:** skor düşük, YOLO tespiti Kalman tahmininin yakınında ama OSTrack kutusuyla
  IoU < `reinit_iou` ise KAYIP'a girmeden doğrudan YOLO kutusuyla yeniden başlatılır. Sonuç aynı ama bir kare kayıp yaşanmaz.
- **TAKİP'te YOLO araması da kapılıdır:** Kalman tahmini etrafında KAYIP'taki yarıçapla (`2×büyük kenar`, en az 20 px)
  aranır. Uzaktaki başka bir nesne aday sayılmaz.
- **Skor yüksekken periyodik doğrulamada IoU tutmazsa** OSTrack'e güvenilir (YOLO başka nesne bulmuş olabilir).
- **KAYIP'tan dönüşte Kalman sıfırdan başlatılır** (hız = 0). Uzun kopmada eski hız tahmini güvenilmez.
- `lost_max_frames`, KAYIP'a geçilen kareden **sonraki** kareleri sayar.
- ADAY durumunda `box` aday kutusu, `score` YOLO güvenidir. Kilit katmanı sadece `TAKIP` durumunu dikkate almalı.

### `method` alanı: o karede hangi yöntem çalışıyor?

| Değer | Ne zaman | Videodaki renk |
|---|---|---|
| `YOLO` | ARAMA/ADAY'da tespit; ya da OSTrack kaydığında veya KAYIP'tan dönüşte YOLO kutusuyla yeniden başlatma karesi | mavi |
| `OSTrack` | Normal takip: kutu OSTrack ölçümü (Kalman ile yumuşatılmış) | yeşil |
| `OSTrack+YOLO` | Periyodik doğrulamada ikisi uyuştu, OSTrack şablonu YOLO kutusuyla tazelendi | camgöbeği |
| `Kalman` | OSTrack skoru düşük ve YOLO görmüyor: kutu Kalman tahmini (sabır süresi) | turuncu |
| `YOLO+Kalman` | KAYIP: YOLO, Kalman tahmininin çevresinde hedefi arıyor | mor |

## Parametre ayarı ipuçları

| Parametre | Varsayılan | İpucu |
|---|---|---|
| `confirm_frames` | 5 | Yanlış pozitif çoksa artırın. Kilit süresinin başlaması bu kadar gecikir (25 fps'te 0.2 s). |
| `confirm_dist` | 40 px | Dron manevra yaparken hedef iki kare arasında çok kayıyorsa artırın. Uzak hedefte (10–15 px) 40 iyi. |
| `det_conf` | 0.5 | Uzak, küçük hedefte YOLO güveni düşer. 0.3–0.4 deneyin; ADAY aşaması yanlış pozitifleri süzer. |
| `verify_every` | 5 | Hedef hızlı büyüyüp dönüyorsa (yaklaşma) 2–3'e indirin: template sık tazelenir, kutu sıkı kalır (%90 kapsama). YOLO maliyeti artar. |
| `reinit_iou` | 0.5 | Küçük hedefte birkaç piksellik fark IoU'yu çok düşürür. 10–15 px hedefte 0.3–0.4 daha gerçekçi. |
| `track_score_min` | 0.35 | Kendi videolarınızda CSV'deki skor dağılımına bakıp ayarlayın: doğru takipte tipik skor ile kopma anındaki skorun arası. |
| `low_score_patience` | 3 | Analog parazit/kare düşmesi çoksa 4–5. Çok büyütmek kopukluğu geç fark ettirir. |
| `lost_max_frames` | 15 | 25 fps'te 0.6 s. Hedef sık sık bulut/güneş içine giriyorsa artırın. |
| `kf_q_vel` | 10 | Dron manevrası sırasında tahmin geride kalıyorsa artırın. Ego-motion telafisi eklenince düşürülebilir. |
| `kf_r` | 4 | Kutu titriyorsa artırın (daha yumuşak), gecikiyorsa azaltın. |
| `output_smoothed` | True | Hızlı manevrada Kalman kutusu gecikip hedefin bir kısmını dışarıda bırakabilir. %90 kapsama sorun olursa `False` yapın. |

Ayar için `--log-level DEBUG` ile çalıştırıp CSV'deki `state`/`score` sütunlarını grafiğe dökmek en hızlı yöntemdir.
