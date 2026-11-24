"""docs/algoritma.pdf dosyasını üretir.

Kullanım (hibrit_takip klasöründen):
    python docs/make_algoritma_pdf.py
Gerekli: reportlab (pip install reportlab). Şekiller docs/img/ altındadır.
"""

from __future__ import annotations

import os
from datetime import date

from reportlab.graphics.shapes import Drawing, Line, PolyLine, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
OUT = os.path.join(HERE, "algoritma.pdf")

# ---------------------------------------------------------------------------
# Yazı tipleri (Türkçe karakterler için TTF şart)
# ---------------------------------------------------------------------------
FONT_DIRS = ["/System/Library/Fonts/Supplemental", "/Library/Fonts", "C:/Windows/Fonts",
             "/usr/share/fonts/truetype/msttcorefonts"]


def _font(name: str) -> str:
    for d in FONT_DIRS:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"{name} bulunamadı (Arial / Courier New gerekli)")


pdfmetrics.registerFont(TTFont("Arial", _font("Arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", _font("Arial Bold.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Italic", _font("Arial Italic.ttf")))
pdfmetrics.registerFont(TTFont("Arial-BoldItalic", _font("Arial Bold Italic.ttf")))
pdfmetrics.registerFont(TTFont("CourierNew", _font("Courier New.ttf")))
pdfmetrics.registerFont(TTFont("CourierNew-Bold", _font("Courier New Bold.ttf")))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic",
                              boldItalic="Arial-BoldItalic")
pdfmetrics.registerFontFamily("CourierNew", normal="CourierNew", bold="CourierNew-Bold",
                              italic="CourierNew", boldItalic="CourierNew-Bold")

# ---------------------------------------------------------------------------
# Renkler ve stiller
# ---------------------------------------------------------------------------
NAVY = colors.HexColor("#1f3b5c")
BLUE = colors.HexColor("#2e75b6")
LIGHT = colors.HexColor("#eef3f9")
GRAY = colors.HexColor("#5a5a5a")
LINEC = colors.HexColor("#c9d3df")
STATE_FILL = {"ARAMA": colors.HexColor("#e7e6e6"), "ADAY": colors.HexColor("#fff2cc"),
              "TAKIP": colors.HexColor("#d9f0d3"), "KAYIP": colors.HexColor("#fbe0cc")}

body = ParagraphStyle("body", fontName="Arial", fontSize=10, leading=14.5, alignment=TA_JUSTIFY,
                      spaceAfter=6)
small = ParagraphStyle("small", parent=body, fontSize=8.8, leading=12, alignment=0, spaceAfter=0)
h1 = ParagraphStyle("h1", fontName="Arial-Bold", fontSize=15, leading=19, textColor=NAVY,
                    spaceBefore=12, spaceAfter=8)
h2 = ParagraphStyle("h2", fontName="Arial-Bold", fontSize=11.5, leading=15, textColor=BLUE,
                    spaceBefore=9, spaceAfter=4)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4, spaceAfter=3)
caption = ParagraphStyle("caption", fontName="Arial-Italic", fontSize=8.8, leading=11.5, textColor=GRAY,
                         alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)
code = ParagraphStyle("code", fontName="CourierNew", fontSize=8.4, leading=11, leftIndent=8,
                      backColor=LIGHT, borderPadding=(6, 6, 6, 6), spaceBefore=6, spaceAfter=10)
title = ParagraphStyle("title", fontName="Arial-Bold", fontSize=22, leading=27, textColor=NAVY,
                       alignment=TA_CENTER, spaceAfter=6)
subtitle = ParagraphStyle("subtitle", fontName="Arial", fontSize=12, leading=16, textColor=GRAY,
                          alignment=TA_CENTER, spaceAfter=2)
note = ParagraphStyle("note", parent=body, backColor=colors.HexColor("#fff8e5"),
                      borderColor=colors.HexColor("#e6c060"), borderWidth=0.6,
                      borderPadding=(6, 7, 6, 7), spaceBefore=6, spaceAfter=10)

W = A4[0] - 4.4 * cm  # metin genişliği


def P(t: str, s: ParagraphStyle = body) -> Paragraph:
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def c(t: str) -> str:
    """Satır içi kod."""
    return f'<font name="CourierNew" size="9" color="#1f3b5c">{t}</font>'


def codeblock(t: str) -> Paragraph:
    t = t.strip("\n").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = t.replace(" ", "&nbsp;").replace("\n", "<br/>")
    return Paragraph(t, code)


def table(rows, widths, header=True, font_size=8.8):
    st = ParagraphStyle("cell", parent=small, fontSize=font_size, leading=font_size + 3)
    sth = ParagraphStyle("cellh", parent=st, fontName="Arial-Bold", textColor=colors.white)
    data = [[Paragraph(str(x), sth if (header and i == 0) else st) for x in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, LINEC),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
             ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
        style += [("BACKGROUND", (0, i), (-1, i), LIGHT) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    t.spaceBefore, t.spaceAfter = 4, 10
    return t


def figure(path: str, width: float, cap: str):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    return KeepTogether([Image(path, width=width, height=width * ih / iw), P(cap, caption)])


# ---------------------------------------------------------------------------
# Diyagram yardımcıları
# ---------------------------------------------------------------------------

def _box(d, x, y, w, h, lines, fill, bold_first=True, fs=8.5):
    d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=fill, strokeColor=NAVY, strokeWidth=0.9))
    n = len(lines)
    top = y + h / 2 + (n - 1) * (fs + 2) / 2 - fs * 0.35
    for i, t in enumerate(lines):
        f = "Arial-Bold" if (i == 0 and bold_first) else "Arial"
        d.add(String(x + w / 2, top - i * (fs + 2), t, fontName=f, fontSize=fs if i == 0 else fs - 1,
                     textAnchor="middle", fillColor=colors.black))


def _arrow(d, pts, color=NAVY, width=1.0):
    d.add(PolyLine(pts, strokeColor=color, strokeWidth=width))
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    L, s = 7, 3.2
    p1 = (x2 - L * math.cos(a) + s * math.sin(a), y2 - L * math.sin(a) - s * math.cos(a))
    p2 = (x2 - L * math.cos(a) - s * math.sin(a), y2 - L * math.sin(a) + s * math.cos(a))
    d.add(Polygon([x2, y2, *p1, *p2], fillColor=color, strokeColor=color, strokeWidth=0.5))


def _label(d, x, y, lines, anchor="middle", fs=7.4, color=GRAY):
    for i, t in enumerate(lines):
        d.add(String(x, y - i * (fs + 1.6), t, fontName="Arial", fontSize=fs, textAnchor=anchor,
                     fillColor=color))


def state_diagram() -> Drawing:
    d = Drawing(470, 245)
    bw, bh = 100, 46
    A, B, T, K = (0, 170), (185, 170), (370, 170), (370, 25)
    _box(d, *A, bw, bh, ["ARAMA", "YOLO her karede"], STATE_FILL["ARAMA"])
    _box(d, *B, bw, bh, ["ADAY", "tutarlılık sayacı"], STATE_FILL["ADAY"])
    _box(d, *T, bw, bh, ["TAKİP", "OSTrack + Kalman"], STATE_FILL["TAKIP"])
    _box(d, *K, bw, bh, ["KAYIP", "Kalman tahmini + YOLO"], STATE_FILL["KAYIP"])
    # ARAMA <-> ADAY
    _arrow(d, [(100, 201), (185, 201)]); _label(d, 142, 206, ["eşik üstü tespit"])
    _arrow(d, [(185, 183), (100, 183)]); _label(d, 142, 172, ["tespit yok"])
    # ADAY -> TAKIP
    _arrow(d, [(285, 193), (370, 193)]); _label(d, 327, 206, ["confirm_frames", "kare tutarlı"])
    _label(d, 235, 160, ["tutarsız tespit → sayaç = 1"])
    # TAKIP <-> KAYIP
    _arrow(d, [(395, 170), (395, 71)]); _label(d, 389, 128, ["low_score_patience kare", "düşük skor ve", "YOLO doğrulamıyor"], anchor="end")
    _arrow(d, [(445, 71), (445, 170)]); _label(d, 451, 128, ["tahmin", "yakınında", "tespit"], anchor="start")
    # KAYIP -> ARAMA
    _arrow(d, [(370, 40), (50, 40), (50, 170)]); _label(d, 210, 28, ["lost_max_frames kare bulunamadı → her şey sıfırlanır"])
    # TAKIP öz-döngü notu
    _label(d, 468, 234, ["skor düşükse / her verify_every karede:", "YOLO doğrulaması, IoU ≥ reinit_iou → tazele"],
           anchor="end")
    return d


def pipeline_diagram() -> Drawing:
    d = Drawing(470, 150)
    bw, bh, gap = 98, 44, 26
    row1 = [["1. Kare", "+ kamera hareketi (dx, dy)"], ["2. Ego-motion", "telafisi (uzantı)"],
            ["3. Kalman", "predict → tahmin kutusu"], ["4. OSTrack", "update → (kutu, skor)"]]
    row2 = [["5. Doğrulama?", "skor < eşik veya periyodik"], ["6. YOLO", "tahmin çevresinde ara"],
            ["7. Karar", "tazele / düzelt / sabır"], ["8. Kalman", "correct → çıktı kutusu"]]
    y1, y2 = 96, 10
    for i, t in enumerate(row1):
        x = i * (bw + gap)
        _box(d, x, y1, bw, bh, t, LIGHT, fs=8.2)
        if i < 3:
            _arrow(d, [(x + bw, y1 + bh / 2), (x + bw + gap, y1 + bh / 2)])
    for i, t in enumerate(row2):
        x = i * (bw + gap)
        _box(d, x, y2, bw, bh, t, STATE_FILL["TAKIP"] if i in (0, 2) else LIGHT, fs=8.2)
        if i < 3:
            _arrow(d, [(x + bw, y2 + bh / 2), (x + bw + gap, y2 + bh / 2)])
    # satır geçişi
    xe = 3 * (bw + gap) + bw / 2
    _arrow(d, [(xe, y1), (xe, 78), (bw / 2, 78), (bw / 2, y2 + bh)])
    _label(d, 2 * (bw + gap) - gap / 2, 82, ["(TAKİP durumunda her kare)"], fs=7)
    return d


# ---------------------------------------------------------------------------
# İçerik
# ---------------------------------------------------------------------------

def build():
    s = []

    # ---- Kapak ----
    s += [Spacer(1, 4.2 * cm),
          P("Hibrit Hedef Takip Algoritması", title),
          P("YOLO + OSTrack + Kalman Filtresi", subtitle),
          P("TEKNOFEST Avcı Drone — Yer İstasyonu Görsel Takip Modülü", subtitle),
          Spacer(1, 1.2 * cm)]
    s.append(table([
        ["Belge", "Algoritma tanımı ve tasarım gerekçeleri"],
        ["Kapsam", "Hedefin görüntüdeki konumu ve güven değeri (kilit kuralları kapsam dışı)"],
        ["Kod", "hibrit_takip/hybrid_tracker.py, run_on_video.py"],
        ["Tarih", date.today().strftime("%d.%m.%Y")],
    ], [3.2 * cm, W - 3.2 * cm], header=False, font_size=9.5))
    s.append(Spacer(1, 1.4 * cm))
    s.append(P("<b>Özet.</b> Modül, dron üzerindeki analog FPV kameradan gelen görüntüde sabit kanatlı hedef "
               "İHA'yı bulur ve takip eder. Tespit için eğitilmiş bir YOLO modeli, kareler arası takip için "
               "Transformer tabanlı OSTrack, kutuyu yumuşatmak ve kopmalarda konumu tahmin etmek için Kalman "
               "filtresi kullanılır. Üç bileşen dört durumlu bir durum makinesiyle (ARAMA → ADAY → TAKİP → "
               "KAYIP) yönetilir. Her karede çıktı: durum, kutu, güven skoru ve Kalman tahmini.", body))
    s.append(PageBreak())

    # ---- 1. Problem ----
    s.append(P("1. Problem ve Kısıtlar", h1))
    s.append(P("Takip modülü yer istasyonunda çalışır. Girdi, dron üzerindeki analog FPV kameradan capture "
               "card ile alınan görüntüdür. Algoritma tasarımını belirleyen koşullar şunlardır:"))
    s += bullets([
        "<b>Küçük hedef:</b> Hedef uzaktayken yalnızca 10–15 piksel. Yaklaştıkça büyür ve döner, yani görünüşü değişir.",
        "<b>Gürültülü görüntü:</b> Analog yayın parazit, kare düşmesi ve gecikme içerir. Tek bir bozuk kare takibi koparmamalıdır.",
        "<b>Kamera hareketi (ego-motion):</b> Kamera dronun üzerinde olduğu için dron manevra yaptığında bütün görüntü kayar.",
        "<b>Tek karelik tespit kabul edilmez:</b> Şartname, hedefin birden fazla ardışık karede tutarlı biçimde görülmesini ister.",
        "<b>Sıkı kutu:</b> Kutunun hedefin en az %90'ını kapsaması gerekir.",
    ])
    s.append(P("<b>Kapsam dışı:</b> AV bölgesi, boyut eşiği, 10 saniyede kümülatif 5 saniye kilit ve 200 ms "
               "tolerans gibi şartname kuralları ayrı bir kilit katmanında uygulanır. Bu modül o katmana "
               "yalnızca \"hedef nerede ve ne kadar eminim\" bilgisini verir.", note))

    # ---- 2. Neden hibrit ----
    s.append(P("2. Neden Hibrit Bir Yaklaşım?", h1))
    s.append(P("Tek başına kullanıldığında her yöntemin bu probleme uymayan bir zayıflığı vardır. Hibrit "
               "yapı, her bileşeni güçlü olduğu işte kullanır ve zayıflığını diğerleriyle kapatır:"))
    s.append(table([
        ["Bileşen", "Güçlü yanı", "Tek başına zayıflığı", "Hibrit yapıda rolü"],
        ["YOLO (tespit)", "Hedefi sıfırdan bulur, kutusu sıkıdır", "Her karede çalıştırmak pahalı; küçük hedefte kareden kareye kaçırır; kimlik takibi yok",
         "Hedefi ilk bulma, periyodik doğrulama, kopmadan sonra yeniden yakalama"],
        ["OSTrack (takip)", "Hızlı; görünüş değişse de aynı nesneyi izler", "Yanlış nesneye kayabilir (drift); kopunca kendi başına geri dönemez",
         "Kareden kareye konum ve güven skoru"],
        ["Kalman filtresi", "Hareket modeliyle ileriyi tahmin eder", "Görüntüye bakmaz; uzun süre ölçümsüz kalırsa sapar",
         "Kutuyu yumuşatma, kısa kopmalarda konum tahmini, YOLO aramasını sınırlama"],
    ], [2.6 * cm, 3.4 * cm, 4.6 * cm, W - 10.6 * cm]))

    # ---- 3. Mimari ----
    s.append(P("3. Genel Mimari", h1))
    s.append(P(f"Modülün tek giriş noktası {c('HybridTracker.step(frame, timestamp, camera_motion)')} "
               "fonksiyonudur. Her kare için bir kez çağrılır ve bir sözlük döndürür. Bileşenler arayüzlerle "
               "ayrılmıştır; bu sayede YOLO ve OSTrack gerçek modeller olmadan, sahte (mock) nesnelerle test edilebilir:"))
    s.append(codeblock("""
detector.detect(frame_bgr)      -> [((x, y, w, h), güven), ...]
tracker.init(frame_bgr, kutu)   -> None        # (yeniden) başlatma
tracker.update(frame_bgr)       -> (kutu, skor)
"""))
    s.append(P(f"Tüm kutular {c('(x, y, w, h)')} biçimindedir: sol üst köşe, genişlik ve yükseklik, piksel "
               "cinsinden. TAKİP durumundaki bir karenin işlenme sırası Şekil 1'de görülüyor."))
    s.append(KeepTogether([pipeline_diagram(),
                           P("Şekil 1 — TAKİP durumunda bir karenin işlenme sırası. Yeşil kutular karar adımlarıdır.", caption)]))

    # ---- 4. Bileşenler ----
    s.append(P("4. Bileşenler", h1))
    s.append(P("4.1 YOLO Dedektörü", h2))
    s.append(P(f"Eğitilmiş YOLO modeli (Ultralytics) her çağrıda karedeki tüm adayları döndürür. Asıl güven "
               f"eşiği ({c('det_conf')}) YOLO'nun içinde değil, takip modülünde uygulanır. Böylece eşik, modeli "
               "yeniden yüklemeden ayarlanabilir. YOLO her karede çalışmaz; çalıştığı durumlar şunlardır:"))
    s += bullets(["ARAMA ve ADAY durumlarında her karede,",
                  f"TAKİP durumunda yalnızca OSTrack skoru düşükken ya da her {c('verify_every')} karede bir,",
                  "KAYIP durumunda her karede, ama sadece Kalman tahmininin çevresindeki tespitler dikkate alınır."])

    s.append(P("4.2 OSTrack", h2))
    s.append(P(f"OSTrack (ECCV 2022), şablon (template) ve arama (search) görüntülerini tek bir Vision "
               f"Transformer'dan birlikte geçiren <i>tek akışlı (one-stream)</i> bir takipçidir. Kullanılan yapılandırma "
               f"{c('vitb_256_mae_ce_32x4_ep300')}:"))
    s.append(table([
        ["Özellik", "Değer", "Anlamı"],
        ["Omurga", "ViT-Base, patch 16, MAE ön eğitimli", "Şablon ve arama bölgesi birlikte işlenir"],
        ["Şablon", "128×128 px, kutunun 2 katı alan", "Hedefin referans görüntüsü (init sırasında alınır)"],
        ["Arama bölgesi", "256×256 px, kutunun 4 katı alan", "Önceki konum çevresinde kırpılan bölge"],
        ["Candidate elimination", "3., 6. ve 9. katmanlarda %70 token tutulur", "Arka plan tokenları atılarak hızlanır"],
        ["Çıktı", "16×16 skor haritası + boyut ve kayma haritaları", "En yüksek skorlu konumdan kutu üretilir"],
    ], [3.3 * cm, 5.3 * cm, W - 8.6 * cm]))
    s.append(P(f"<b>Güven skoru.</b> Orijinal OSTrack kodu skor döndürmez. {c('ostrack_patch.diff')} ile "
               f"{c('track()')} fonksiyonuna şu satır eklenir:"))
    s.append(codeblock("""
response   = hann_penceresi * skor_haritasi   # merkezden uzağı cezalandır
best_score = float(response.max())            # 0..1 arası güven
"""))
    s.append(P("Skor, Hann pencereli haritanın maksimumudur. Hedef görünür ve net olduğunda yüksek, hedef "
               "kaybolduğunda ya da başka bir nesneye kaydığında düşük olur. Durum makinesinin TAKİP → KAYIP "
               "kararı bu değere dayanır."))
    s.append(P(f"<b>Şablon tazeleme.</b> Hedef yaklaştıkça büyür ve döner. İlk karede alınan şablon zamanla "
               f"eskir. YOLO ile OSTrack aynı yeri gösterdiğinde ({c('IoU ≥ reinit_iou')}) OSTrack, YOLO'nun "
               "kutusuyla yeniden başlatılır. Bu hem şablonu tazeler hem de kutuyu sıkılaştırır (%90 kapsama şartı)."))
    s.append(P(f"<b>Cihaz.</b> Orijinal kod yalnızca NVIDIA GPU ile çalışır ({c('.cuda()')}). "
               f"{c('prepare_ostrack.py')} bu çağrıları seçilebilir cihaza çevirir: CUDA, Apple GPU (MPS) veya CPU."))

    s.append(P("4.3 Kalman Filtresi", h2))
    s.append(P(f"Sabit hız modeli kullanılır ({c('cv2.KalmanFilter(8, 4)')}). Zaman adımı 1 karedir:"))
    s.append(codeblock("""
durum   x = [cx, cy, w, h, vx, vy, vw, vh]^T    (merkez, boyut ve hızları)
ölçüm   z = [cx, cy, w, h]^T

F = | I4  I4 |     H = [ I4  0 ]     Q  = diag(q_pos*I4, q_vel*I4)
    | 0   I4 |                       R  = r*I4
                                     P0 = diag(10*I4, 100*I4)

tahmin:     x' = F x            P' = F P F^T + Q
düzeltme:   K  = P' H^T (H P' H^T + R)^-1
            x  = x' + K (z - H x')
"""))
    s.append(P("Filtrenin üç görevi vardır:"))
    s += bullets([
        "<b>Yumuşatma:</b> OSTrack kutusundaki kareden kareye titreşimi azaltır.",
        "<b>Tahmin:</b> Ölçüm olmayan karelerde (bozuk kare, kısa süreli gizlenme) hedefin nerede olması gerektiğini söyler.",
        "<b>Arama alanını sınırlama:</b> YOLO'nun yalnızca tahmin çevresindeki tespitleri dikkate alınır. Böylece uzaktaki başka bir nesne (kuş, başka İHA) hedef sanılmaz.",
    ])
    s.append(P(f"Hız gürültüsü ({c('q_vel')} = 10) konum gürültüsünden ({c('q_pos')} = 1) bilerek yüksek "
               "tutuldu. Dron manevrasında görüntü aniden kayar; filtrenin bu ani hız değişimlerine hızlı "
               "uyum sağlaması gerekir. Ego-motion telafisi eklendiğinde bu değer düşürülebilir."))

    # ---- 5. Durum makinesi ----
    s.append(PageBreak())
    s.append(P("5. Durum Makinesi", h1))
    s.append(KeepTogether([state_diagram(),
                           P("Şekil 2 — Durumlar ve geçiş koşulları. Parametre adları Bölüm 6'daki tablodadır.", caption)]))

    s.append(P("5.1 ARAMA", h2))
    s.append(P(f"YOLO her karede çalışır. {c('det_conf')} eşiğini geçen tespitlerden en güvenlisi aday "
               "olarak seçilir ve ADAY durumuna geçilir. Çıktıda kutu yoktur."))

    s.append(P("5.2 ADAY", h2))
    s.append(P(f"Şartname tek karelik tespiti kabul etmediği için bu durum vardır. Her karede, son aday "
               f"kutusuna en yakın tespit aranır. Merkezler arası mesafe {c('confirm_dist')} pikselden küçükse "
               f"sayaç bir artar. {c('confirm_frames')} kare tutarlı tespit olunca hedef doğrulanır: OSTrack ve "
               "Kalman bu kutuyla başlatılır ve TAKİP'e geçilir."))
    s += bullets([f"Yakın tespit yoksa ama başka tespit varsa: sayaç 1'e döner, en güvenli tespit yeni aday olur.",
                  "Hiç tespit yoksa: ARAMA'ya dönülür."])

    s.append(P("5.3 TAKİP", h2))
    s.append(P(f"Her karede önce Kalman tahmini, sonra OSTrack güncellemesi yapılır. OSTrack skoru "
               f"{c('track_score_min')} altındaysa ya da {c('verify_every')} karede bir YOLO çalıştırılır. "
               "Kalman tahmininin çevresindeki en yakın tespit alınır ve aşağıdaki karar tablosu uygulanır:"))
    s.append(table([
        ["Durum", "Eylem", "Kalman'a giden ölçüm"],
        [f"YOLO kutusu ile OSTrack kutusu örtüşüyor (IoU ≥ {c('reinit_iou')})",
         "OSTrack YOLO kutusuyla yeniden başlatılır (şablon tazelenir). Düşük skor sayacı sıfırlanır.", "YOLO kutusu"],
        ["Skor düşük, YOLO tahmin yakınında ama OSTrack kutusundan farklı yerde",
         "OSTrack başka nesneye kaymış kabul edilir ve YOLO kutusuyla yeniden başlatılır.", "YOLO kutusu"],
        ["Skor düşük, YOLO hiçbir şey bulamadı",
         f"Düşük skor sayacı artar. {c('low_score_patience')} kareye ulaşınca KAYIP'a geçilir. O zamana kadar çıktı Kalman tahminidir.",
         "Yok (bozuk ölçüm filtreyi kirletmesin)"],
        ["Skor yeterli", "Normal takip. Sayaç sıfırlanır.", "OSTrack kutusu"],
    ], [5.0 * cm, 7.4 * cm, W - 12.4 * cm]))
    s.append(P(f"Sabır sayacı ({c('low_score_patience')}) sayesinde analog yayındaki tek bir bozuk kare "
               "takibi düşürmez. Birim testlerde tek bozuk karenin TAKİP durumunu bozmadığı doğrulanmıştır."))

    s.append(P("5.4 KAYIP", h2))
    s.append(P("Kalman her karede tahmin yapmaya devam eder. YOLO her karede çalışır, ama yalnızca tahmin "
               "çevresindeki tespitler kabul edilir:"))
    s.append(codeblock("arama yarıçapı = max(lost_search_scale * max(w, h), lost_search_min)\n"
                       "               = max(2 * büyük kenar, 20 px)      # varsayılanlarla"))
    s.append(P(f"Bu alanda tespit bulunursa OSTrack ve Kalman o kutuyla yeniden başlatılır ve TAKİP'e dönülür. "
               f"{c('lost_max_frames')} kare içinde bulunamazsa her şey sıfırlanır ve ARAMA'ya dönülür. "
               "Kalman yeniden başlatılırken hız sıfırlanır, çünkü uzun bir kopmadan sonra eski hız tahmini güvenilmezdir."))

    # ---- 6. Parametreler ----
    s.append(P("6. Parametreler", h1))
    s.append(P(f"Tümü {c('TrackerParams')} veri sınıfındadır ve {c('run_on_video.py')} komut satırından "
               f"değiştirilebilir (ör. {c('--det-conf 0.4')})."))
    s.append(table([
        ["Parametre", "Varsayılan", "Etkisi ve ayar yönü"],
        ["confirm_frames", "5", "Doğrulama için gereken ardışık tespit. Artırmak yanlış pozitifi azaltır ama kilidin başlamasını geciktirir (25 fps'te 5 kare = 0,2 s)."],
        ["confirm_dist", "40 px", "ADAY'da ardışık tespitler arası izin verilen kayma. Sert manevrada artırılabilir."],
        ["det_conf", "0,5", "YOLO güven eşiği. Uzak ve küçük hedefte 0,3–0,4 denenebilir; yanlış pozitifleri ADAY aşaması süzer."],
        ["verify_every", "5", "TAKİP'te periyodik YOLO doğrulama aralığı. Hedef hızla büyüyüp dönüyorsa 2–3 yapılabilir (daha sık şablon tazeleme)."],
        ["reinit_iou", "0,5", "YOLO-OSTrack uyuşma eşiği. 10–15 px hedefte birkaç piksel fark IoU'yu çok düşürür; 0,3–0,4 daha gerçekçi."],
        ["track_score_min", "0,35", "OSTrack skor eşiği. Kendi videolarındaki skor dağılımına göre ayarlanmalı."],
        ["low_score_patience", "3", "KAYIP'a geçmeden önce tolere edilen ardışık düşük skorlu kare sayısı."],
        ["lost_max_frames", "15", "KAYIP'ta ARAMA'ya dönmeden önce beklenen kare (25 fps'te 0,6 s)."],
        ["lost_search_scale / _min", "2,0 / 20 px", "KAYIP ve doğrulama aramalarının yarıçapı."],
        ["kf_q_pos / kf_q_vel / kf_r", "1 / 10 / 4", "Kalman gürültüleri. Kutu titriyorsa r artırılır; tahmin geride kalıyorsa q_vel artırılır."],
        ["output_smoothed", "True", "Çıktı Kalman kutusu mu, ham kutu mu? %90 kapsama sorun olursa False yapılır."],
    ], [3.8 * cm, 2.0 * cm, W - 5.8 * cm]))

    # ---- 7. Çıktı ve ego-motion ----
    s.append(P("7. Çıktı ve Uzantı Noktaları", h1))
    s.append(P(f"{c('step()')} her karede şu sözlüğü döndürür:"))
    s.append(codeblock("""
{
  "state":            "TAKIP",           # ARAMA | ADAY | TAKIP | KAYIP
  "box":              (x, y, w, h),      # yoksa None; kareye kırpılmış
  "score":            0.71,              # güven skoru
  "predicted_center": (cx, cy),          # Kalman tahmini; TAKİP/KAYIP dışında None
  "method":           "OSTrack",         # kutuyu üreten yöntem (aşağıdaki tablo)
  "frame_idx":        1234,
  "timestamp":        1732441234.567     # dışarıdan verilebilir (sunucu saati)
}
"""))
    s.append(P("Kilit katmanı yalnızca TAKİP durumundaki kutuları dikkate almalıdır. Her durum geçişi kare "
               f"numarası ve sebebiyle loglanır, ayrıca {c('HybridTracker.history')} listesinde tutulur."))
    s.append(P(f"{c('method')} alanı o karede kutuyu hangi yöntemin ürettiğini söyler; demo videosunda durumun "
               "altında yazar ve CSV'ye de kaydedilir:"))
    s.append(table([
        ["method", "Ne zaman"],
        ["YOLO", "ARAMA/ADAY'da tespit; OSTrack kaydığında veya KAYIP'tan dönüşte YOLO ile yeniden başlatma"],
        ["OSTrack", "Normal takip: kutu OSTrack ölçümü (Kalman ile yumuşatılmış)"],
        ["OSTrack+YOLO", "Periyodik doğrulamada ikisi uyuştu, şablon YOLO kutusuyla tazelendi"],
        ["Kalman", "OSTrack skoru düşük, YOLO görmüyor: kutu Kalman tahmini (sabır süresi)"],
        ["YOLO+Kalman", "KAYIP: YOLO, Kalman tahmininin çevresinde arıyor"],
    ], [3.0 * cm, W - 3.0 * cm]))
    s.append(P(f"<b>Ego-motion uzantı noktası.</b> {c('step(..., camera_motion=(dx, dy))')} ile ya da kurucuya "
               f"{c('ego_motion=fonksiyon')} verilerek kamera hareketi bildirilebilir. Değer, Kalman tahmininden "
               "hemen önce durumun konum bileşenine eklenir. (dx, dy), sabit sahnenin o karede görüntüde kaç "
               "piksel kaydığıdır. Tahminci (ör. optik akış veya homografi) henüz yazılmadı; uzantı noktası birim testle doğrulandı."))

    # ---- 8. Doğrulama ----
    s.append(P("8. Doğrulama", h1))
    s.append(P("8.1 Birim testler", h2))
    s.append(P(f"{c('tests/test_state_machine.py')} içindeki 17 test, sahte dedektör ve takipçiyle tüm "
               "durum geçişlerini kontrol eder. Testler gerçek model gerektirmez ve 1 saniyenin altında çalışır:"))
    s += bullets([
        "Tespit yokken ve eşik altı tespitte ARAMA'da kalma",
        "Tam olarak confirm_frames karede doğrulama; tek karelik tespitin reddedilmesi; tutarsızlıkta sayacın sıfırlanması",
        "Sabır süresi dolunca KAYIP'a geçiş; tek bozuk karenin takibi düşürmemesi",
        "KAYIP'ta yakın tespitle geri dönüş; uzak tespitin yok sayılması; zaman aşımında ARAMA'ya dönüş",
        "Periyodik doğrulamada OSTrack'in yeniden başlatılması; dışarıdan zaman damgası; kamera hareketi telafisi; Kalman sabit hız takibi",
        "Her durumda doğru yöntemin (method alanı) raporlanması",
    ])
    s.append(P("8.2 Uçtan uca test (Apple M2, gerçek ağırlıklar)", h2))
    s.append(P("Gerçek OSTrack ağırlıkları ve hazır bir YOLO modeli (COCO yolo11n, hedefe özel eğitilmemiş) ile "
               "sentetik bir video üzerinde tam sistem çalıştırıldı. Videoda parazitli gökyüzü önünde dalgalı "
               "yol izleyen küçük bir uçak (~35×21 px) var. Hedef 40–47. karelerde tamamen gizleniyor."))
    s.append(figure(os.path.join(IMG, "test_kareler.png"), W,
                    "Şekil 3 — run_on_video.py çıktısından kareler. Kırmızı kutu takip kutusu, mavi artı Kalman tahmini."))
    s.append(figure(os.path.join(IMG, "durum_skor.png"), W,
                    "Şekil 4 — Durum ve skor. Hedef gizlenince skor eşiğin altına düşüyor; 3 kare sonra KAYIP'a "
                    "geçiliyor ve hedef geri gelince takip yeniden başlıyor."))
    s.append(KeepTogether([table([
        ["Ölçüt", "Sonuç"],
        ["Tam sistem hızı (YOLO + OSTrack, Apple GPU / MPS)", "18 kare/s"],
        ["Yalnızca OSTrack — Apple GPU (MPS) / CPU", "41 ms (~25 kare/s) / 76 ms (~13 kare/s)"],
        ["TAKİP karelerinde ortalama IoU (gerçek kutuya göre)", "0,92"],
        ["Hedefin kutu içinde kalan oranı — ortalama / en düşük", "%97 / %81"],
        ["Gizlenme sonrası yeniden yakalama", "Hedef göründüğü ilk karede (kare 48)"],
    ], [9.5 * cm, W - 9.5 * cm]),
        P("<b>Önemli:</b> Bu sonuçlar sentetik bir videodan alındı. Gerçek uçuş videosunda parazit, "
          "bulut, güneş yansıması ve ego-motion çok daha zorludur. Takımın kendi YOLO modeli ve gerçek "
          "kayıtlarla parametreler yeniden ayarlanmalıdır.", note)]))

    # ---- 9. Sınırlamalar ----
    s.append(P("9. Bilinen Sınırlamalar ve Sonraki Adımlar", h1))
    s += bullets([
        "<b>Ego-motion tahmincisi yok.</b> Uzantı noktası hazır, ama dron manevrasında görüntü kaymasını hesaplayan modül (optik akış / homografi) yazılmadı. Şu an bu kayma yüksek Kalman hız gürültüsüyle karşılanıyor.",
        "<b>%90 kapsama.</b> Testte bazı karelerde hedefin %81'i kutu içinde kaldı. Kalman yumuşatması hızlı harekette kutuyu geride bırakabilir. Gerekirse output_smoothed=False ya da daha sık YOLO doğrulaması (verify_every) kullanılmalı.",
        "<b>Çok küçük hedef.</b> OSTrack 256 px arama bölgesine ölçekleme yapıyor; 10 px'lik hedefte görüntü bilgisi çok az. Uzak mesafede YOLO'nun yüksek çözünürlükte (imgsz 1280) çalıştırılması önerilir.",
        "<b>Gecikme.</b> Capture card ve analog yayın gecikmesi bu modülde ölçülmüyor. Zaman damgası dışarıdan verilerek kilit katmanında hesaba katılmalı.",
        "<b>Parametreler sentetik veriyle seçildi.</b> Gerçek uçuş kayıtlarıyla ayar yapılmalı (CSV çıktısı ve Colab notebook'undaki analiz grafiği bunun için var).",
    ])
    s.append(P("Ek: Dosyalar", h2))
    s.append(table([
        ["Dosya", "İçerik"],
        ["hybrid_tracker.py", "Arayüzler, TrackerParams, KalmanBoxFilter, OSTrackWrapper, HybridTracker"],
        ["run_on_video.py", "Video üzerinde çalıştırma, çizim, CSV ve video kaydı"],
        ["ostrack_patch.diff", "OSTrack track() fonksiyonuna güven skoru ekler"],
        ["colab/prepare_ostrack.py", "OSTrack'i güncel PyTorch/NumPy'a ve CPU/MPS'e uyarlar"],
        ["kurulum_mac.sh, colab/avci_takip_colab.ipynb", "Mac ve Google Colab kurulumu"],
        ["tests/test_state_machine.py", "Durum makinesi birim testleri"],
        ["docs/make_algoritma_pdf.py", "Bu belgeyi yeniden üretir"],
    ], [6.0 * cm, W - 6.0 * cm]))
    return s


def _page(canv, doc):
    canv.saveState()
    if doc.page > 1:
        canv.setFont("Arial", 8)
        canv.setFillColor(GRAY)
        canv.drawString(2.2 * cm, 1.3 * cm, "Avcı Drone — Hibrit Hedef Takip Algoritması")
        canv.drawRightString(A4[0] - 2.2 * cm, 1.3 * cm, f"{doc.page}")
        canv.setStrokeColor(LINEC)
        canv.line(2.2 * cm, 1.65 * cm, A4[0] - 2.2 * cm, 1.65 * cm)
    canv.restoreState()


if __name__ == "__main__":
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2.0 * cm, bottomMargin=2.2 * cm,
                            title="Hibrit Hedef Takip Algoritması", author="Avcı Drone Takımı",
                            subject="YOLO + OSTrack + Kalman")
    doc.build(build(), onFirstPage=_page, onLaterPages=_page)
    print("Yazıldı:", OUT)
