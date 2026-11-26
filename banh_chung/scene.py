"""Vẽ cảnh tĩnh: sân gạch quê, nhà mái rạ, lũy tre, cây nêu, đống rơm,
nồi bánh chưng trên kiềng, củi, cành đào. Xuất ra các bản đồ ánh sáng.

Mỗi lớp có: màu bề mặt (albedo), phát sáng (emissive), mức nhận ánh lửa.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from common import W, H

S = 2  # vẽ siêu mẫu rồi thu nhỏ để khử răng cưa
rng = np.random.default_rng(2026)


def P(pts):
    return [(float(x) * S, float(y) * S) for x, y in pts]


class Layer:
    def __init__(self, recv=1.0, fg=False):
        self.alb = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
        self.emi = Image.new("RGB", (W * S, H * S), (0, 0, 0))
        self.d = ImageDraw.Draw(self.alb)
        self.e = ImageDraw.Draw(self.emi)
        self.recv = recv
        self.fg = fg


def leaf(d, x, y, ang, L, w, col):
    c, s = np.cos(ang), np.sin(ang)
    px, py = -s, c
    pts = [(x, y), (x + c * L * .45 + px * w, y + s * L * .45 + py * w),
           (x + c * L, y + s * L), (x + c * L * .45 - px * w, y + s * L * .45 - py * w)]
    d.polygon(P(pts), fill=col)


def jitter(c, a=8):
    return tuple(int(np.clip(v + rng.integers(-a, a + 1), 0, 255)) for v in c[:3]) + (255,)


# ---------------------------------------------------------------- trời
def sky():
    L = Layer(recv=0.22)
    y = np.linspace(0, 1, H * S)[:, None]
    top, hor = np.array([5, 7, 18]), np.array([30, 30, 52])
    g = top + (hor - top) * (y ** 1.6)[..., None]
    g = np.broadcast_to(g, (H * S, W * S, 3)).astype(np.uint8)
    L.emi = Image.fromarray(g.copy())
    L.e = ImageDraw.Draw(L.emi)
    L.d.rectangle([0, 0, W * S, H * S], fill=(90, 90, 105, 255))
    for _ in range(320):
        x, yy = rng.uniform(0, W), rng.uniform(0, 470) ** 1.0
        b = rng.uniform(40, 200) * (1 - yy / 600)
        r = rng.choice([0.7, 0.9, 1.2, 1.6], p=[.5, .3, .15, .05])
        col = (int(b * .9), int(b * .95), int(b))
        L.e.ellipse([(x - r) * S, (yy - r) * S, (x + r) * S, (yy + r) * S], fill=col)
    return L


# ---------------------------------------------------------------- lũy tre
def bamboo():
    L = Layer(recv=0.35)
    base = 610
    xs = np.linspace(-60, W + 60, 30) + rng.normal(0, 25, 30)
    for cx in xs:
        height = rng.uniform(230, 400)
        if 820 < cx < 1250:
            height *= 0.85
        bend = rng.normal(0, 0.12)
        for k in range(rng.integers(6, 11)):
            x0 = cx + rng.normal(0, 22)
            hk = height * rng.uniform(.7, 1.08)
            lean = rng.normal(0, .12) + bend
            pts = []
            for s in np.linspace(0, 1, 24):
                pts.append((x0 + lean * hk * s + .25 * lean * hk * s * s * 3, base - hk * s))
            shade = int(rng.uniform(13, 22))
            col = (shade - 3, shade + 3, shade, 255)
            L.d.line(P(pts), fill=col, width=int(rng.uniform(2, 4.5) * S))
            for j, s in enumerate(np.linspace(.35, 1, 26)):
                px, py = pts[int(s * 23)]
                for _ in range(4):
                    side = rng.choice([-1, 1])
                    ang = np.pi / 2 - side * rng.uniform(.5, 1.35) + np.pi * (side < 0) * 0
                    ang = (np.pi - rng.uniform(.25, 1.2)) if side < 0 else rng.uniform(.25, 1.2)
                    ang = ang if rng.random() < .8 else -ang * .4
                    leaf(L.d, px, py, ang, rng.uniform(14, 30), rng.uniform(2, 4),
                         (shade - 2, shade + 5, shade + 1, 255))
    # chân bụi tre rậm
    for _ in range(3500):
        x = rng.uniform(-20, W + 20)
        y = base - abs(rng.normal(0, 55))
        shade = int(rng.uniform(10, 19))
        leaf(L.d, x, y, rng.uniform(0, 2 * np.pi), rng.uniform(10, 24), rng.uniform(2, 4),
             (shade - 2, shade + 4, shade, 255))
    return L


# ---------------------------------------------------------------- sân gạch
def yard():
    L = Layer(recv=1.0)
    top, bot = 600, H
    L.d.rectangle(P([(0, top), (W, bot)]), fill=(60, 44, 36, 255))
    vx, vy = 990, 180
    K = 16
    ys = [top + (bot - top + 60) * (k / K) ** 1.7 for k in range(K + 1)]
    xb = np.arange(-3200, 5200, 150)

    def xat(xbot, y):
        return vx + (xbot - vx) * (y - vy) / (bot - vy)

    for r in range(K):
        y0, y1 = ys[r], ys[r + 1]
        off = 75 * (r % 2)
        for xi in xb:
            a, b = xi + off, xi + off + 150
            quad = [(xat(a, y0), y0), (xat(b, y0), y0), (xat(b, y1), y1), (xat(a, y1), y1)]
            if quad[1][0] < -50 or quad[0][0] > W + 50:
                continue
            base = np.array([88, 56, 44]) * rng.uniform(.8, 1.06)
            if rng.random() < .12:  # rêu mốc
                base = base * np.array([.75, .95, .8])
            col = tuple(int(v) for v in base) + (255,)
            L.d.polygon(P(quad), fill=col, outline=(52, 40, 34, 255))
            # vạch vữa đậm hơn
    for r in range(K + 1):
        L.d.line(P([(0, ys[r]), (W, ys[r])]), fill=(50, 38, 32, 255),
                 width=max(1, int(S * (1 + 2.5 * (ys[r] - top) / (bot - top)))))
    return L


# ---------------------------------------------------------------- nhà
def house():
    L = Layer(recv=0.85)
    d, e = L.d, L.e
    # nền hè
    d.polygon(P([(20, 640), (860, 640), (875, 662), (5, 662)]), fill=(88, 82, 74, 255))
    # tường đất
    d.rectangle(P([(50, 470), (830, 642)]), fill=(122, 94, 62, 255))
    for _ in range(900):  # vết đất loang
        x, y = rng.uniform(50, 830), rng.uniform(470, 640)
        r = rng.uniform(3, 14)
        c = jitter((112, 86, 58), 12)
        d.ellipse(P([(x - r, y - r * .6), (x + r, y + r * .6)]), fill=c)
    # bóng hiên
    for i in range(40):
        a = int(200 * (1 - i / 40) ** 1.5)
        d.line(P([(50, 470 + i), (830, 470 + i)]), fill=(25, 18, 12, a), width=S)
    # cửa chính mở, có đèn dầu bên trong
    d.rectangle(P([(365, 505), (495, 642)]), fill=(22, 14, 9, 255))
    glow = Image.new("RGB", (W * S, H * S))
    gd = ImageDraw.Draw(glow)
    for i in range(30, 0, -1):
        r = i * 5
        c = int(185 * (1 - i / 30) ** 1.4)
        gd.ellipse(P([(455 - r, 585 - r), (455 + r, 585 + r)]), fill=(c, int(c * .55), int(c * .2)))
    mask = Image.new("L", (W * S, H * S))
    ImageDraw.Draw(mask).rectangle(P([(365, 505), (495, 642)]), fill=255)
    L.emi.paste(glow, (0, 0), mask)
    # bàn thờ thấp thoáng trong nhà
    d.rectangle(P([(395, 560), (470, 568)]), fill=(70, 30, 15, 255))
    d.rectangle(P([(405, 568), (412, 642)]), fill=(40, 20, 12, 255))
    d.rectangle(P([(455, 568), (462, 642)]), fill=(40, 20, 12, 255))
    e.ellipse(P([(450, 548), (458, 556)]), fill=(255, 190, 90))  # ngọn đèn
    # cánh cửa gỗ
    d.polygon(P([(365, 505), (340, 512), (340, 642), (365, 642)]), fill=(60, 38, 22, 255))
    d.polygon(P([(495, 505), (520, 512), (520, 642), (495, 642)]), fill=(60, 38, 22, 255))
    # câu đối đỏ
    for x in (312, 528):
        d.rectangle(P([(x, 508), (x + 24, 628)]), fill=(168, 32, 26, 255))
        for k in range(6):
            yy = 516 + k * 19
            d.rectangle(P([(x + 7, yy), (x + 17, yy + 10)]), fill=(150, 112, 40, 255))
    # cửa sổ chấn song
    for (x0, x1) in ((120, 245), (620, 745)):
        d.rectangle(P([(x0, 520), (x1, 595)]), fill=(55, 36, 22, 255))
        d.rectangle(P([(x0 + 7, 527), (x1 - 7, 588)]), fill=(18, 12, 8, 255))
        e.rectangle(P([(x0 + 7, 527), (x1 - 7, 588)]), fill=(38, 20, 8))
        for xx in np.linspace(x0 + 14, x1 - 14, 7):
            d.rectangle(P([(xx - 2.5, 527), (xx + 2.5, 588)]), fill=(62, 40, 24, 255))
    # cột hiên
    for x in (60, 300, 560, 805):
        d.rectangle(P([(x - 8, 478), (x + 8, 645)]), fill=(66, 42, 26, 255))
        d.rectangle(P([(x - 12, 640), (x + 12, 650)]), fill=(80, 74, 66, 255))
    # mái rạ
    roof = [(-10, 492), (875, 492), (760, 318), (100, 318)]
    d.polygon(P(roof), fill=(120, 96, 56, 255))
    for _ in range(5000):
        t = rng.uniform(0, 1)
        y0 = 318 + (492 - 318) * t
        xl = 100 + (-10 - 100) * t
        xr = 760 + (875 - 760) * t
        x = rng.uniform(xl, xr)
        ln = rng.uniform(10, 32)
        c = jitter((112, 88, 50), 22)
        d.line(P([(x, y0), (x + rng.normal(0, 2), y0 + ln)]), fill=c, width=S)
    # mép rạ lởm chởm
    for x in np.arange(-10, 876, 3):
        ln = rng.uniform(2, 14)
        d.line(P([(x, 490), (x + rng.normal(0, 1.5), 490 + ln)]), fill=jitter((96, 74, 42), 14), width=S)
    # nóc
    d.polygon(P([(95, 312), (765, 312), (760, 326), (100, 326)]), fill=(72, 58, 36, 255))
    return L


# ---------------------------------------------------------------- đống rơm
def haystack():
    L = Layer(recv=1.0)
    cx, base = 1475, 655
    pts = []
    for a in np.linspace(0, np.pi, 60):
        r = 125 + rng.normal(0, 2)
        pts.append((cx + r * np.cos(a), base - 165 * np.sin(a) ** 0.85))
    L.d.polygon(P(pts), fill=(128, 102, 58, 255))
    for _ in range(2500):
        a = rng.uniform(0.05, np.pi - .05)
        rr = rng.uniform(0, 1) ** .5
        x = cx + 120 * rr * np.cos(a)
        y = base - 158 * rr * np.sin(a) ** .85
        ang = rng.normal(np.pi / 2, .5)
        ln = rng.uniform(8, 22)
        L.d.line(P([(x, y), (x + np.cos(ang) * ln, y + np.sin(ang) * ln)]),
                 fill=jitter((118, 92, 50), 24), width=S)
    L.d.rectangle(P([(cx - 3, base - 215), (cx + 3, base - 150)]), fill=(60, 44, 28, 255))
    L.d.polygon(P([(cx - 30, base - 158), (cx + 30, base - 158), (cx + 8, base - 178), (cx - 8, base - 178)]),
                fill=(96, 74, 40, 255))
    return L


# ---------------------------------------------------------------- cây nêu
def cay_neu():
    L = Layer(recv=0.8)
    pts = [(1700 + 14 * (1 - s) ** 2 - 10 * s, 655 - 600 * s) for s in np.linspace(0, 1, 30)]
    L.d.line(P(pts), fill=(78, 82, 52, 255), width=int(7 * S))
    for s in np.linspace(.05, .95, 14):  # đốt tre
        x, y = pts[int(s * 29)]
        L.d.line(P([(x - 5, y), (x + 5, y)]), fill=(50, 54, 34, 255), width=S * 2)
    tx, ty = pts[-1]
    for _ in range(70):  # ngọn tre còn lá
        ang = rng.uniform(-np.pi * .95, -np.pi * .05) if rng.random() < .3 else rng.uniform(.1, np.pi - .1)
        leaf(L.d, tx + rng.normal(0, 6), ty + rng.normal(0, 8), ang, rng.uniform(25, 48), rng.uniform(3, 6),
             jitter((40, 60, 34), 8))
    # dây treo khánh đất, chuông gió, lá phướn đỏ
    hx, hy = tx - 2, ty + 18
    L.d.line(P([(hx, hy), (hx - 40, hy + 120)]), fill=(140, 120, 80, 255), width=S)
    for k, s in enumerate(np.linspace(.25, 1, 4)):
        x, y = hx - 40 * s, hy + 120 * s
        L.d.ellipse(P([(x - 5, y - 4), (x + 5, y + 6)]), fill=(130, 85, 50, 255))
    L.d.polygon(P([(hx - 40, hy + 120), (hx - 28, hy + 165), (hx - 50, hy + 160)]), fill=(170, 34, 28, 255))
    L.d.line(P([(hx + 4, hy + 6), (hx + 26, hy + 70)]), fill=(140, 120, 80, 255), width=S)
    L.d.polygon(P([(hx + 20, hy + 70), (hx + 34, hy + 70), (hx + 30, hy + 98), (hx + 24, hy + 98)]),
                fill=(180, 140, 40, 255))
    return L


# ---------------------------------------------------------------- kiềng đá + nồi + củi
POT_CX, RIM_Y = 1000, 468


def pot_r(y):
    if y < 560:
        return 255 + 30 * np.sin((y - RIM_Y) / (560 - RIM_Y) * np.pi / 2)
    u = (y - 560) / 238.0
    return 285 * np.sqrt(max(0.0, 1 - u ** 2)) ** 0.8


def stones():
    L = Layer(recv=1.0, fg=True)
    for (x0, x1, y0, y1) in ((735, 805, 748, 838), (1195, 1265, 748, 838), (965, 1035, 778, 820)):
        pts = [(x0 + 6, y0), (x1 - 4, y0 + 3), (x1, y1), (x0, y1)]
        L.d.polygon(P(pts), fill=(46, 42, 38, 255))
        L.d.polygon(P([(x0 + 6, y0), (x1 - 4, y0 + 3), (x1 - 6, y0 + 14), (x0 + 4, y0 + 12)]), fill=(62, 56, 50, 255))
    return L


def pot():
    L = Layer(recv=1.0, fg=True)
    d = L.d
    ys = np.arange(RIM_Y, 799, 1.0)
    left = [(POT_CX - pot_r(y), y) for y in ys]
    right = [(POT_CX + pot_r(y), y) for y in ys[::-1]]
    body = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    bd = ImageDraw.Draw(body)
    for xs in np.arange(POT_CX - 290, POT_CX + 291, 0.5):
        u = (xs - POT_CX) / 288.0
        nrm = np.sqrt(max(0.0, 1 - u * u))
        spec = 0.9 * np.exp(-((u + 0.52) / 0.07) ** 2) + 0.35 * np.exp(-((u - 0.6) / 0.05) ** 2)
        c = np.array([44, 36, 30]) * (0.55 + 0.45 * nrm) + np.array([70, 58, 44]) * spec
        bd.line([(xs * S, RIM_Y * S), (xs * S, 800 * S)], fill=tuple(int(v) for v in c) + (255,), width=S)
    soot = Image.new("RGBA", body.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(soot)
    for y in np.arange(600, 800, 0.5):  # muội than dày dần xuống đáy
        a = int(235 * ((y - 600) / 200) ** 1.4)
        sd.line([(0, y * S), (W * S, y * S)], fill=(14, 11, 10, a), width=S)
    body = Image.alpha_composite(body, soot)
    m = Image.new("L", (W * S, H * S))
    ImageDraw.Draw(m).polygon(P(left + right), fill=255)
    L.alb.paste(body, (0, 0), Image.composite(body, Image.new("RGBA", body.size), m).split()[3])
    # quai nồi
    for sgn in (-1, 1):
        x = POT_CX + sgn * 268
        d.ellipse(P([(x - 16, 495), (x + 16, 530)]), outline=(40, 34, 30, 255), width=int(6 * S))
    # miệng nồi
    d.ellipse(P([(POT_CX - 266, RIM_Y - 40), (POT_CX + 266, RIM_Y + 40)]), fill=(70, 58, 46, 255))
    d.ellipse(P([(POT_CX - 250, RIM_Y - 33), (POT_CX + 250, RIM_Y + 31)]), fill=(22, 24, 16, 255))
    # lá dong phủ trên mặt bánh
    inner = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    idr = ImageDraw.Draw(inner)
    for _ in range(46):
        x = POT_CX + rng.uniform(-235, 235)
        y = RIM_Y + rng.uniform(-28, 26)
        ang = rng.uniform(-.6, .6) + (np.pi if rng.random() < .5 else 0)
        Ln, w = rng.uniform(110, 170), rng.uniform(26, 38)
        col = jitter((48, 96, 44) if rng.random() < .6 else (70, 120, 56), 10)
        c, s = np.cos(ang), np.sin(ang) * .32
        x -= c * Ln / 2
        y -= s * Ln / 2
        leaf(idr, x, y, np.arctan2(s, c), Ln, w * .45, col)
        idr.line(P([(x, y), (x + c * Ln, y + s * Ln)]), fill=(100, 140, 80, 255), width=S)
    m = Image.new("L", (W * S, H * S))
    ImageDraw.Draw(m).ellipse(P([(POT_CX - 248, RIM_Y - 31), (POT_CX + 248, RIM_Y + 29)]), fill=255)
    L.alb.paste(inner, (0, 0), Image.composite(inner, Image.new("RGBA", inner.size), m).split()[3])
    # vài lá rủ qua miệng nồi
    for x in (790, 905, 1080, 1190):
        pts = [(x - 30, RIM_Y + 18), (x + 28, RIM_Y + 20), (x + 18, RIM_Y + 62), (x - 4, RIM_Y + 78), (x - 24, RIM_Y + 55)]
        d.polygon(P(pts), fill=jitter((52, 100, 46), 10))
        d.line(P([(x, RIM_Y + 18), (x - 3, RIM_Y + 74)]), fill=(96, 138, 80, 255), width=S)
    # lạt buộc / mép nồi phía trước sáng hơn
    d.arc(P([(POT_CX - 266, RIM_Y - 40), (POT_CX + 266, RIM_Y + 40)]), 0, 180, fill=(92, 76, 60, 255), width=int(4 * S))
    return L


def logs():
    L = Layer(recv=1.0, fg=True)
    d, e = L.d, L.e
    spec = [((905, 808), (630, 905), 44), ((1100, 808), (1370, 902), 42),
            ((955, 820), (815, 975), 54), ((1050, 820), (1170, 980), 52), ((1000, 826), (985, 1010), 48)]
    for (ix, iy), (ox, oy), w in spec:
        dx, dy = ox - ix, oy - iy
        ln = np.hypot(dx, dy)
        ux, uy = dx / ln, dy / ln
        px, py = -uy, ux
        wi, wo = w * .8, w
        poly = [(ix + px * wi / 2, iy + py * wi / 2), (ox + px * wo / 2, oy + py * wo / 2),
                (ox - px * wo / 2, oy - py * wo / 2), (ix - px * wi / 2, iy - py * wi / 2)]
        d.polygon(P(poly), fill=(66, 46, 32, 255))
        for k in range(9):  # vân vỏ cây
            o = rng.uniform(-.45, .45) * w
            a, b = rng.uniform(.15, .5), rng.uniform(.55, 1)
            d.line(P([(ix + dx * a + px * o, iy + dy * a + py * o), (ix + dx * b + px * o, iy + dy * b + py * o)]),
                   fill=jitter((44, 30, 22), 6), width=S)
        # đầu khúc củi (mặt cắt)
        ew, eh = wo / 2, wo / 2 * .62
        d.ellipse(P([(ox - ew, oy - eh), (ox + ew, oy + eh)]), fill=(146, 106, 66, 255))
        for rr in (.72, .45, .2):
            d.ellipse(P([(ox - ew * rr, oy - eh * rr), (ox + ew * rr, oy + eh * rr)]), outline=(118, 84, 52, 255), width=S)
        # phần cháy đen & than hồng ở đầu trong
        for k in range(26):
            a = rng.uniform(0, .38)
            o = rng.uniform(-.4, .4) * wi
            x, y = ix + dx * a + px * o, iy + dy * a + py * o
            r = rng.uniform(4, 11)
            d.ellipse(P([(x - r, y - r), (x + r, y + r)]), fill=(20, 15, 12, 255))
            if a < .3 and rng.random() < .75:
                hot = (1 - a / .3)
                c = (int(255 * hot), int(95 * hot ** 1.5), int(20 * hot ** 2))
                e.ellipse(P([(x - r * .7, y - r * .7), (x + r * .7, y + r * .7)]), fill=c)
    return L


def woodpile():
    L = Layer(recv=1.0, fg=True)
    d = L.d
    d.polygon(P([(30, 1080), (30, 870), (120, 800), (470, 800), (520, 860), (540, 1080)]), fill=(34, 24, 18, 255))
    rows = [(1028, 6, 48, 70), (948, 5, 46, 118), (870, 4, 44, 160), (800, 2, 40, 210)]
    for y, n, r, x0 in rows:
        for k in range(n):
            x = x0 + k * r * 1.95 + rng.normal(0, 5)
            rr = r * rng.uniform(.88, 1.05)
            yy = y + rng.normal(0, 4)
            d.ellipse(P([(x - rr - 4, yy - rr * .9 - 4), (x + rr + 4, yy + rr * .9 + 4)]), fill=(52, 36, 26, 255))
            d.ellipse(P([(x - rr, yy - rr * .9), (x + rr, yy + rr * .9)]), fill=jitter((150, 110, 70), 10))
            for q in (.75, .52, .3, .12):
                d.ellipse(P([(x - rr * q, yy - rr * .9 * q), (x + rr * q, yy + rr * .9 * q)]),
                          outline=(122, 88, 56, 255), width=S)
            d.line(P([(x, yy), (x + rr * .8, yy - rr * .3)]), fill=(100, 70, 45, 255), width=S)
    return L


def peach():
    L = Layer(recv=3.2, fg=True)
    d = L.d

    def branch(x, y, ang, ln, w, depth):
        pts = [(x, y)]
        for _ in range(8):
            ang += rng.normal(0, .18)
            x += np.cos(ang) * ln / 8
            y += np.sin(ang) * ln / 8
            pts.append((x, y))
        d.line(P(pts), fill=(46, 30, 24, 255), width=max(S, int(w * S)), joint="curve")
        tips.extend(pts[2:])
        if depth > 0:
            for k in range(rng.integers(2, 4)):
                i = rng.integers(2, len(pts))
                bx, by = pts[i]
                branch(bx, by, ang + rng.choice([-1, 1]) * rng.uniform(.4, .9), ln * rng.uniform(.45, .7), w * .6, depth - 1)

    tips = []
    branch(1960, 40, np.pi * .86, 420, 15, 3)
    branch(1960, 160, np.pi * .93, 260, 9, 2)
    for (x, y) in tips:
        if x > W + 10 or y > 560:
            continue
        for _ in range(rng.integers(0, 3)):
            fx, fy = x + rng.normal(0, 9), y + rng.normal(0, 9)
            r = rng.uniform(5, 10)
            if rng.random() < .25:  # nụ
                d.ellipse(P([(fx - r * .45, fy - r * .55), (fx + r * .45, fy + r * .55)]), fill=jitter((200, 70, 110), 12))
                continue
            base = np.array([232, 120, 150]) * rng.uniform(.85, 1.05)
            for k in range(5):
                a = k * 2 * np.pi / 5 + rng.uniform(0, 1)
                px, py = fx + np.cos(a) * r * .55, fy + np.sin(a) * r * .55
                d.ellipse(P([(px - r * .5, py - r * .5), (px + r * .5, py + r * .5)]), fill=jitter(base, 14))
            d.ellipse(P([(fx - r * .22, fy - r * .22), (fx + r * .22, fy + r * .22)]), fill=(150, 40, 70, 255))
        if rng.random() < .3:
            leaf(d, x, y, rng.uniform(0, 2 * np.pi), rng.uniform(12, 20), 3, jitter((60, 90, 40), 10))
    return L


# ---------------------------------------------------------------- ghép & ánh sáng
def to_np(img):
    return np.asarray(img.resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0


def tile_noise(h, w, beta, seed):
    r = np.random.default_rng(seed)
    F = np.fft.fft2(r.normal(size=(h, w)))
    fy = np.fft.fftfreq(h)[:, None] * h
    fx = np.fft.fftfreq(w)[None, :] * w
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1
    F /= f ** beta
    F[0, 0] = 0
    x = np.real(np.fft.ifft2(F))
    x = (x - x.min()) / (x.max() - x.min())
    return x.astype(np.float32)


def build():
    layers = [sky(), bamboo(), yard(), house(), haystack(), cay_neu(), stones(), pot(), logs(), woodpile(), peach()]
    alb = np.zeros((H, W, 3), np.float32)
    emi = np.zeros((H, W, 3), np.float32)
    recv = np.zeros((H, W), np.float32)
    fga = np.zeros((H, W), np.float32)
    for L in layers:
        a = to_np(L.alb)
        al, c = a[..., 3], a[..., :3]
        e = to_np(L.emi)
        alb = alb * (1 - al[..., None]) + c * al[..., None]
        emi = emi * (1 - al[..., None]) + e * al[..., None]
        recv = recv * (1 - al) + L.recv * al
        if L.fg:
            fga = fga + al * (1 - fga)
    # vân bề mặt li ti
    tex = tile_noise(H, W, 1.1, 5)
    alb *= (0.86 + 0.28 * tex)[..., None]

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    def glow(cx, cy, R, sx=1.3):
        return 1.0 / (1.0 + ((xx - cx) / (R * sx)) ** 2 + ((yy - cy) / R) ** 2)

    vign = 1 - 0.55 * (((xx - W / 2) / (W * .62)) ** 2 + ((yy - H / 2) / (H * .7)) ** 2)
    vign = np.clip(vign, 0, 1) ** 1.3
    amb = np.array([0.16, 0.18, 0.30], np.float32)
    warm = np.array([1.0, 0.50, 0.19], np.float32)

    g1 = 2.6 * glow(1000, 800, 190) ** 1.25
    g2 = 0.9 * glow(745, 690, 150, 1.1)
    g3 = 0.9 * glow(1255, 690, 150, 1.1)
    lit = alb * recv[..., None]
    A = (alb * amb + emi) * vign[..., None]
    haze = (0.10 * glow(1000, 650, 260) ** 1.5)[..., None] * warm
    B1 = (lit * g1[..., None] * warm + haze) * vign[..., None]
    B2 = lit * g2[..., None] * warm * vign[..., None]
    B3 = lit * g3[..., None] * warm * vign[..., None]
    # than hồng ở đầu củi chập chờn theo lửa: đưa vào B1
    tips = to_np(layers[8].emi) * to_np(layers[8].alb)[..., 3:4]
    A -= tips * vign[..., None] * 0.6
    B1 += tips * vign[..., None] * 0.6
    return dict(A=A, B1=B1, B2=B2, B3=B3, fga=fga, vign=vign)


if __name__ == "__main__":
    import sys
    m = build()
    np.savez(sys.argv[1] if len(sys.argv) > 1 else "scene.npz", **m)
    img = np.clip(m["A"] + m["B1"] + m["B2"] + m["B3"], 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save("scene_preview.png")
