"""Dựng từng khung hình: cảnh tĩnh + ánh lửa chập chờn + ngọn lửa + hơi nước + tàn lửa.

Mọi chuyển động tuần hoàn đúng LOOP giây -> vòng lặp liền mạch.
python render.py scene.npz out.mp4 [số_khung] | python render.py scene.npz --still t out.png
"""
import sys, subprocess
import numpy as np
from scipy.ndimage import map_coordinates, zoom, gaussian_filter
from common import LOOP, FPS, W, H, gust, flicker
from scene import tile_noise

NF = int(LOOP * FPS)
M = None  # bản đồ cảnh, nạp trước khi fork

TH, TW = 512, 512
TEX1 = tile_noise(TH, TW, 1.6, 101)
TEX2 = tile_noise(TH, TW, 1.3, 102)
TEX3 = tile_noise(TH, TW, 1.8, 103)
TEX4 = tile_noise(TH, TW, 1.5, 104)

# vùng lửa (toạ độ đầy đủ), tính ở nửa độ phân giải
FX0, FX1, FY0, FY1 = 560, 1440, 300, 900
FGX = np.arange(FX0, FX1, 2, dtype=np.float32) + .5
FGY = np.arange(FY0, FY1, 2, dtype=np.float32) + .5
FYY, FXX = np.meshgrid(FGY, FGX, indexing="ij")

# vùng hơi nước
SX0, SX1, SY0, SY1 = 600, 1420, 0, 540
SGX = np.arange(SX0, SX1, 2, dtype=np.float32) + .5
SGY = np.arange(SY0, SY1, 2, dtype=np.float32) + .5
SYY, SXX = np.meshgrid(SGY, SGX, indexing="ij")


def speed(k, scale):
    """tốc độ cuộn texture sao cho sau LOOP giây cuộn đúng k vòng."""
    return k * TH / LOOP


def samp(tex, y, x):
    return map_coordinates(tex, [y, x], order=1, mode="grid-wrap")


# bảng màu lửa
_stops = np.array([0, .18, .38, .58, .78, 1.0])
_cols = np.array([[0, 0, 0], [.45, .04, .0], [1.0, .26, .02], [1.0, .55, .08], [1.0, .82, .35], [1.0, .97, .8]])
LUT = np.stack([np.interp(np.linspace(0, 1, 1024), _stops, _cols[:, c]) for c in range(3)], 1).astype(np.float32)


def colormap(v):
    return LUT[np.clip(v * 1023, 0, 1023).astype(np.int32)]


def g1(x, c, s):
    return np.exp(-((x - c) / s) ** 2)


def fire_layer(t, front):
    g = float(gust(t))
    fl = float(flicker(t))
    X, Y = FXX, FYY
    if front:
        base = 838.0
        h = base - Y
        heat_fn = lambda x: (0.75 * g1(x, 920, 70) + 0.95 * g1(x, 1000, 60) + 0.8 * g1(x, 1085, 70)
                             + 0.35 * g1(x, 1000, 200))
        hmax_fn = lambda x: 70 + 75 * g1(x, 1000, 110)
        k1, k2 = 22, 41
    else:
        base = 815.0
        h = base - Y
        heat_fn = lambda x: (1.0 * g1(x, 742, 48) + 1.0 * g1(x, 1258, 48) + 0.7 * g1(x, 1000, 250)
                             + 0.4 * g1(x, 800, 50) + 0.4 * g1(x, 1200, 50))
        hmax_fn = lambda x: 95 + 260 * (g1(x, 735, 55) + g1(x, 1265, 55)) + 70 * g1(x, 1000, 160)
        k1, k2 = 18, 33
    hp = np.clip(h, 0, None)
    # gió đẩy ngọn lửa nghiêng sang phải, thêm lắc nhẹ
    sway = 6 * np.sin(2 * np.pi * 7 * t / LOOP) + 4 * np.sin(2 * np.pi * 17 * t / LOOP + 1)
    lean = (85 * g + sway) * (hp / 260) ** 1.35
    d = samp(TEX3, Y * .5 + speed(26, 0) * t, X * .5) - .5
    xs = X - lean + d * 46 * np.clip(hp / 120, .2, 1)
    n1 = samp(TEX1, Y * .30 + speed(k1, 0) * t, xs * .42 + (300 if front else 0))
    n2 = samp(TEX2, Y * .62 + speed(k2, 0) * t, xs * .9)
    n = .62 * n1 + .38 * n2
    n = (n - .25) / .5
    hmax = hmax_fn(xs) * (1 + .12 * fl + .35 * g)
    heat = heat_fn(xs)
    v = heat * (1.15 - hp / hmax) * 1.25 - (1 - n) * 0.85
    v = np.where(h < 0, v * np.clip(1 + h / 10, 0, 1), v)
    v = np.clip(v, 0, 1)
    if not front:
        # lớp than hồng dưới đáy nồi
        coal = samp(TEX4, Y * 1.2 + speed(3, 0) * t, X * 1.2)
        cm = g1(Y, 812, 16) * g1(X, 1000, 220)
        v = np.maximum(v, cm * (0.35 + 0.55 * coal ** 1.5) * (0.9 + .1 * fl))
    rgb = colormap(v ** 0.9) * (1.6 + 0.15 * fl)
    # bloom
    small = rgb[::4, ::4]
    bl = np.stack([gaussian_filter(small[..., c], 6, mode="constant") for c in range(3)], -1)
    bl = zoom(bl, (rgb.shape[0] / bl.shape[0], rgb.shape[1] / bl.shape[1], 1), order=1)
    rgb = rgb + 0.55 * bl
    return zoom(rgb, (2, 2, 1), order=1)


def steam_layer(t):
    g = float(gust(t))
    X, Y = SXX, SYY
    h = 480 - Y
    hp = np.clip(h, 0, None)
    shear = (220 * g + 25) * (hp / 300) ** 1.3 + 18 * np.sin(2 * np.pi * 3 * t / LOOP) * hp / 300
    xs = X - shear
    n1 = samp(TEX2, Y * .35 + speed(6, 0) * t, xs * .35 + 200)
    n2 = samp(TEX1, Y * .7 + speed(10, 0) * t, xs * .7 + 100)
    n = .6 * n1 + .4 * n2
    width = 230 + 0.55 * hp
    rad = np.exp(-((xs - 1000) / width) ** 2)
    fade = np.clip(hp / 40, 0, 1) * np.clip(1 - hp / 470, 0, 1) ** 1.2
    a = np.clip((n - .38) * 2.6, 0, 1) * rad * fade * (0.42 - 0.18 * g)
    warm = np.clip(1 - hp / 260, 0, 1)[..., None]
    col = warm * np.array([.95, .62, .38]) * (1.0 + .1 * float(flicker(t))) + (1 - warm) * np.array([.34, .37, .46])
    a = zoom(a, 2, order=1)[..., None]
    col = zoom(col, (2, 2, 1), order=1)
    return a, col


# ---------------------------------------------------------------- tàn lửa
NSPARK = 90
_sr = np.random.default_rng(55)
SPARK_M = _sr.integers(32, 75, NSPARK)          # số vòng đời / LOOP
SPARK_PH = _sr.uniform(0, 1, NSPARK)
SPARK_LIFE = _sr.uniform(.45, .95, NSPARK)
yy5, xx5 = np.mgrid[-4:5, -4:5]


def sparks(img, t):
    g = float(gust(t))
    for i in range(NSPARK):
        P = LOOP / SPARK_M[i]
        u = t / P + SPARK_PH[i]
        c = int(np.floor(u)) % SPARK_M[i]
        a = (u - np.floor(u)) * P
        life = SPARK_LIFE[i] * P
        if a > life:
            continue
        r = np.random.default_rng(i * 1000 + c)
        if r.random() < .3 * (1 - g):
            continue  # ít tàn hơn lúc lặng gió
        src = r.choice(3)
        x0 = (742, 1258, 1000)[src] + r.normal(0, 35 if src < 2 else 120)
        y0 = (620, 620, 790)[src] + r.normal(0, 30)
        vy = r.uniform(110, 260)
        wa, wf, wp = r.uniform(8, 35), r.uniform(.6, 2.2), r.uniform(0, 6.3)
        x = x0 + wa * np.sin(2 * np.pi * wf * a + wp) + (r.normal(0, 15) + 160 * g) * a ** 1.4
        y = y0 - vy * a - 25 * a * a
        b = (1 - a / life) ** 1.3 * r.uniform(.6, 1.4)
        sz = r.uniform(.9, 2.0)
        xi, yi = int(x), int(y)
        if not (5 <= xi < W - 5 and 5 <= yi < H - 5):
            continue
        k = np.exp(-((xx5 - (x - xi)) ** 2 + (yy5 - (y - yi)) ** 2) / (2 * sz * sz))[..., None]
        col = LUT[int(np.clip(.45 + .5 * b, 0, 1) * 1023)] * b * 1.6
        img[yi - 4:yi + 5, xi - 4:xi + 5] += k * col


# ---------------------------------------------------------------- khung hình
TONE = None
GRAIN = None


def init_tables():
    global TONE, GRAIN
    x = np.linspace(0, 4, 4096)
    y = 1 - np.exp(-x * 1.08)
    y = y / (1 - np.exp(-4 * 1.08))
    TONE = np.clip(y * 255 + .5, 0, 255).astype(np.uint8)
    r = np.random.default_rng(9)
    GRAIN = (r.normal(0, 0.012, (6, H, W, 1))).astype(np.float32)


def frame(i):
    t = i / FPS
    f1 = 0.92 + 0.10 * float(flicker(t, 0)) + 0.25 * float(gust(t))
    f2 = 0.85 + 0.25 * float(flicker(t, 1))
    f3 = 0.85 + 0.25 * float(flicker(t, 2))
    img = M["A"] + M["B1"] * f1
    img += M["B2"] * f2
    img += M["B3"] * f3
    # lửa sau nồi (bị nồi/củi che)
    bf = fire_layer(t, False)
    occ = 1 - M["fga"][FY0:FY1, FX0:FX1, None]
    img[FY0:FY1, FX0:FX1] += bf * occ
    # hơi nước bốc từ miệng nồi
    a, col = steam_layer(t)
    roi = img[SY0:SY1, SX0:SX1]
    roi *= 1 - a
    roi += a * col
    # lửa phía trước trên đầu củi
    ff = fire_layer(t, True)
    img[FY0:FY1, FX0:FX1] += ff * 0.85
    sparks(img, t)
    img += GRAIN[i % 6]
    idx = np.clip(img * 1024, 0, 4095).astype(np.int32)
    return TONE[idx]


def _work(i):
    return frame(i).tobytes()


def main():
    global M
    M = dict(np.load(sys.argv[1]))
    init_tables()
    if sys.argv[2] == "--still":
        from PIL import Image
        Image.fromarray(frame(int(float(sys.argv[3]) * FPS))).save(sys.argv[4])
        return
    out = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else NF
    from multiprocessing import Pool
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-tune", "grain",
                           "-pix_fmt", "yuv420p", "-g", str(FPS * 4), "-movflags", "+faststart", out],
                          stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for k, b in enumerate(pool.imap(_work, range(n), chunksize=4)):
            ff.stdin.write(b)
            if k % 300 == 0:
                print(f"frame {k}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main()
