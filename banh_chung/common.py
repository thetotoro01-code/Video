"""Thông số dùng chung cho hình và tiếng: độ dài vòng lặp, gió mùa, nhịp lửa.

Mọi hàm theo thời gian ở đây đều tuần hoàn với chu kỳ LOOP giây, nên
video/âm thanh nối đuôi nhau liền mạch khi lặp lại.
"""
import numpy as np

LOOP = 120.0          # giây / vòng lặp
FPS = 30
W, H = 1920, 1080
SR = 48000

# Các cơn gió mùa đông bắc: (thời điểm bắt đầu, độ dài, cường độ)
GUSTS = [(14.0, 11.0, 0.75), (52.0, 14.0, 1.0), (93.0, 10.0, 0.65)]


def periodic_noise(n, rate, cutoff, seed):
    """Nhiễu trơn tuần hoàn (n mẫu ở tần số lấy mẫu `rate`), chuẩn hoá std=1."""
    rng = np.random.default_rng(seed)
    F = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n, 1.0 / rate)
    F *= np.exp(-(f / cutoff) ** 2)
    F[0] = 0
    x = np.fft.irfft(F, n)
    return (x / x.std()).astype(np.float64)


# Bảng tra 100 Hz cho các tín hiệu điều khiển chậm
CTRL_RATE = 100
_N = int(LOOP * CTRL_RATE)
_flick = periodic_noise(_N, CTRL_RATE, 2.5, 11)       # chập chờn của lửa
_flick_l = periodic_noise(_N, CTRL_RATE, 3.5, 12)
_flick_r = periodic_noise(_N, CTRL_RATE, 3.5, 13)
_gustwob = periodic_noise(_N, CTRL_RATE, 0.6, 14)     # gió giật trong cơn gió


def _sample(tab, t):
    t = np.mod(np.asarray(t, dtype=np.float64), LOOP) * CTRL_RATE
    i0 = np.floor(t).astype(int) % _N
    fr = t - np.floor(t)
    return tab[i0] * (1 - fr) + tab[(i0 + 1) % _N] * fr


def gust(t):
    """Đường bao cơn gió 0..~1, tuần hoàn."""
    t = np.mod(np.asarray(t, dtype=np.float64), LOOP)
    g = np.zeros_like(t)
    for t0, d, a in GUSTS:
        s = np.mod(t - t0, LOOP) / d
        m = s < 1
        ss = np.where(m, s, 0)
        # lên nhanh, tắt chậm
        g += np.where(m, a * np.sin(np.pi * ss ** 0.6) ** 2, 0)
    g *= 0.8 + 0.2 * np.tanh(_sample(_gustwob, t))
    return g + 0.04  # gió hiu hiu nền


def flicker(t, which=0):
    tab = (_flick, _flick_l, _flick_r)[which]
    return _sample(tab, t)
