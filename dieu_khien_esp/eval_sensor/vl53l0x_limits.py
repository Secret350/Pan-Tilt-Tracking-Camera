import csv
import math
import statistics
import time
from datetime import datetime

import serial

# ===== CONFIG =====
PORT = "COM9"                 # đổi theo máy
BAUD = 115200
TEST = "angle"             # "distance" | "angle" | "both"

BUDGET_US = 33000             # timing budget gửi cho firmware
INVALID_MM = 8190             # >= giá trị này (hoặc -1) coi là mẫu không hợp lệ
OUT_PREFIX = "vl53l0x"

# --- test khoảng cách ---
DIST_MODES = [0, 1]          # 0 = mặc định, 1 = long range
N_DIST = 50                  # số mẫu mỗi điểm
ERR_ABS_MM = 10              # sai số cho phép: lấy mức lớn hơn giữa ERR_ABS_MM
ERR_REL = 0.03               # và ERR_REL * khoảng cách thật
VALID_MIN_PCT = 90           # tỷ lệ mẫu hợp lệ tối thiểu để điểm được coi là đạt

# --- test góc (servo pan) ---
ANGLE_CENTER = 90            # góc servo khi cảm biến hướng thẳng vào vật
ANGLE_HALF_SWEEP = 40        # quét từ CENTER-HALF đến CENTER+HALF
ANGLE_STEP = 1               # bước quét (độ)
N_ANGLE = 10                 # số mẫu mỗi góc
TARGET_DIST_MM = 300         # khoảng cách từ cảm biến tới vật hẹp
TARGET_WIDTH_MM = 20         # bề rộng vật hẹp (càng nhỏ càng tốt)
DETECT_TOL_MM = 20           # |median - TARGET_DIST_MM| <= ngưỡng này => "thấy vật"
DETECT_MIN_PCT = 50          # và ít nhất ngần này % mẫu hợp lệ
ANGLE_MODE = 0               # chế độ đo dùng cho test góc
# ==================


class Tester:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=5)
        time.sleep(2.0)                      # ESP32 reset khi mở cổng
        self.ser.reset_input_buffer()
        for _ in range(3):
            self.ser.write(b"PING\n")
            if self.ser.readline().decode(errors="ignore").strip() == "PONG":
                return
        raise RuntimeError("Không nhận được PONG: kiểm tra cổng COM và firmware")

    def _cmd(self, s):
        self.ser.write((s + "\n").encode())
        r = self.ser.readline().decode(errors="ignore").strip()
        if r != "OK":
            raise RuntimeError(f"Lệnh '{s}' lỗi, phản hồi: '{r}'")

    def set_mode(self, mode):
        self._cmd(f"M,{int(mode)}")

    def set_budget(self, us):
        self._cmd(f"B,{int(us)}")

    def set_angle(self, deg):
        self._cmd(f"A,{int(deg)}")

    def read(self, n):
        self.ser.timeout = 5 + n * BUDGET_US / 1e6 * 1.5
        self.ser.write(f"R,{n}\n".encode())
        vals = []
        while True:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line == "END":
                return vals
            if line.startswith("ERR") or line == "":
                raise RuntimeError(f"Firmware báo lỗi/timeout: '{line}'")
            try:
                vals.append(int(line))
            except ValueError:
                pass                          # bỏ dòng rác

    def close(self):
        self.ser.close()


def stats(samples):
    valid = [v for v in samples if 0 <= v < INVALID_MM]
    out = {"valid_pct": 100.0 * len(valid) / len(samples) if samples else 0.0,
           "mean": None, "median": None, "std": None}
    if valid:
        out["mean"] = statistics.mean(valid)
        out["median"] = statistics.median(valid)
        out["std"] = statistics.stdev(valid) if len(valid) > 1 else 0.0
    return out


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------- khoảng cách
def dist_ok(true_mm, st):
    if st["median"] is None or st["valid_pct"] < VALID_MIN_PCT:
        return False
    return abs(st["median"] - true_mm) <= max(ERR_ABS_MM, ERR_REL * true_mm)


def summarize_distance(rows):
    print("\n===== TÓM TẮT GIỚI HẠN KHOẢNG CÁCH =====")
    print(f"Tiêu chí đạt: sai số <= max({ERR_ABS_MM} mm, {ERR_REL*100:.0f}% khoảng cách) "
          f"và >= {VALID_MIN_PCT}% mẫu hợp lệ")
    groups = {}
    for r in rows:
        groups.setdefault((r["label"], r["mode"]), []).append(r)
    for (label, mode), g in sorted(groups.items()):
        g.sort(key=lambda r: r["true_mm"])
        name = "long range" if mode else "mặc định"
        first = next((i for i, r in enumerate(g) if r["ok"]), None)
        if first is None:
            print(f"[{label} | {name}] không có điểm nào đạt")
            continue
        last = first
        while last + 1 < len(g) and g[last + 1]["ok"]:
            last += 1
        msg = (f"[{label} | {name}] min = {g[first]['true_mm']:.0f} mm, "
               f"max = {g[last]['true_mm']:.0f} mm")
        if last + 1 < len(g):
            msg += f" (điểm hỏng đầu tiên phía xa: {g[last+1]['true_mm']:.0f} mm)"
        if first > 0:
            msg += f" (điểm hỏng phía gần: {g[first-1]['true_mm']:.0f} mm)"
        print(msg)


def run_distance(t):
    label = input("Nhãn vật mục tiêu (vd white, gray): ").strip() or "target"
    print("Gốc đo = mặt kính của chip. Vật phẳng, vuông góc với trục đo.")
    rows = []
    while True:
        s = input("\nKhoảng cách thật (mm), Enter để kết thúc: ").strip()
        if not s:
            break
        try:
            true_mm = float(s)
        except ValueError:
            print("Hãy nhập một số.")
            continue
        for mode in DIST_MODES:
            t.set_mode(mode)
            st = stats(t.read(N_DIST))
            ok = dist_ok(true_mm, st)
            err = None if st["median"] is None else st["median"] - true_mm
            rows.append({"label": label, "mode": mode, "true_mm": true_mm,
                         "mean_mm": st["mean"], "median_mm": st["median"],
                         "std_mm": st["std"], "err_mm": err,
                         "valid_pct": st["valid_pct"], "ok": ok})
            print(f"  mode={mode} median={fmt(st['median'])} std={fmt(st['std'])} "
                  f"err={fmt(err)} valid={st['valid_pct']:.0f}% -> {'OK' if ok else 'FAIL'}")
    if not rows:
        return
    path = f"{OUT_PREFIX}_distance_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nĐã lưu {path}")
    summarize_distance(rows)
    plot_distance(rows, path.replace(".csv", ".png"))


# --------------------------------------------------------------------- góc
def run_angle(t):
    t.set_mode(ANGLE_MODE)
    angles = list(range(ANGLE_CENTER - ANGLE_HALF_SWEEP,
                        ANGLE_CENTER + ANGLE_HALF_SWEEP + 1, ANGLE_STEP))
    print(f"\nQuét {angles[0]}..{angles[-1]} độ, bước {ANGLE_STEP}, "
          f"vật rộng {TARGET_WIDTH_MM} mm cách {TARGET_DIST_MM} mm")
    t.set_angle(angles[0])
    time.sleep(1.0)
    rows = []
    for a in angles:
        t.set_angle(a)
        st = stats(t.read(N_ANGLE))
        det = (st["median"] is not None and st["valid_pct"] >= DETECT_MIN_PCT
               and abs(st["median"] - TARGET_DIST_MM) <= DETECT_TOL_MM)
        rows.append({"angle": a, "median_mm": st["median"], "std_mm": st["std"],
                     "valid_pct": st["valid_pct"], "detected": det})
        print(f"  {a:4d} deg  median={fmt(st['median']):>7}  valid={st['valid_pct']:3.0f}%  "
              f"{'#' if det else '.'}")
    t.set_angle(ANGLE_CENTER)

    path = f"{OUT_PREFIX}_angle_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nĐã lưu {path}")
    summarize_angle(rows)
    plot_angle(rows, path.replace(".csv", ".png"))


def summarize_angle(rows):
    # đoạn liên tiếp dài nhất các góc có "thấy vật"
    best, cur = [], []
    for r in rows:
        if r["detected"]:
            cur.append(r["angle"])
            if len(cur) > len(best):
                best = cur[:]
        else:
            cur = []
    print("\n===== TÓM TẮT GÓC NHÌN =====")
    if not best:
        print("Không thấy vật ở góc nào: kiểm tra TARGET_DIST_MM, DETECT_TOL_MM, hướng cảm biến")
        return
    a0, a1 = best[0], best[-1]
    window = (a1 - a0) + ANGLE_STEP                      # cửa sổ góc thấy vật
    beta = math.degrees(math.atan(TARGET_WIDTH_MM / (2 * TARGET_DIST_MM)))
    fov = window - 2 * beta                              # trừ phần do bề rộng vật
    centre = (a0 + a1) / 2
    beam = 2 * TARGET_DIST_MM * math.tan(math.radians(fov / 2)) if fov > 0 else float("nan")
    print(f"Thấy vật từ {a0} đến {a1} độ -> cửa sổ = {window:.1f} độ (sai số ~ ±{ANGLE_STEP})")
    print(f"Bề rộng góc của vật = {2*beta:.1f} độ")
    print(f"FoV ≈ {fov:.1f} độ (cửa sổ - bề rộng góc của vật)")
    print(f"Tâm cửa sổ = {centre:.1f} độ (lệch {centre - ANGLE_CENTER:+.1f} so với CENTER)")
    print(f"Đường kính vùng đo tại {TARGET_DIST_MM} mm ≈ {beam:.0f} mm")


# --------------------------------------------------------------------- vẽ
def plot_distance(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = {}
    for r in rows:
        if r["err_mm"] is not None:
            groups.setdefault((r["label"], r["mode"]), []).append(r)
    for (label, mode), g in sorted(groups.items()):
        g.sort(key=lambda r: r["true_mm"])
        ax.plot([r["true_mm"] for r in g], [r["err_mm"] for r in g], "o-",
                label=f"{label} | {'long range' if mode else 'mặc định'}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Khoảng cách thật (mm)")
    ax.set_ylabel("Sai số median (mm)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Đã lưu {path}")


def plot_angle(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    xs = [r["angle"] for r in rows if r["median_mm"] is not None]
    ys = [r["median_mm"] for r in rows if r["median_mm"] is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-", ms=3)
    ax.axhline(TARGET_DIST_MM, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Góc servo (độ)")
    ax.set_ylabel("Median (mm)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Đã lưu {path}")


def main():
    t = Tester(PORT, BAUD)
    try:
        t.set_budget(BUDGET_US)
        if TEST in ("distance", "both"):
            run_distance(t)
        if TEST in ("angle", "both"):
            run_angle(t)
    finally:
        t.close()


if __name__ == "__main__":
    main()
