"""
probe_camera.py
---------------
Đo FPS THỰC TẾ của webcam theo từng tổ hợp (backend, FOURCC, độ phân giải, thứ tự set).
Mục đích: tìm cấu hình để dán vào pan_tilt_tracking.py.

Chạy:  python probe_camera.py      (mất khoảng 1-3 phút)
"""

import os
import sys
import time

# Với MSMF trên Windows: tắt hardware transforms để mở camera nhanh và ổn định hơn
# (phải đặt TRƯỚC khi import cv2)
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402

# ----------------------------- CẤU HÌNH -----------------------------
CAMERA_INDEX = 1
MEASURE_SEC = 1.5      # thời gian đo cho mỗi cấu hình
WARMUP_FRAMES = 10     # bỏ qua các frame đầu
MODES = [              # (width, height, fps yêu cầu)
    (1920, 1080, 30),
    (1920, 1080, 30),
    (1280, 720, 30),
    (1280, 720, 30),
    (640, 480, 30),
    (640, 480, 30),
]
# ---------------------------------------------------------------------


def get_backends():
    if sys.platform.startswith("win"):
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
    if sys.platform.startswith("linux"):
        return [("V4L2", cv2.CAP_V4L2)]
    return [("ANY", cv2.CAP_ANY)]


def fourcc_to_str(v: int) -> str:
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


def probe(backend, fourcc, w, h, fps, fourcc_first):
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not cap.isOpened():
        return None

    fcc = cv2.VideoWriter_fourcc(*fourcc)
    if fourcc_first:
        cap.set(cv2.CAP_PROP_FOURCC, fcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not fourcc_first:
        cap.set(cv2.CAP_PROP_FOURCC, fcc)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    afcc = fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))

    for _ in range(WARMUP_FRAMES):
        cap.read()

    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < MEASURE_SEC:
        ok, _ = cap.read()
        if ok:
            n += 1
    measured = n / (time.perf_counter() - t0)
    cap.release()
    time.sleep(0.2)
    return afcc, aw, ah, measured


def main():
    rows = []
    print(f"{'backend':7} {'req_fourcc':10} {'order':13} {'req_mode':14} -> "
          f"{'applied':22} {'REAL fps':>8}")
    print("-" * 82)

    for bname, backend in get_backends():
        for fourcc in ("MJPG", "YUY2"):
            orders = (True, False) if fourcc == "MJPG" else (True,)
            for fourcc_first in orders:
                for (w, h, fps) in MODES:
                    res = probe(backend, fourcc, w, h, fps, fourcc_first)
                    order = "fourcc-first" if fourcc_first else "fourcc-last"
                    req = f"{w}x{h}@{fps}"
                    if res is None:
                        print(f"{bname:7} {fourcc:10} {order:13} {req:14} -> không mở được camera")
                        continue
                    afcc, aw, ah, measured = res
                    print(f"{bname:7} {fourcc:10} {order:13} {req:14} -> "
                          f"{afcc} {aw}x{ah:<12} {measured:8.1f}")
                    rows.append((measured, bname, fourcc, order, w, h, fps, afcc, aw, ah))

    print("\n=== TOP cấu hình (độ phân giải áp dụng == yêu cầu, sắp theo FPS thực) ===")
    good = [r for r in rows if (r[8], r[9]) == (r[4], r[5])]
    for r in sorted(good, reverse=True)[:5]:
        measured, bname, fourcc, order, w, h, fps, afcc, aw, ah = r
        print(f"{measured:6.1f} fps | backend={bname} | yêu cầu {fourcc} {w}x{h}@{fps} "
              f"({order}) | driver báo FOURCC={afcc}")


if __name__ == "__main__":
    main()