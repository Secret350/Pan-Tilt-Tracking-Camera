"""
infer_webcam.py - CORE của chương trình Pan-Tilt Tracking.

File này làm toàn bộ phần "chạy chương trình": camera, model YOLO + tracker, chọn/khóa mục tiêu,
gửi lệnh sang ESP32, nhận khoảng cách từ VL53L0X và zoom số theo khoảng cách.
File gui.py chỉ vẽ giao diện và gọi vào TrackingEngine ở đây.

CÁCH CHẠY
  python gui.py            # có giao diện (khuyến nghị)
  python infer_webcam.py   # không giao diện (chạy nền, in log ra console, Ctrl+C để thoát)

GIAO THỨC SERIAL
  PC -> ESP32 : E,<err_x>,<err_y>,<has_target>\n   |  P,<tên>,<giá trị>\n  |  H,<pan>,<tilt>\n
  ESP32 -> PC : D,<khoảng_cách_mm>,<valid>\n        |  T,ERR\n (không tìm thấy VL53L0X)

YÊU CẦU OPENCV
  OpenCV >= 5 (xem CV_MIN_MAJOR). Cài (opencv 5 cần numpy >= 2):
      pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
      pip install opencv-python==5.0.0.93 opencv-contrib-python==5.0.0.93
  Lý do cài cả hai CÙNG phiên bản: ultralytics đòi opencv-python, mediapipe đòi opencv-contrib-python.
  Kiểm tra bản thực sự được nạp: log "[CV] OpenCV ..." khi chương trình khởi động.

LOGIC ZOOM THEO KHOẢNG CÁCH
  - Cảm biến gắn cùng khung camera nên đo vật ở TÂM khung hình. Vì vậy chỉ tin số đo khi mục tiêu
    đang gần tâm (|err_x|, |err_y| < DIST_GATE_ERR_PX).
  - Khoảng cách > ZOOM_DIST_THRESHOLD_MM  -> zoom x ZOOM_FACTOR (cắt giữa khung + resize bằng OpenCV,
    ảnh này được đưa vào YOLO). Sai số gửi sang ESP32 luôn được quy về pixel của khung GỐC
    (chia cho hệ số zoom) nên các tham số PID/hfov giữ nguyên ý nghĩa.
  - Box mục tiêu biến mất quá ZOOM_LOST_GRACE_SEC -> trả về x1.0 như ban đầu.
"""
import contextlib
import csv
import json
import os
import sys
import threading
import time
import collections
import statistics

from eval.eval_sys.prob_cam import CAMERA_INDEX

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import serial

try:  # tùy chọn: thiếu mediapipe thì chương trình vẫn chạy, chỉ không có mask khuôn mặt
    import mediapipe as mp
except ImportError:
    mp = None

# --- OpenCV ---
CV_MIN_MAJOR = 5  # thấp hơn vẫn chạy được nhưng sẽ in cảnh báo lúc khởi động

# ---Distance (VL53L0X sensor)---
DIST_MEDIAN_N = 30  # lọc median 5 mẫu gần nhất (khử nhiễu xung)
DIST_MAX_AGE_S = 0.4  # thời gian đợi trước khi coi như mất dữ liệu từ sensor
DIST_GATE_ERR_PX = 80  # chỉ tin khoảng cách khi |errx|,|erry| nhỏ hơn mức này

# --- Đo khoảng cách bằng thị giác (OpenCV), hiển thị song song với VL53L0X để so sánh ---
#   "pnp" : cv2.solvePnP (SQPNP + LM) trên 14 landmark MediaPipe + mô hình mặt 3D chuẩn -> độ sâu Z của đỉnh mũi.
#           Cần FACE_MASK_ENABLE (lấy landmark từ MediaPipe), chạy theo nhịp FACE_MASK_EVERY_N.
#   "bbox": tam giác đồng dạng Z = f * bề_ngang_đầu_thật / bề_ngang_box. Chạy mọi frame, không cần MediaPipe.
DIST_VISION_ENABLE = True
CAM_FOCAL_PX = None  # tiêu cự (px) ở độ phân giải CAP_WIDTH, khung chưa zoom. None: suy từ thanh trượt hfov.
#                      CHỈ đúng nếu hfov = FOV ngang THẬT của webcam (sai tiêu cự là nguồn lỗi lớn nhất). Đã hiệu chuẩn bằng
#                      cv2.calibrateCamera thì nhập fx ở đây.
HEAD_WIDTH_CM = 15.3  # bề ngang đầu cho "bbox" (lấy từ mô hình mặt chuẩn: khoảng cách hai má). Box "head" thật có thể rộng hơn (tóc, tai) -> k sửa
HEAD_DEPTH_CM = 9.9  # "bbox" đo tới mặt phẳng hai má, nằm sau đỉnh mũi chừng này (cũng từ mô hình chuẩn); cảm biến thì đo tới mũi nên phải trừ đi
DIST_PNP_SCALE = 1.0  # hệ số nhân sửa kết quả "pnp". Sửa bằng: giá trị hiện tại x k hiển thị ở GUI (k ≈ 1.00 là khớp)
DIST_BBOX_SCALE = 1.0  # tương tự cho "bbox"
DIST_VISION_MEDIAN_N = 5  # median các ước lượng gần nhất cho đỡ nhảy
DIST_VISION_MAX_AGE_S = 0.5  # quá lâu không có ước lượng mới thì bỏ số cũ
DIST_VISION_MIN_CM = 10.0  # ngoài khoảng [MIN, MAX] coi là rác
DIST_VISION_MAX_CM = 400.0
DIST_PNP_MAX_REPROJ = 0.08  # bỏ kết quả PnP nếu sai số tái chiếu (RMS px) / bề ngang landmark lớn hơn mức này (chưa tune trên dữ liệu thật)
DIST_CMP_WINDOW = 300  # số cặp (sensor, thị giác) gần nhất dùng tính MAE / k
DIST_CMP_MAX_SPREAD_MM = 40  # chỉ ghi cặp so sánh khi các mẫu sensor gần nhất dao động <= mức này (median của sensor bị trễ khi người đang di chuyển)

# --- Camera ---
CAMERA_INDEX=0
# CAMERA_INDEX = 1
CAP_BACKEND = "MSMF"  # đã probe: MSMF cho ~28.9 fps ở 1280x720 (backend tốt nhất trên thiết bị)
CAP_FOURCC_FIRST = False  # MSMF bỏ qua FOURCC nên FALSE/TRUE đều cho 28.9 fps;
# giữ False để nếu phải rơi về DSHOW thì vẫn đặt MJPG đúng thứ tự
CAP_WIDTH = 1280  # đã probe: driver áp dụng 1280x720
CAP_HEIGHT = 720
CAP_FPS = 30  # MSMF bỏ qua giá trị này; thực tế đo được ~28.9 fps
CAP_FOURCC = "MJPG"
CAP_BUFFERSIZE = 1
CAP_EXPOSURE = None

# Danh sách mode thử lần lượt nếu mode đầu không cho frame: (w, h, fps, fourcc)
CAP_FALLBACK_MODES = [
    (CAP_WIDTH, CAP_HEIGHT, CAP_FPS, CAP_FOURCC),
    (640, 480, 30, CAP_FOURCC),
    (640, 480, 30, None),  # None = để driver tự chọn
]

# --- Model ---
MODEL_PATH = "../models/HEAD_RB/runs/head_det/weights/best.pt"
IMG_SIZE = 640
CONF_THRES = 0.65
TARGET_CLASSES = [0]
DEVICE = 0
WARMUP_ITERS = 5


# --- Hiển thị / log ---
LOG_INTERVAL_SEC = 1.0  # chu kỳ in FPS/sai số ra console (0 = không in)
LOG_CSV_ENABLE = True  # True: ghi t, err_x, err_y, has_target, infer_ms ra CSV để vẽ đồ thị khi tune PID
LOG_CSV_PATH = "../logs/tracking_log.csv"
TUNING_FILE = "../json/pid_params.json"  # tự lưu khi bấm SAVE_KEY hoặc khi thoát, và tự nạp lại lần chạy sau

PARAM_RESEND_SEC = 1.0  # gửi lại toàn bộ tham số định kỳ (phòng ESP32 reset hoặc rớt byte)
# (tên khớp bảng params[] trong .ino, mặc định, min, max, bước, đơn vị)
# Khi GUI bật, giá trị ở đây (hoặc trong pid_params.json) là NGUỒN CHÍNH và ghi đè mặc định của ESP32.
TUNE_PARAMS = [
    ("hfov", 65.0, 20.0, 120.0, 0.5, "deg"),  # HFOV webcam
    ("omega_pan", 4.2, 0.0, 12.0, 0.1, "rad/s"),
    ("omega_tilt", 4.2, 0.0, 12.0, 0.1, "rad/s"),
    ("kd_pan", 0.0, 0.0, 0.3, 0.005, ""),
    ("kd_tilt", 0.0, 0.0, 0.3, 0.005, ""),
    ("deadzone", 20.0, 0.0, 50.0, 1.0, "px"),
    ("max_vel", 56.0, 1.0, 120.0, 1.0, "deg/s"),
    ("acq_vel", 11.0, 1.0, 60.0, 1.0, "deg/s"),
    ("acq_ramp", 650.0, 0.0, 3000.0, 50.0, "ms"),
    ("max_accel", 600.0, 50.0, 2000.0, 50.0, "deg/s2"),
    ("err_alpha", 0.25, 0.05, 1.0, 0.05, ""),
    ("d_alpha", 0.2, 0.05, 1.0, 0.05, ""),
    ("hold_ms", 100.0, 0.0, 300.0, 10.0, "ms"),  # giữ vận tốc khi mất detection ngắn
]

# --- Serial -> ESP32 ---
SERIAL_ENABLE = True  # đặt False khi chỉ test camera/model mà chưa cắm ESP32
SERIAL_PORT = "COM9"  # thay bằng cổng ESP32 trong Device Manager (Ports - COM & LPT)
SERIAL_BAUD = 115200
SERIAL_BOOT_WAIT_SEC = 2.0  # ESP32 tự reset khi mở cổng COM, cần chờ khởi động

# --- Zoom số theo khoảng cách VL53L0X (MỚI) ---
ZOOM_ENABLE = True
ZOOM_FACTOR = 1.5  # hệ số zoom (cắt giữa khung rồi resize lại bằng OpenCV)
ZOOM_DIST_THRESHOLD_MM = 800  # khoảng cách > ngưỡng này thì zoom. Phải nhỏ hơn TOF_MAX_VALID_MM trong .ino (1200)
ZOOM_RELEASE_HYST_MM = 100  # đang zoom: chỉ thôi zoom khi đo được khoảng cách < ngưỡng - 100 mm (chống nháy)
ZOOM_FAR_AS_EXCEED = True  # cảm biến báo "ngoài tầm đo" (xa hơn ~1.2 m) cũng tính là vượt ngưỡng
ZOOM_LOST_GRACE_SEC = 0.3  # box biến mất bao lâu thì trả về x1.0 (0 = trả về ngay lập tức)

# --- Mask khuôn mặt bám theo đường viền (MediaPipe landmark -> convex hull) ---
FACE_MASK_ENABLE = True
FACE_MASK_DRAW = True  # True: vẽ mask thẳng lên frame trong snapshot (gui.py không cần sửa)
FACE_MASK_ONLY_TARGET = True  # True: chỉ làm mask cho mục tiêu đang bám (nhanh); False: mọi track (tối đa MAX_DET_GUI)
FACE_MASK_EVERY_N = 2  # chạy landmark mỗi N frame; frame giữa dùng lại landmark, chiếu lại lên bbox hiện tại
FACE_MASK_HOLD_FRAMES = 10  # landmark thất bại (mặt quay đi...) vẫn giữ mask cũ tối đa chừng này frame
FACE_MASK_SMOOTH = 0.5  # EMA cho landmark, 1.0 = không lọc, nhỏ hơn = mượt hơn nhưng trễ hơn
FACE_MASK_PAD = 0.10  # nới bbox khi cắt crop đưa vào MediaPipe để landmark không bị cắt ở rìa
FACE_MASK_MIN_PX = 40  # bbox nhỏ hơn mức này (px) thì bỏ qua
FACE_MASK_COLOR = (0, 255, 0)  # BGR, mục tiêu đang bám
FACE_MASK_COLOR_OTHER = (255, 200, 0)  # BGR, các track còn lại
FACE_MASK_ALPHA = 0.35
FACE_MASK_OUTLINE_PX = 2
# Chỉ cần khi mediapipe KHÔNG còn mp.solutions (bản mới): tải face_landmarker.task từ trang MediaPipe Face Landmarker
FACE_LANDMARKER_MODEL = "../models/face_landmarker.task"
FACE_LANDMARKER_AUTODOWNLOAD = True  # file thiếu thì tự tải (~4 MB) về FACE_LANDMARKER_MODEL ở lần chạy đầu
FACE_LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                       "face_landmarker/float16/1/face_landmarker.task")

# --- Chọn / khóa mục tiêu (chuyển từ gui.py sang core, giá trị giữ nguyên) ---
TRACKER_CFG = "bytetrack.yaml"  # hoặc "botsort.yaml"
MAX_DET_GUI = 3  # số đối tượng tối đa mỗi khung
NMS_IOU_THRESHOLD = 0.7  # IoU > ngưỡng này thì loại box có confidence thấp hơn

HOME_PAN = 90.0
HOME_TILT = 130.0
HOME_DELAY_SEC = 1.0  # mất hết mục tiêu bao lâu thì mới về Home (chống nháy 1-2 frame)
HOME_RESEND_SEC = 0.5  # gửi lại lệnh H định kỳ trong lúc ở Home (phòng rớt byte)
LOCK_LOST_GRACE_SEC = 0.5  # ID bị khóa mất bao lâu thì nhả khóa (0 = nhả ngay)

# Ghi chú: SHOW_WINDOW, WINDOW_NAME, QUIT_KEY, TUNING_GUI, TUNING_WINDOW, SAVE_KEY, TARGET_SELECT, MAX_DET, USE_HALF
# là hằng số của bản cũ (cửa sổ OpenCV + trackbar). Giá trị giữ nguyên nhưng hiện không còn được dùng vì
# giao diện đã chuyển sang gui.py và việc chọn mục tiêu do tracker ID đảm nhiệm.

def log_cv_version():
    """In phiên bản + vị trí cv2 ĐANG ĐƯỢC NẠP. Vì ultralytics/mediapipe kéo thêm gói opencv riêng,
    rất dễ tưởng đã lên 5 trong khi Python vẫn nạp bản cũ."""
    try:
        major = int(cv2.__version__.split(".")[0])
    except ValueError:
        major = 0
    print(f"[CV] OpenCV {cv2.__version__}  ({os.path.dirname(os.path.abspath(cv2.__file__))})")
    if major < CV_MIN_MAJOR:
        print(f"[CV] CẢNH BÁO: cần OpenCV >= {CV_MIN_MAJOR}, đang nạp {cv2.__version__}. "
              f"Gỡ hết gói opencv-* rồi cài lại (xem đầu file).")


_ser = None
_ser_lock = threading.Lock()  # ghi serial từ nhiều luồng (engine + GUI)


def serial_open():
    global _ser
    if not SERIAL_ENABLE:
        return
    _ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0, write_timeout=0.05)
    time.sleep(SERIAL_BOOT_WAIT_SEC)
    print(f"[ESP32] đã mở {SERIAL_PORT} @ {SERIAL_BAUD}")


def serial_close():
    global _ser
    if _ser is not None:
        _ser.close()
        _ser = None


def _ser_write(text):
    ser = _ser
    if ser is None:
        return
    with _ser_lock:
        try:
            ser.write(text.encode("ascii"))
        except (serial.SerialException, OSError):
            pass  # rớt gói này (kể cả write timeout), gói sau thay thế


def send_error(ex, ey, has_target):
    _ser_write(f"E,{ex:.1f},{ey:.1f},{1 if has_target else 0}\n")


def send_param(name, value):
    _ser_write(f"P,{name},{value:.4f}\n")


def send_home():
    _ser_write(f"H,{HOME_PAN:.1f},{HOME_TILT:.1f}\n")


# ======================= THAM SỐ TUNING (đọc/ghi pid_params.json) =======================
def load_tuning():
    """-> {tên: giá trị} lấy từ TUNING_FILE (nếu có), kẹp trong [min, max] và làm tròn theo bước."""
    saved = {}
    try:
        with open(TUNING_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        pass
    if not isinstance(saved, dict):
        saved = {}
    vals = {}
    for name, default, lo, hi, step, _u in TUNE_PARAMS:
        try:
            v = float(saved.get(name, default))
        except (TypeError, ValueError):
            v = default
        v = min(max(v, lo), hi)
        vals[name] = round(lo + round((v - lo) / step) * step, 4)
    return vals


def save_tuning(vals):
    with open(TUNING_FILE, "w", encoding="utf-8") as f:
        json.dump(vals, f, indent=2)


# ======================= CẢM BIẾN KHOẢNG CÁCH =======================
class DistanceReader:
    """Luồng đọc các dòng D,<mm>,<valid> do ESP32 gửi về.

    Cổng serial đang mở với timeout=0 (không chặn) nên KHÔNG dùng readline() được: nó trả về ngay
    cả khi mới có nửa dòng và luồng sẽ quay vòng chiếm CPU. Ở đây tự ghép byte rồi tách theo '\n'.
    """

    def __init__(self, ser):
        self.ser = ser
        self._buf = collections.deque(maxlen=DIST_MEDIAN_N)
        self._last_ts = 0.0
        self._last_valid = False
        self._sensor_err = False
        self._lock = threading.Lock()
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        rx = bytearray()
        while self._run:
            try:
                n = self.ser.in_waiting
                data = self.ser.read(n) if n else b""
            except Exception:
                time.sleep(0.05)  # cổng lỗi / đã đóng
                continue
            if not data:
                time.sleep(0.002)
                continue
            rx.extend(data)
            while True:
                i = rx.find(b"\n")
                if i < 0:
                    break
                line = bytes(rx[:i]).decode("ascii", errors="ignore").strip()
                del rx[:i + 1]
                self._handle(line)
            if len(rx) > 1024:  # rác không có '\n' (vd log boot của ESP32)
                rx.clear()

    def _handle(self, line):
        if line == "T,ERR":
            with self._lock:
                self._sensor_err = True
            return
        if not line.startswith("D,"):
            return  # bỏ qua dòng khác
        parts = line.split(",")
        if len(parts) != 3:
            return
        try:
            mm, valid = int(parts[1]), int(parts[2])
        except ValueError:
            return
        with self._lock:
            self._sensor_err = False
            self._last_valid = bool(valid)
            if valid:
                self._buf.append(mm)
            else:
                self._buf.clear()  # mẫu xấu: xóa lịch sử để khỏi giữ số cũ
            self._last_ts = time.time()

    def read(self):
        """-> (state, mm). state: 'OK' (mm = median đã lọc) | 'OUT_OF_RANGE' | 'NO_DATA' | 'SENSOR_ERR'."""
        with self._lock:
            if self._sensor_err:
                return "SENSOR_ERR", None
            if self._last_ts == 0.0 or time.time() - self._last_ts > DIST_MAX_AGE_S:
                return "NO_DATA", None
            if self._last_valid and self._buf:
                return "OK", int(statistics.median(self._buf))
            return "OUT_OF_RANGE", None

    def get_mm(self):
        """Trả về khoảng cách (mm) đã lọc median, hoặc None nếu không tin được."""
        state, mm = self.read()
        return mm if state == "OK" else None

    def spread_mm(self):
        """Độ dao động (max - min, mm) của các mẫu hợp lệ gần nhất; None nếu chưa đủ mẫu."""
        with self._lock:
            if len(self._buf) < max(3, DIST_MEDIAN_N // 2):
                return None
            return max(self._buf) - min(self._buf)

    def stop(self):
        self._run = False


# ======================= ĐO KHOẢNG CÁCH BẰNG THỊ GIÁC =======================
# 14 landmark MediaPipe và toạ độ 3D (cm) của chúng trong canonical_face_model.obj của MediaPipe (Apache-2.0).
# Quy ước gốc của file: x sang phải, y lên trên, z về phía camera. Hàng đầu (landmark 1) là đỉnh mũi.
_PNP_LM_IDX = [1, 168, 33, 133, 362, 263, 129, 358, 61, 291, 0, 17, 152, 10]
_PNP_MODEL_MP = np.array([
    [0.000, -1.127, 7.476],  # 1   đỉnh mũi
    [0.000, 3.271, 5.236],  # 168 sống mũi giữa hai mắt
    [-4.446, 2.664, 3.173],  # 33  khóe ngoài mắt
    [-1.856, 2.585, 3.758],  # 133 khóe trong mắt
    [1.856, 2.585, 3.758],  # 362 khóe trong mắt
    [4.446, 2.664, 3.173],  # 263 khóe ngoài mắt
    [-1.786, -0.978, 4.850],  # 129 cánh mũi
    [1.786, -0.978, 4.850],  # 358 cánh mũi
    [-2.456, -4.343, 4.284],  # 61  khóe miệng
    [2.456, -4.343, 4.284],  # 291 khóe miệng
    [0.000, -3.406, 5.980],  # 0   môi trên
    [0.000, -5.365, 5.535],  # 17  môi dưới
    [0.000, -9.403, 4.264],  # 152 cằm
    [0.000, 8.262, 4.482],  # 10  trán
], dtype=np.float64)
# Đổi sang quy ước OpenCV (x phải, y xuống, z ra xa camera) và dời gốc về đỉnh mũi: khi đó tvec của
# solvePnP là vị trí đỉnh mũi trong hệ camera, tvec[2] = độ sâu (cm) của mũi - điểm gần cảm biến nhất.
_PNP_MODEL = _PNP_MODEL_MP * np.array([1.0, -1.0, -1.0])
_PNP_MODEL = _PNP_MODEL - _PNP_MODEL[0]


class VisionDistance:
    """Ước lượng khoảng cách (cm) từ ảnh bằng OpenCV và so với cảm biến.

    Hai phương pháp chạy song song (xem khối cấu hình DIST_VISION_*): 'pnp' và 'bbox'.
    Cặp (sensor, pnp, bbox) chỉ được ghi khi đủ tin cậy; thống kê trên cửa sổ trượt DIST_CMP_WINDOW.
    Gọi từ luồng engine, ngoại trừ reset_stats() có thể gọi từ GUI (đã khóa).
    """

    def __init__(self):
        self._v = {"pnp": collections.deque(maxlen=DIST_VISION_MEDIAN_N),
                   "bbox": collections.deque(maxlen=DIST_VISION_MEDIAN_N)}
        self._t = {"pnp": 0.0, "bbox": 0.0}
        self._pairs = collections.deque(maxlen=DIST_CMP_WINDOW)  # (sensor, pnp, bbox) cm; NaN nếu thiếu
        self._mu = threading.Lock()

    def _push(self, key, cm, now):
        if not (DIST_VISION_MIN_CM <= cm <= DIST_VISION_MAX_CM):
            return
        if now - self._t[key] > DIST_VISION_MAX_AGE_S:
            self._v[key].clear()  # mục tiêu vừa xuất hiện lại: bỏ số cũ để median không bị trộn
        self._v[key].append(cm)
        self._t[key] = now

    def update_bbox(self, box, fx, now):
        w = box[2] - box[0]
        if w >= 4:
            self._push("bbox", (fx * HEAD_WIDTH_CM / w - HEAD_DEPTH_CM) * DIST_BBOX_SCALE, now)

    def update_pnp(self, lm, fx, fy, cx, cy, now):
        """lm: (468,2) landmark theo pixel của khung đang đưa vào model (đã tính cả zoom số)."""
        img = np.ascontiguousarray(lm[_PNP_LM_IDX], dtype=np.float64)
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        try:
            ok, rvec, tvec = cv2.solvePnP(_PNP_MODEL, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                return
            rvec, tvec = cv2.solvePnPRefineLM(_PNP_MODEL, img, K, None, rvec, tvec)
            z = float(tvec[2, 0])
            if z <= 0.0:
                return
            proj, _ = cv2.projectPoints(_PNP_MODEL, rvec, tvec, K, None)
        except cv2.error:
            return
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        if rms / max(1.0, float(np.ptp(img[:, 0]))) > DIST_PNP_MAX_REPROJ:
            return
        self._push("pnp", z * DIST_PNP_SCALE, now)

    def read(self, now):
        """-> {'pnp': cm|None, 'bbox': cm|None} (median các mẫu gần nhất còn mới)."""
        out = {}
        for key in ("pnp", "bbox"):
            d = self._v[key]
            fresh = bool(d) and now - self._t[key] <= DIST_VISION_MAX_AGE_S
            out[key] = float(statistics.median(d)) if fresh else None
        return out

    def add_pair(self, sensor_cm, vals):
        if vals["pnp"] is None and vals["bbox"] is None:
            return
        nan = float("nan")
        with self._mu:
            self._pairs.append((sensor_cm, nan if vals["pnp"] is None else vals["pnp"],
                                nan if vals["bbox"] is None else vals["bbox"]))

    def reset_stats(self):
        with self._mu:
            self._pairs.clear()

    def stats(self):
        """-> {'n': số cặp, 'pnp'|'bbox': None | {n, mae (cm), bias_pct, k}}.
        k = median(sensor / thị_giác): hệ số cần NHÂN THÊM vào giá trị đang hiển thị để khớp cảm biến."""
        with self._mu:
            a = np.array(self._pairs, dtype=np.float64).reshape(-1, 3)
        out = {"n": int(len(a))}
        for j, key in ((1, "pnp"), (2, "bbox")):
            m = ~np.isnan(a[:, j]) if len(a) else np.zeros(0, bool)
            if int(m.sum()) < 3:
                out[key] = None
                continue
            s, v = a[m, 0], a[m, j]
            e = v - s
            out[key] = dict(n=int(m.sum()), mae=float(np.mean(np.abs(e))),
                            bias_pct=float(np.mean(e / s) * 100.0), k=float(np.median(s / v)))
        return out


# ======================= ZOOM SỐ =======================
def apply_zoom(frame, factor):
    """Cắt vùng giữa khung rồi resize về kích thước cũ. -> (ảnh, hệ số thực theo x, hệ số thực theo y)."""
    h, w = frame.shape[:2]
    cw = max(2, int(round(w / factor / 2.0)) * 2)  # chẵn để tâm cắt trùng tâm khung
    ch = max(2, int(round(h / factor / 2.0)) * 2)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    crop = frame[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR), w / cw, h / ch


class ZoomController:
    """Quyết định bật/tắt zoom cho frame kế tiếp."""

    def __init__(self):
        self.active = False
        self._lost_t = None

    def update(self, has_target, gated, dist_state, dist_mm, now):
        if not ZOOM_ENABLE:
            self.active = False
            return False

        if not has_target:  # box biến mất -> về trạng thái ban đầu (sau thời gian ân hạn)
            if self.active:
                if self._lost_t is None:
                    self._lost_t = now
                if now - self._lost_t >= ZOOM_LOST_GRACE_SEC:
                    self.active = False
                    self._lost_t = None
            else:
                self._lost_t = None
            return self.active

        self._lost_t = None
        if not gated:  # mục tiêu lệch tâm: số đo không đại diện cho nó -> giữ nguyên
            return self.active

        far = ((dist_state == "OK" and dist_mm > ZOOM_DIST_THRESHOLD_MM)
               or (ZOOM_FAR_AS_EXCEED and dist_state == "OUT_OF_RANGE"))
        near = dist_state == "OK" and dist_mm < ZOOM_DIST_THRESHOLD_MM - ZOOM_RELEASE_HYST_MM
        if not self.active and far:
            self.active = True
        elif self.active and near:
            self.active = False
        return self.active


# ======================= TRACKING =======================
def parse_tracks(result):
    """-> {id: (x1, y1, x2, y2, conf)} theo tọa độ ảnh gốc, sau lọc box."""
    boxes = result.boxes
    tracks = {}
    if boxes is None or len(boxes) == 0 or boxes.id is None:
        return tracks

    ids = boxes.id.int().cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    candidates = []
    for i, tid in enumerate(ids):
        candidates.append((int(tid), tuple(map(float, xyxy[i])), float(conf[i])))

    # Loại box nhỏ nếu nó nằm hoàn toàn bên trong một box lớn hơn.
    # Quy tắc này không phụ thuộc confidence: box chứa lớn hơn luôn được ưu tiên.
    not_contained = []
    for i, (tid, box, score) in enumerate(candidates):
        x1, y1, x2, y2 = box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        is_contained = False

        for j, (_other_tid, other_box, _other_score) in enumerate(candidates):
            if i == j:
                continue
            ox1, oy1, ox2, oy2 = other_box
            other_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)

            # Chỉ loại box thực sự nhỏ hơn; box bằng kích thước không bị coi là box con.
            if (other_area > area and
                    ox1 <= x1 and oy1 <= y1 and
                    ox2 >= x2 and oy2 >= y2):
                is_contained = True
                break

        if not is_contained:
            not_contained.append((tid, box, score))

    # Greedy NMS cho các box còn lại; loại box có IoU > 0.7.
    not_contained.sort(key=lambda item: item[2], reverse=True)
    kept = []
    for tid, box, score in not_contained:
        x1, y1, x2, y2 = box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        suppress = False

        for _kept_tid, kept_box, _kept_score in kept:
            kx1, ky1, kx2, ky2 = kept_box
            ix1 = max(x1, kx1)
            iy1 = max(y1, ky1)
            ix2 = min(x2, kx2)
            iy2 = min(y2, ky2)
            intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

            kept_area = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)
            union = area + kept_area - intersection
            iou = intersection / union if union > 0.0 else 0.0

            if iou > NMS_IOU_THRESHOLD:
                suppress = True
                break

        if not suppress:
            kept.append((tid, box, score))

    for tid, box, score in kept:
        tracks[tid] = (*box, score)

    return tracks


class CameraStream:
    def __init__(self):
        self._is_win = sys.platform.startswith("win")
        self.cap = None

        for name, backend in self._backend_candidates():
            for (w, h, fps, fourcc_str) in CAP_FALLBACK_MODES:
                cap = self._try_open(backend, w, h, fps, fourcc_str)
                if cap is not None:
                    self.cap = cap
                    print(f"[CAM] mở OK: backend={name} index={CAMERA_INDEX} "
                          f"yêu cầu {w}x{h}@{fps} {fourcc_str or 'default'}")
                    break
                print(f"[CAM] thất bại: backend={name} {w}x{h}@{fps} {fourcc_str or 'default'}")
            if self.cap is not None:
                break

        if self.cap is None:
            raise RuntimeError(
                f"Không đọc được frame từ camera index {CAMERA_INDEX}. "
                "Kiểm tra: app khác đang dùng cam, quyền Camera của Windows, cổng USB, thử CAMERA_INDEX=1/2.")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAP_BUFFERSIZE)
        if CAP_EXPOSURE is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if self._is_win else 1)  # chế độ manual
            self.cap.set(cv2.CAP_PROP_EXPOSURE, CAP_EXPOSURE)

        # Đọc lại giá trị driver THỰC SỰ áp dụng (driver có thể lặng lẽ từ chối)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
        print(f"[CAM] áp dụng : {self.width}x{self.height} "
              f"@ {self.cap.get(cv2.CAP_PROP_FPS):.0f}fps {fourcc}")

        self._cond = threading.Condition()
        self._frame = None
        self._frame_id = 0
        self._stopped = False
        self.alive = True
        self.fps = 0.0  # FPS thực tế của thread capture (EMA)
        self._thread = threading.Thread(target=self._loop, daemon=True)

    @staticmethod
    def _backend_candidates():
        if sys.platform.startswith("win"):
            table = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF}
            first = CAP_BACKEND if CAP_BACKEND in table else "DSHOW"
            order = [first] + [b for b in table if b != first]
            return [(n, table[n]) for n in order]
        if sys.platform.startswith("linux"):
            return [("V4L2", cv2.CAP_V4L2)]
        return [("ANY", cv2.CAP_ANY)]

    def _try_open(self, backend, w, h, fps, fourcc_str):
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if not cap.isOpened():
            cap.release()
            return None

        fcc = cv2.VideoWriter.fourcc(*fourcc_str) if fourcc_str else None
        if fcc is not None and CAP_FOURCC_FIRST:
            cap.set(cv2.CAP_PROP_FOURCC, fcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        if fcc is not None and not CAP_FOURCC_FIRST:
            cap.set(cv2.CAP_PROP_FOURCC, fcc)

        for _ in range(30):  # warm-up: bỏ các frame đầu / chờ stream khởi động
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            time.sleep(0.03)

        cap.release()
        return None

    def start(self):
        self._thread.start()
        return self

    def _loop(self):
        t_prev = time.perf_counter()
        fail = 0
        t_win = time.perf_counter()
        n_win = 0
        while not self._stopped:
            ok, frame = self.cap.read()  # read() cấp phát mảng mới mỗi lần -> không cần copy khi chuyển giao
            if not ok:
                fail += 1
                if fail > 200:
                    break
                time.sleep(0.005)
                continue
            fail = 0

            now = time.perf_counter()
            n_win += 1
            if now - t_win >= 1.0:
                self.fps = n_win / (now - t_win)
                n_win = 0
                t_win = now

            with self._cond:
                self._frame = frame
                self._frame_id += 1
                self._cond.notify_all()

        self.alive = False
        with self._cond:
            self._cond.notify_all()

    def read(self, last_id: int, timeout: float = 1.0):
        with self._cond:
            self._cond.wait_for(lambda: self._frame_id != last_id or not self.alive, timeout)
            if self._frame_id == last_id:
                return last_id, None
            return self._frame_id, self._frame

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=1.0)
        self.cap.release()


# ======================= MASK KHUÔN MẶT =======================
def _is_ascii(p):
    try:
        p.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


@contextlib.contextmanager
def _ascii_cwd():
    """MediaPipe (C++) trên Windows abort cả tiến trình nếu thư mục làm việc có ký tự không phải ASCII
    (vd 'Đồ án môn'). Tạm chuyển sang thư mục gốc của ổ đĩa trong lúc khởi tạo rồi trả lại."""
    old = os.getcwd()
    root = (os.path.splitdrive(sys.executable)[0] or "") + os.sep
    try:
        os.chdir(root)
        yield
    finally:
        os.chdir(old)


def _check_mediapipe_paths():
    """Các đường dẫn KHÔNG thể né bằng chdir: nếu có ký tự lạ thì bỏ mask còn hơn để MediaPipe abort chương trình."""
    bad = [(n, p) for n, p in (("python", sys.executable), ("mediapipe", os.path.dirname(mp.__file__)))
           if not _is_ascii(p)]
    if bad:
        raise RuntimeError("đường dẫn có ký tự không phải ASCII (" + "; ".join(f"{n}: {p}" for n, p in bad) +
                           ") -> MediaPipe sẽ crash. Cài lại Python/venv vào đường dẫn chỉ gồm chữ không dấu")


class FaceMasker:
    """YOLO bbox -> crop -> MediaPipe Face Mesh (468 landmark) -> convex hull.

    - Landmark lưu theo toạ độ CHUẨN HÓA trong bbox (0..1), nên ở frame bỏ qua (FACE_MASK_EVERY_N) vẫn
      chiếu lại đúng lên bbox mới của YOLO, và không bị lệch khi zoom bật/tắt.
    - Lọc EMA trên 468 điểm (số điểm cố định) rồi mới lấy hull -> hull ít rung.
    - Chỉ dùng trong luồng engine (không thread-safe). Chỉ để HIỂN THỊ, không ảnh hưởng tâm bbox gửi ESP32.
    """

    def __init__(self):
        if mp is None:
            raise RuntimeError("chưa cài mediapipe (pip install mediapipe)")
        _check_mediapipe_paths()
        self._lm = None
        self._legacy = False
        if hasattr(mp, "solutions"):  # mediapipe cũ
            self._legacy = True
            with _ascii_cwd():
                self._lm = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1, refine_landmarks=False,
                    min_detection_confidence=0.5)
        else:  # mediapipe mới: Tasks API, cần file .task
            if not os.path.isfile(FACE_LANDMARKER_MODEL):
                if not FACE_LANDMARKER_AUTODOWNLOAD:
                    raise RuntimeError(f"thiếu {FACE_LANDMARKER_MODEL} (mediapipe mới cần face_landmarker.task)")
                self._download_model()
            from mediapipe.tasks.python import BaseOptions, vision
            with open(FACE_LANDMARKER_MODEL, "rb") as f:  # Python đọc được đường dẫn Unicode; C++ thì không
                model_bytes = f.read()
            opts = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=model_bytes),
                running_mode=vision.RunningMode.IMAGE, num_faces=1,
                min_face_detection_confidence=0.5, min_face_presence_confidence=0.5)
            with _ascii_cwd():
                self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._state = {}  # tid -> {"pts": (468,2) float32 theo bbox, "ok": frame_idx lần thành công cuối}
        self._tried = {}  # tid -> frame_idx lần thử cuối (để cách quãng EVERY_N cả khi thất bại)
        self._fi = 0
        # tid -> (468,2) landmark THÔ (chưa EMA) theo pixel khung, chỉ có ở frame vừa chạy MediaPipe. Dùng cho PnP:
        # landmark tính trực tiếp từ crop nên không phụ thuộc bề ngang box YOLO (khác với hull đã chiếu lại lên box).
        self.fresh = {}

    @staticmethod
    def _download_model():
        import urllib.request
        print(f"[MASK] đang tải face_landmarker.task -> {FACE_LANDMARKER_MODEL}")
        try:
            os.makedirs(os.path.dirname(os.path.abspath(FACE_LANDMARKER_MODEL)), exist_ok=True)
            tmp = FACE_LANDMARKER_MODEL + ".part"
            with urllib.request.urlopen(FACE_LANDMARKER_URL, timeout=30) as r, open(tmp, "wb") as f:
                f.write(r.read())
            os.replace(tmp, FACE_LANDMARKER_MODEL)
        except Exception as e:
            raise RuntimeError(f"không tải được face_landmarker.task ({e}); tải tay từ {FACE_LANDMARKER_URL} "
                               f"rồi đặt vào {FACE_LANDMARKER_MODEL}")

    def close(self):
        try:
            if self._lm is not None:
                self._lm.close()
        except Exception:
            pass

    def _run(self, rgb):
        """-> (N,2) float32 chuẩn hóa theo crop, hoặc None."""
        if self._legacy:
            res = self._lm.process(rgb)
            if not res.multi_face_landmarks:
                return None
            lms = res.multi_face_landmarks[0].landmark
        else:
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = self._lm.detect(img)
            if not res.face_landmarks:
                return None
            lms = res.face_landmarks[0]
        return np.array([(p.x, p.y) for p in lms], dtype=np.float32)

    def _detect(self, frame, box):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        H, W = frame.shape[:2]
        cx1 = max(0, int(x1 - FACE_MASK_PAD * bw))
        cy1 = max(0, int(y1 - FACE_MASK_PAD * bh))
        cx2 = min(W, int(x2 + FACE_MASK_PAD * bw))
        cy2 = min(H, int(y2 + FACE_MASK_PAD * bh))
        crop = frame[cy1:cy2, cx1:cx2]
        ch, cw = crop.shape[:2]
        if ch < 8 or cw < 8:
            return None
        lms = self._run(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if lms is None:
            return None
        ax = cx1 + lms[:, 0] * cw  # crop -> toạ độ frame
        ay = cy1 + lms[:, 1] * ch
        return np.stack([(ax - x1) / bw, (ay - y1) / bh], axis=1).astype(np.float32)  # frame -> theo bbox

    def update(self, frame, tracks, follow_id):
        """-> {tid: hull int32 (N,1,2)} theo toạ độ frame đang đưa vào model."""
        self._fi += 1
        self.fresh = {}
        if FACE_MASK_ONLY_TARGET:
            ids = [follow_id] if follow_id in tracks else []
        else:
            ids = list(tracks)
        for tid in [t for t in self._state if t not in ids]:  # dọn id không còn dùng
            self._state.pop(tid, None)
            self._tried.pop(tid, None)

        H, W = frame.shape[:2]
        hulls = {}
        for tid in ids:
            x1, y1, x2, y2, _ = tracks[tid]
            bw, bh = x2 - x1, y2 - y1
            if bw < FACE_MASK_MIN_PX or bh < FACE_MASK_MIN_PX:
                continue
            if self._fi - self._tried.get(tid, -10 ** 9) >= FACE_MASK_EVERY_N:
                self._tried[tid] = self._fi
                new = self._detect(frame, (x1, y1, x2, y2))
                if new is not None:
                    self.fresh[tid] = np.array([x1, y1], np.float32) + new * np.array([bw, bh], np.float32)
                    old = self._state.get(tid)
                    pts = new if old is None else old["pts"] + FACE_MASK_SMOOTH * (new - old["pts"])
                    self._state[tid] = {"pts": pts, "ok": self._fi}
            st = self._state.get(tid)
            if st is None:
                continue
            if self._fi - st["ok"] > FACE_MASK_HOLD_FRAMES:  # mất mặt quá lâu -> bỏ mask
                self._state.pop(tid, None)
                continue
            absp = np.array([x1, y1], np.float32) + st["pts"] * np.array([bw, bh], np.float32)
            absp[:, 0] = np.clip(absp[:, 0], 0, W - 1)
            absp[:, 1] = np.clip(absp[:, 1], 0, H - 1)
            hulls[tid] = cv2.convexHull(absp.astype(np.int32))
        return hulls


def draw_face_masks(frame, hulls, follow_id=None):
    """Vẽ overlay trong suốt + viền cho từng hull, sửa trực tiếp `frame`. Chỉ xử lý trong bounding rect của hull."""
    H, W = frame.shape[:2]
    for tid, hull in hulls.items():
        color = FACE_MASK_COLOR if tid == follow_id else FACE_MASK_COLOR_OTHER
        x, y, w, h = cv2.boundingRect(hull)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        roi = frame[y0:y1, x0:x1]  # view: ghi vào roi là ghi vào frame
        m = np.zeros(roi.shape[:2], np.uint8)
        cv2.fillConvexPoly(m, hull - np.array([x0, y0], np.int32), 255)
        layer = np.empty_like(roi)
        layer[:] = color
        blended = cv2.addWeighted(roi, 1.0 - FACE_MASK_ALPHA, layer, FACE_MASK_ALPHA, 0)
        roi[m > 0] = blended[m > 0]
        cv2.polylines(frame, [hull], True, color, FACE_MASK_OUTLINE_PX, cv2.LINE_AA)


# ======================= ENGINE =======================
class TrackingEngine(threading.Thread):
    """Luồng chạy chương trình chính. GUI chỉ cần: snapshot(), set_lock(), toggle_lock(),
    set_param(), stop(), và đọc các thuộc tính ready / status / error / serial_text."""

    def __init__(self):
        super().__init__(daemon=True)
        self._stop_evt = threading.Event()
        self._mu = threading.Lock()
        self._locked_id = None
        self._snap = None
        self._n = 0
        self._lost_id = None
        self._lost_t = 0.0
        self._last_seen = time.perf_counter()
        self._params = load_tuning()
        self._params_mu = threading.Lock()
        self._dist = None
        self._zoom = ZoomController()
        self._masker = None
        self._vis = VisionDistance() if DIST_VISION_ENABLE else None
        self.mask_text = "-"  # trạng thái mask hiển thị ở GUI
        self.ready = threading.Event()
        self.status = "khởi động..."
        self.error = None
        self.serial_text = "-"

    # ---- API cho UI ----
    def snapshot(self):
        with self._mu:
            return self._snap

    def set_lock(self, tid):
        with self._mu:
            self._locked_id = tid

    def toggle_lock(self, tid):
        with self._mu:
            self._locked_id = None if self._locked_id == tid else tid

    @property
    def locked_id(self):
        with self._mu:
            return self._locked_id

    def _release(self, tid):
        with self._mu:
            if self._locked_id == tid:  # không ghi đè nếu người dùng vừa chọn ID khác
                self._locked_id = None

    def set_param(self, name, value):
        with self._params_mu:
            self._params[name] = value
        send_param(name, value)

    def push_all(self):
        with self._params_mu:
            items = list(self._params.items())
        for name, value in items:
            send_param(name, value)

    def reset_dist_stats(self):
        """GUI gọi (phím r): xóa cửa sổ so sánh sensor / thị giác."""
        if self._vis is not None:
            self._vis.reset_stats()

    def stop(self):
        self._stop_evt.set()

    # ---- Logic chọn mục tiêu ----
    def _decide(self, tracks, now):
        locked = self.locked_id
        if locked is not None:
            if locked in tracks:
                self._lost_id = None
                self._last_seen = now
                return locked, "LOCK"
            if self._lost_id != locked:
                self._lost_id = locked
                self._lost_t = now
            if now - self._lost_t < LOCK_LOST_GRACE_SEC:
                return None, "SEARCH"
            self._release(locked)  # mất ID / đổi ID -> nhả khóa, rơi xuống AUTO
            self._lost_id = None
        if tracks:
            self._last_seen = now
            return min(tracks), "AUTO"
        if now - self._last_seen >= HOME_DELAY_SEC:
            return None, "HOME"
        return None, "SEARCH"

    # ---- Luồng chính ----
    def run(self):
        global _ser
        cam = None
        log_cv_version()
        try:
            device = DEVICE or ("cuda:0" if torch.cuda.is_available() else "cpu")
            self.status = f"đang nạp model ({device})..."
            model = YOLO(MODEL_PATH)
            kw = dict(imgsz=IMG_SIZE, conf=CONF_THRES, device=device, quantize="fp16",
                      max_det=MAX_DET_GUI, classes=TARGET_CLASSES, verbose=False, nms=True, iou=NMS_IOU_THRESHOLD)
            dummy = np.zeros((CAP_HEIGHT, CAP_WIDTH, 3), dtype=np.uint8)
            for _ in range(WARMUP_ITERS):
                model.track(dummy, persist=True, tracker=TRACKER_CFG, **kw)

            self.status = "đang mở ESP32..."
            try:
                serial_open()
            except Exception as e:
                _ser = None
                print(f"[ESP32] không mở được {SERIAL_PORT}: {e}")
            if _ser is not None:
                self.serial_text = f"{SERIAL_PORT} @ {SERIAL_BAUD}"
                self._dist = DistanceReader(_ser)
            elif not SERIAL_ENABLE:
                self.serial_text = "tắt (SERIAL_ENABLE=False)"
            else:
                self.serial_text = f"KHÔNG mở được {SERIAL_PORT}"

            self.status = "đang mở camera..."
            cam = CameraStream().start()
            if not FACE_MASK_ENABLE:
                self.mask_text = "tắt (FACE_MASK_ENABLE=False)"
            else:
                try:
                    self.status = "đang nạp mask khuôn mặt..."
                    self._masker = FaceMasker()
                    self.mask_text = "sẵn sàng"
                except Exception as e:  # thiếu mediapipe / thiếu file .task -> chạy tiếp, không mask
                    self._masker = None
                    self.mask_text = f"TẮT: {e}"
                    print(f"[MASK] tắt mask khuôn mặt: {e}")
            self.push_all()  # đẩy toàn bộ tham số sang ESP32 ngay khi sẵn sàng
            self.ready.set()
            self._loop(model, kw, cam)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(f"[ENGINE] {self.error}")
        finally:
            send_error(0.0, 0.0, False)  # báo "không mục tiêu" để servo không bám theo giá trị cũ
            if self._dist is not None:
                self._dist.stop()
            if self._masker is not None:
                self._masker.close()
            if cam is not None:
                cam.stop()
            serial_close()

    def _loop(self, model, kw, cam):
        last_id = 0
        n_inf = 0
        t_start = t_win = t_log = t_param = time.perf_counter()
        infer_fps = 0.0
        last_home_tx = 0.0
        csv_f = csv_w = None
        if LOG_CSV_ENABLE:
            csv_f = open(LOG_CSV_PATH, "w", newline="")
            csv_w = csv.writer(csv_f)
            csv_w.writerow(["t_s", "err_x", "err_y", "has_target", "infer_ms", "dist_mm", "zoom", "pnp_cm", "bbox_cm"])
        try:
            while not self._stop_evt.is_set():
                last_id, frame = cam.read(last_id)
                if frame is None:
                    if not cam.alive:
                        raise RuntimeError("Camera ngừng cấp frame")
                    continue

                # Zoom áp dụng TRƯỚC khi đưa vào model
                zx = zy = 1.0
                if self._zoom.active:
                    frame, zx, zy = apply_zoom(frame, ZOOM_FACTOR)
                zoomed = zx > 1.0
                h, w = frame.shape[:2]

                t0 = time.perf_counter()
                result = model.track(frame, persist=True, tracker=TRACKER_CFG, **kw)[0]
                tracks = parse_tracks(result)
                now = time.perf_counter()
                infer_ms = (now - t0) * 1000.0

                n_inf += 1
                if now - t_win >= 1.0:
                    infer_fps = n_inf / (now - t_win)
                    n_inf = 0
                    t_win = now

                follow_id, state = self._decide(tracks, now)
                has_target = follow_id is not None

                # Sai số quy về pixel của khung GỐC (chia hệ số zoom) -> PID/hfov trên ESP32 không đổi
                ex = ey = 0.0
                if has_target:
                    x1, y1, x2, y2, _ = tracks[follow_id]
                    ex = ((x1 + x2) / 2.0 - w / 2.0) / zx
                    ey = ((y1 + y2) / 2.0 - h / 2.0) / zy
                    send_error(ex, ey, True)
                    last_home_tx = 0.0
                elif state == "HOME":
                    if now - last_home_tx >= HOME_RESEND_SEC:
                        send_home()
                        last_home_tx = now
                else:  # SEARCH: báo không mục tiêu, ESP32 tự giữ vận tốc theo hold_ms rồi dừng
                    send_error(0.0, 0.0, False)
                    last_home_tx = 0.0

                # Khoảng cách + quyết định zoom cho frame kế tiếp
                if self._dist is not None:
                    dist_state, dist_mm = self._dist.read()
                else:
                    dist_state, dist_mm = "NO_SERIAL", None
                gated = has_target and abs(ex) < DIST_GATE_ERR_PX and abs(ey) < DIST_GATE_ERR_PX
                self._zoom.update(has_target, gated, dist_state, dist_mm, now)

                if now - t_param >= PARAM_RESEND_SEC:  # gửi lại tham số định kỳ (phòng ESP32 reset / rớt byte)
                    self.push_all()
                    t_param = now

                if LOG_INTERVAL_SEC > 0 and now - t_log >= LOG_INTERVAL_SEC:
                    t_log = now
                    d = f"{dist_mm}mm" if dist_mm is not None else dist_state
                    print(f"[FPS] cam={cam.fps:5.1f} | infer={infer_fps:5.1f} | target={has_target} | "
                          f"err_x={ex:+7.1f} | err_y={ey:+7.1f} | dist={d} | zoom={'x%g' % ZOOM_FACTOR if zoomed else 'off'}")

                # Mask khuôn mặt: chạy SAU send_error để không làm trễ lệnh servo, và nằm ngoài infer_ms
                hulls, mask_ms = {}, 0.0
                if self._masker is not None and tracks:
                    t_m = time.perf_counter()
                    hulls = self._masker.update(frame, tracks, follow_id)
                    if FACE_MASK_DRAW and hulls:
                        draw_face_masks(frame, hulls, follow_id)
                    mask_ms = (time.perf_counter() - t_m) * 1000.0
                    self.mask_text = f"bật  {mask_ms:.1f} ms" if hulls else "không thấy mặt trong box"

                # Đo khoảng cách bằng thị giác, song song với cảm biến (cũng chạy SAU send_error, ngoài infer_ms)
                vis, vis_stats, vis_ms = {"pnp": None, "bbox": None}, {"n": 0}, 0.0
                if self._vis is not None:
                    t_v = time.perf_counter()
                    if has_target:
                        with self._params_mu:
                            hfov = float(self._params.get("hfov", 65.0))
                        if CAM_FOCAL_PX:
                            f0 = CAM_FOCAL_PX * w / CAP_WIDTH
                        else:
                            f0 = (w / 2.0) / np.tan(np.radians(hfov) / 2.0)
                        fx, fy = f0 * zx, f0 * zy  # zoom số = cắt giữa khung rồi phóng to -> tiêu cự hiệu dụng tăng theo
                        self._vis.update_bbox(tracks[follow_id], fx, now)
                        lm = self._masker.fresh.get(follow_id) if self._masker is not None else None
                        if lm is not None:
                            self._vis.update_pnp(lm, fx, fy, w / 2.0, h / 2.0, now)
                    vis = self._vis.read(now)
                    spread = self._dist.spread_mm() if self._dist is not None else None
                    if (gated and dist_state == "OK" and spread is not None
                            and spread <= DIST_CMP_MAX_SPREAD_MM):
                        self._vis.add_pair(dist_mm / 10.0, vis)
                    vis_stats = self._vis.stats()
                    vis_ms = (time.perf_counter() - t_v) * 1000.0

                if csv_w is not None:
                    csv_w.writerow([f"{now - t_start:.4f}", f"{ex:.1f}", f"{ey:.1f}", int(has_target),
                                    f"{infer_ms:.1f}", "" if dist_mm is None else dist_mm, int(zoomed),
                                    "" if vis["pnp"] is None else f"{vis['pnp']:.1f}",
                                    "" if vis["bbox"] is None else f"{vis['bbox']:.1f}"])

                with self._mu:
                    self._n += 1
                    self._snap = dict(n=self._n, frame=frame, tracks=tracks,
                                      dist_vis=dict(pnp_cm=vis["pnp"], bbox_cm=vis["bbox"],
                                                    stats=vis_stats, ms=vis_ms),
                                      face_hulls=hulls, mask_ms=mask_ms,
                                      follow_id=follow_id, state=state, locked=self._locked_id,
                                      ex=ex, ey=ey, cam_fps=cam.fps, infer_fps=infer_fps,
                                      infer_ms=infer_ms, dist_state=dist_state, dist_mm=dist_mm,
                                      zoom_active=zoomed)
        finally:
            if csv_f is not None:
                csv_f.close()


def main():
    engine = TrackingEngine()
    engine.start()
    try:
        while engine.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        engine.join(timeout=4.0)


if __name__ == "__main__":
    main()