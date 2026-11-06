"""
gui.py - Giao diện điều khiển Pan-Tilt Tracking (nền sáng, phong cách Claude Code)

File này CHỈ lo giao diện: vẽ video + box, danh sách ID, thanh trượt PID, trạng thái, khoảng cách.
Toàn bộ phần chạy chương trình (camera, model, tracking, serial ESP32, cảm biến VL53L0X, zoom)
nằm trong infer_webcam.py (core) -> lớp TrackingEngine.

CÁCH DÙNG
  Để gui.py CÙNG THƯ MỤC với infer_webcam.py rồi chạy:  python gui.py

LOGIC BÁM MỤC TIÊU (nằm ở core)
  - Bấm một ID trong danh sách (hoặc bấm vào box trên video)  -> KHÓA vào ID đó.
  - Bấm lại ID đang khóa, hoặc bấm hàng "auto", hoặc phím Esc -> về chế độ AUTO.
  - ID bị khóa biến mất / đổi ID quá LOCK_LOST_GRACE_SEC -> nhả khóa, tự bám ID nhỏ nhất.
  - Không còn ID nào quá HOME_DELAY_SEC -> gửi lệnh Home (pan=90, tilt=130) sang ESP32.
"""
import os
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import ctypes
import importlib
import math
import tkinter as tk
from tkinter import font as tkfont

import cv2
from PIL import Image, ImageTk

# ======================= CẤU HÌNH GUI =======================
CORE_MODULE = "infer_webcam"

DISPLAY_W = 800
MAX_ROWS = 6
ROW_H = 26
PANEL_W = 372

core = importlib.import_module(CORE_MODULE)

# ======================= BẢNG MÀU (kiểu Claude Code, sáng) =======================
BG = "#FAF9F5"
CARD = "#FFFFFF"
BORDER = "#E3DFD3"
ROW_LINE = "#F1EFE7"
TEXT = "#1F1E1D"
MUTED = "#85837A"
ACCENT = "#D97757"
ACCENT_DK = "#C15F3C"
TINT = "#FBEDE6"
GREEN = "#3E8E5A"
AMBER = "#B7791F"
BLUE = "#4A6FA5"
RED = "#B3402A"
TRACK = "#E3DFD3"
ACCENT_BGR = (87, 119, 217)
IDLE_BGR = (215, 215, 215)

STATE_COLOR = {"LOCK": ACCENT, "AUTO": GREEN, "SEARCH": AMBER, "HOME": BLUE}

# ======================= VẼ KHUNG HÌNH =======================
def _chip(img, text, y, fg, bg):
    """Nhãn chữ nền đặc ở góc trái video. Trả về tọa độ y đáy của nhãn."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 1)
    cv2.rectangle(img, (8, y), (8 + tw + 14, y + th + 12), bg, -1)
    cv2.putText(img, text, (15, y + th + 5), font, 0.55, fg, 1, cv2.LINE_AA)
    return y + th + 12


def dist_text(dist_state, dist_mm):
    if dist_state == "OK":
        return f"{dist_mm / 10:.1f} cm"
    return {"OUT_OF_RANGE": "ngoài tầm đo", "SENSOR_ERR": "lỗi cảm biến",
            "NO_SERIAL": "tắt", "NO_DATA": "--"}.get(dist_state, "--")


def _cm(v):
    return "--" if v is None else f"{v:.1f}"


def vision_rows(dv, dist_state, dist_mm):
    """Định dạng 2 dòng Status so sánh thị giác với cảm biến.
    -> (chữ "đo ảnh", có_số, chữ "so sánh", có_cặp). Trong ngoặc là chênh lệch (thị giác - sensor), cm."""
    s_cm = dist_mm / 10.0 if dist_state == "OK" and dist_mm is not None else None

    def one(name, v):
        if v is None:
            return f"{name} --"
        return f"{name} {v:5.1f}" + (f" ({v - s_cm:+.1f})" if s_cm is not None else "")

    p, b = dv.get("pnp_cm"), dv.get("bbox_cm")
    row1 = f"{one('PnP', p)}    {one('bbox', b)}    cm"
    st = dv.get("stats") or {}
    if not st.get("n"):
        return row1, (p is not None or b is not None), "chưa có cặp (cần sensor OK, mặt ở giữa, sensor ổn định)", False
    mp_, mb = st.get("pnp"), st.get("bbox")
    mae = lambda m: "--" if m is None else f"{m['mae']:.1f}"
    kk = lambda m: "--" if m is None else f"{m['k']:.2f}"
    row2 = f"n={st['n']}  MAE pnp {mae(mp_)} / bbox {mae(mb)} cm   k {kk(mp_)} / {kk(mb)}"
    return row1, (p is not None or b is not None), row2, True


def annotate(frame, tracks, follow_id, state, dist_state, dist_mm, zoom_active, dist_vis=None):
    """Resize về DISPLAY_W rồi vẽ box/ID/khoảng cách. Trả về (ảnh RGB, scale)."""
    h, w = frame.shape[:2]
    s = DISPLAY_W / w
    img = cv2.resize(frame, (DISPLAY_W, int(round(h * s))), interpolation=cv2.INTER_LINEAR)
    dh, dw = img.shape[:2]
    cx, cy = dw // 2, dh // 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 24, 1, cv2.LINE_AA)

    for tid, (x1, y1, x2, y2, conf) in tracks.items():
        p1 = (int(x1 * s), int(y1 * s))
        p2 = (int(x2 * s), int(y2 * s))
        followed = tid == follow_id
        col = ACCENT_BGR if followed else IDLE_BGR
        cv2.rectangle(img, p1, p2, col, 3 if followed else 1, cv2.LINE_AA)
        label = f"#{tid} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        ty = max(p1[1], th + 8)
        cv2.rectangle(img, (p1[0], ty - th - 8), (p1[0] + tw + 8, ty), col, -1)
        txt_col = (255, 255, 255) if followed else (60, 60, 60)
        cv2.putText(img, label, (p1[0] + 4, ty - 5), font, 0.5, txt_col, 1, cv2.LINE_AA)
        if followed:
            bc = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.circle(img, bc, 5, col, -1, cv2.LINE_AA)
            cv2.line(img, (cx, cy), bc, col, 1, cv2.LINE_AA)

    banner = {"LOCK": f"LOCK #{follow_id}", "AUTO": f"AUTO #{follow_id}",
              "SEARCH": "SEARCHING...", "HOME": f"HOME {core.HOME_PAN:.0f}/{core.HOME_TILT:.0f}"}[state]
    y = _chip(img, banner, 8, ACCENT_BGR, (255, 255, 255))
    y = _chip(img, f"DIST {dist_text(dist_state, dist_mm)}", y + 6,
              (60, 60, 60) if dist_state == "OK" else (150, 150, 150), (255, 255, 255))
    if dist_vis is not None and core.DIST_VISION_ENABLE:
        has = dist_vis.get("pnp_cm") is not None or dist_vis.get("bbox_cm") is not None
        y = _chip(img, f"PnP {_cm(dist_vis.get('pnp_cm'))} | BOX {_cm(dist_vis.get('bbox_cm'))} cm",
                  y + 6, (60, 60, 60) if has else (150, 150, 150), (255, 255, 255))
    if zoom_active:
        _chip(img, f"ZOOM x{core.ZOOM_FACTOR:g}", y + 6, (255, 255, 255), ACCENT_BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), s


# ======================= WIDGET =======================
class Slider(tk.Canvas):
    W, H = PANEL_W, 26
    X0, X1 = 104, 262

    def __init__(self, master, font, name, default, lo, hi, step, unit, value, on_change):
        super().__init__(master, width=self.W, height=self.H, bg=CARD,
                         highlightthickness=0, cursor="hand2")
        self.font, self.name, self.default = font, name, default
        self.lo, self.hi, self.step, self.unit = lo, hi, step, unit
        self.on_change = on_change
        self.dec = max(0, -math.floor(math.log10(step)))
        self.value = value
        self.bind("<Button-1>", self._drag)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<Double-Button-1>", lambda _e: self.set(self.default))   # nhấp đúp = về mặc định
        self.draw()

    def _snap(self, v):
        v = min(max(v, self.lo), self.hi)
        return round(self.lo + round((v - self.lo) / self.step) * self.step, 4)

    def _drag(self, e):
        frac = min(max((e.x - self.X0) / (self.X1 - self.X0), 0.0), 1.0)
        self.set(self.lo + frac * (self.hi - self.lo))

    def set(self, v):
        v = self._snap(v)
        if v != self.value:
            self.value = v
            self.draw()
            self.on_change(self.name, v)

    def draw(self):
        self.delete("all")
        cy = self.H // 2
        frac = (self.value - self.lo) / (self.hi - self.lo)
        tx = self.X0 + frac * (self.X1 - self.X0)
        changed = abs(self.value - self.default) > 1e-9
        self.create_text(2, cy, text=self.name, anchor="w", font=self.font, fill=TEXT)
        self.create_line(self.X0, cy, self.X1, cy, width=4, fill=TRACK, capstyle="round")
        self.create_line(self.X0, cy, tx, cy, width=4, fill=ACCENT, capstyle="round")
        self.create_oval(tx - 6, cy - 6, tx + 6, cy + 6, fill=ACCENT, outline=CARD, width=2)
        txt = f"{self.value:.{self.dec}f} {self.unit}".strip()
        self.create_text(self.W - 2, cy, text=txt, anchor="e", font=self.font,
                         fill=ACCENT_DK if changed else MUTED)


# ======================= ỨNG DỤNG =======================
class App(tk.Tk):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.title("Pan-Tilt Tracking")
        self.configure(bg=BG)
        self.resizable(False, False)

        fams = set(tkfont.families())
        self.fam = next((f for f in ("Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono",
                                     "Courier New") if f in fams), "TkFixedFont")
        self.vals = core.load_tuning()
        self._last_n = -1
        self._scale = 1.0
        self._photo = None
        self._img_size = None
        self._row_ids = []
        self._last_sig = None

        self._build()

        self.bind("<Escape>", lambda _e: self.engine.set_lock(None))
        self.bind("<Key-s>", lambda _e: self.save())
        self.bind("<Control-s>", lambda _e: self.save())
        self.bind("<Key-q>", lambda _e: self._on_close())
        self.bind("<Key-r>", lambda _e: self.engine.reset_dist_stats())  # xóa cửa sổ so sánh sensor / thị giác
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(15, self._tick)

    # ---- UI ----
    def f(self, size, bold=False):
        return (self.fam, -size, "bold") if bold else (self.fam, -size)

    def _card(self, parent):
        return tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)

    def _heading(self, parent, title, hint=""):
        fr = tk.Frame(parent, bg=BG)
        tk.Label(fr, text="›", fg=ACCENT, bg=BG, font=self.f(15, True)).pack(side="left")
        tk.Label(fr, text=f" {title}", fg=TEXT, bg=BG, font=self.f(14, True)).pack(side="left")
        if hint:
            tk.Label(fr, text=f"  {hint}", fg=MUTED, bg=BG, font=self.f(12)).pack(side="left", pady=(2, 0))
        return fr

    def _button(self, parent, text, cmd, primary=False):
        bg = ACCENT if primary else CARD
        hv = ACCENT_DK if primary else TINT
        b = tk.Label(parent, text=text, bg=bg, fg="#FFFFFF" if primary else TEXT, font=self.f(12),
                     padx=10, pady=1, cursor="hand2", highlightthickness=1,
                     highlightbackground=ACCENT if primary else BORDER)
        b.bind("<Button-1>", lambda _e: cmd())
        b.bind("<Enter>", lambda _e: b.config(bg=hv))
        b.bind("<Leave>", lambda _e: b.config(bg=bg))
        return b

    def _build(self):
        # Header
        hd = tk.Frame(self, bg=BG)
        hd.pack(fill="x", padx=18, pady=(14, 10))
        tk.Label(hd, text=" Pan-Tilt Tracking", fg=TEXT, bg=BG, font=self.f(19, True)).pack(side="left")
        tk.Label(hd, text="  head tracking → ESP32", fg=MUTED, bg=BG, font=self.f(13)).pack(side="left", pady=(7, 0))
        tk.Label(hd, text="esc auto · s save · r reset so sánh · q quit", fg=MUTED, bg=BG,
                 font=self.f(12)).pack(side="right", pady=(7, 0))

        body = tk.Frame(self, bg=BG)
        body.pack(padx=18)
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="n")
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="n", padx=(18, 0))

        # --- video + trạng thái ---
        self._heading(left, "Camera", "bấm vào box để khóa ID").pack(anchor="w", pady=(0, 6))
        vcard = self._card(left)
        vcard.pack()
        vh = int(round(core.CAP_HEIGHT * DISPLAY_W / core.CAP_WIDTH))
        self.video = tk.Canvas(vcard, width=DISPLAY_W, height=vh, bg="#F1EFE7",
                               highlightthickness=0, cursor="crosshair")
        self.video.pack()
        self.vimg = self.video.create_image(0, 0, anchor="nw")
        self.vtext = self.video.create_text(DISPLAY_W // 2, vh // 2, text="khởi động...", fill=MUTED,
                                            font=self.f(15), width=DISPLAY_W - 60, justify="center")
        self.video.bind("<Button-1>", self._on_video_click)

        self._heading(left, "Status").pack(anchor="w", pady=(14, 6))
        scard = self._card(left)
        scard.pack(fill="x")
        grid = tk.Frame(scard, bg=CARD)
        grid.pack(fill="x", padx=14, pady=10)
        self.sv, self.sl = {}, {}
        rows = [("mode", "●"), ("error", "⎿"), ("speed", "⎿"), ("sensor", "⎿"), ("đo ảnh", "⎿"), ("so sánh", "⎿"), ("zoom", "⎿"),
                ("serial", "⎿"), ("mask", "⎿"), ("model", "⎿")]
        for i, (key, glyph) in enumerate(rows):
            tk.Label(grid, text=glyph, fg=ACCENT if i == 0 else MUTED, bg=CARD,
                     font=self.f(13)).grid(row=i, column=0, sticky="w")
            tk.Label(grid, text=key, fg=MUTED, bg=CARD, font=self.f(13),
                     width=7, anchor="w").grid(row=i, column=1, sticky="w", padx=(6, 0))
            self.sv[key] = tk.StringVar(value="-")
            self.sl[key] = tk.Label(grid, textvariable=self.sv[key], fg=TEXT, bg=CARD,
                                    font=self.f(13, key == "mode"), anchor="w")
            self.sl[key].grid(row=i, column=2, sticky="w")
        self.sv["model"].set(f"{os.path.basename(core.MODEL_PATH)}  conf≥{core.CONF_THRES}  {core.TRACKER_CFG}")

        # --- Targets ---
        self._heading(right, "Targets", "bấm ID để khóa · bấm lại = auto").pack(anchor="w", pady=(0, 6))
        tcard = self._card(right)
        tcard.pack()
        self.tc = tk.Canvas(tcard, width=PANEL_W + 16, height=ROW_H * (MAX_ROWS + 2), bg=CARD,
                            highlightthickness=0, cursor="hand2")
        self.tc.pack()
        self.tc.bind("<Button-1>", self._on_target_click)
        self._draw_targets(None)

        # --- Tuning ---
        th = tk.Frame(right, bg=BG)
        th.pack(fill="x", pady=(14, 6))
        self._heading(th, "Tuning", "nhấp đúp = mặc định").pack(side="left")
        self._button(th, "reset", self.reset_params).pack(side="right")
        self._button(th, "save", self.save, primary=True).pack(side="right", padx=(0, 6))

        pcard = self._card(right)
        pcard.pack()
        inner = tk.Frame(pcard, bg=CARD)
        inner.pack(padx=8, pady=8)
        self.sliders = []
        for name, default, lo, hi, step, unit in core.TUNE_PARAMS:
            s = Slider(inner, self.f(12), name, default, lo, hi, step, unit, self.vals[name], self._on_param)
            s.pack()
            self.sliders.append(s)
        self.kp_var = tk.StringVar()
        tk.Label(right, textvariable=self.kp_var, fg=GREEN, bg=BG, font=self.f(12),
                 anchor="w").pack(fill="x", pady=(6, 0))
        self._update_kp()

        # Footer
        self.foot = tk.StringVar(value="")
        tk.Label(self, textvariable=self.foot, fg=MUTED, bg=BG, font=self.f(12),
                 anchor="w").pack(fill="x", padx=18, pady=(8, 12))

    # ---- tham số ----
    def _on_param(self, name, v):
        self.vals[name] = v
        self.engine.set_param(name, v)
        self._update_kp()

    def _update_kp(self):
        dpp = self.vals["hfov"] / core.CAP_WIDTH
        self.kp_var.set(f"  Kp_pan={self.vals['omega_pan'] * dpp:.4f}   Kp_tilt={self.vals['omega_tilt'] * dpp:.4f}")

    def reset_params(self):
        for s in self.sliders:
            s.set(s.default)
        self._flash("đã đưa tham số về mặc định")

    def save(self):
        core.save_tuning(self.vals)
        self._flash(f"đã lưu {core.TUNING_FILE}")

    def _flash(self, msg):
        self.foot.set(msg)
        self.after(2000, lambda: self.foot.set(""))

    def _on_video_click(self, e):
        snap = self.engine.snapshot()
        if not snap:
            return
        px, py = e.x / self._scale, e.y / self._scale
        hits = [(tid, (b[2] - b[0]) * (b[3] - b[1])) for tid, b in snap["tracks"].items()
                if b[0] <= px <= b[2] and b[1] <= py <= b[3]]
        if hits:
            self.engine.toggle_lock(min(hits, key=lambda t: t[1])[0])

    def _on_target_click(self, e):
        i = e.y // ROW_H
        if i == 0:
            self.engine.set_lock(None)
        elif 1 <= i <= len(self._row_ids):
            self.engine.toggle_lock(self._row_ids[i - 1])

    # ---- vẽ danh sách ID ----
    def _draw_targets(self, snap):
        c = self.tc
        c.delete("all")
        W = PANEL_W + 16
        tracks = snap["tracks"] if snap else {}
        follow = snap["follow_id"] if snap else None
        locked = snap["locked"] if snap else None
        ids = sorted(tracks)
        shown = ids[:MAX_ROWS]
        self._row_ids = shown

        def line(i):
            c.create_line(0, (i + 1) * ROW_H, W, (i + 1) * ROW_H, fill=ROW_LINE)

        # hàng 0: AUTO
        auto_sel = locked is None
        c.create_text(14, ROW_H // 2, text="●" if auto_sel else "○", anchor="w",
                      fill=ACCENT if auto_sel else MUTED, font=self.f(13))
        c.create_text(34, ROW_H // 2, text="auto", anchor="w", fill=TEXT, font=self.f(13, True))
        c.create_text(W - 12, ROW_H // 2, text="ID nhỏ nhất", anchor="e", fill=MUTED, font=self.f(12))
        line(0)

        for k, tid in enumerate(shown, start=1):
            y0 = k * ROW_H
            cy = y0 + ROW_H // 2
            is_follow = tid == follow
            if is_follow:
                c.create_rectangle(0, y0, W, y0 + ROW_H, fill=TINT, outline="")
                c.create_rectangle(0, y0, 3, y0 + ROW_H, fill=ACCENT, outline="")
            c.create_text(14, cy, text="●" if is_follow else "○", anchor="w",
                          fill=ACCENT if is_follow else MUTED, font=self.f(13))
            c.create_text(34, cy, text=f"#{tid}", anchor="w", fill=TEXT, font=self.f(13, True))
            c.create_text(100, cy, text=f"conf {tracks[tid][4]:.2f}", anchor="w", fill=MUTED, font=self.f(12))
            if locked == tid:
                c.create_text(W - 12, cy, text="LOCKED", anchor="e", fill=ACCENT_DK, font=self.f(12, True))
            elif is_follow:
                c.create_text(W - 12, cy, text="auto", anchor="e", fill=GREEN, font=self.f(12))
            line(k)

        if not ids:
            c.create_text(14, ROW_H + ROW_H // 2, text="không có head nào trong khung hình",
                          anchor="w", fill=MUTED, font=self.f(12))
        elif len(ids) > MAX_ROWS:
            c.create_text(14, (len(shown) + 1) * ROW_H + ROW_H // 2,
                          text=f"… +{len(ids) - MAX_ROWS} ID nữa (bấm box trên video)",
                          anchor="w", fill=MUTED, font=self.f(12))

    # ---- vòng cập nhật UI ----
    def _tick(self):
        w = self.engine
        snap = w.snapshot()
        if w.error:
            self.video.coords(self.vtext, (self._img_size or (DISPLAY_W, 450))[0] // 2,
                              (self._img_size or (DISPLAY_W, 450))[1] // 2)
            self.video.itemconfig(self.vtext, text=w.error, fill=RED, state="normal")
            self.sv["mode"].set("ERROR")
            self.sl["mode"].config(fg=RED)
        elif snap is None:
            self.video.itemconfig(self.vtext, text=w.status)
        elif snap["n"] != self._last_n:
            self._last_n = snap["n"]
            self._show(snap)

        self.after(15, self._tick)

    def _show(self, snap):
        img, self._scale = annotate(snap["frame"], snap["tracks"], snap["follow_id"], snap["state"],
                                    snap["dist_state"], snap["dist_mm"], snap["zoom_active"],
                                    dist_vis=snap.get("dist_vis"))
        ih, iw = img.shape[:2]
        if (iw, ih) != self._img_size:
            self._img_size = (iw, ih)
            self.video.config(width=iw, height=ih)
            self.video.coords(self.vtext, iw // 2, ih // 2)
        self.video.itemconfig(self.vtext, state="hidden")
        self._photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.video.itemconfig(self.vimg, image=self._photo)

        sig = (tuple((t, round(b[4], 2)) for t, b in sorted(snap["tracks"].items())),
               snap["follow_id"], snap["locked"])
        if sig != self._last_sig:
            self._last_sig = sig
            self._draw_targets(snap)

        st, fid, lk = snap["state"], snap["follow_id"], snap["locked"]
        text = {"LOCK": f"LOCK  #{fid}",
                "AUTO": f"AUTO  #{fid} (ID nhỏ nhất)",
                "SEARCH": "SEARCH" + (f"  (mất lock #{lk})" if lk is not None else "  (đang tìm lại)"),
                "HOME": f"HOME  pan {core.HOME_PAN:.0f} / tilt {core.HOME_TILT:.0f}"}[st]
        self.sv["mode"].set(text)
        self.sl["mode"].config(fg=STATE_COLOR[st])
        if fid is not None:
            self.sv["error"].set(f"x {snap['ex']:+7.1f}   y {snap['ey']:+7.1f}  px")
        else:
            self.sv["error"].set("-")
        self.sv["speed"].set(f"cam {snap['cam_fps']:5.1f} fps   infer {snap['infer_fps']:5.1f} fps "
                             f"({snap['infer_ms']:.0f} ms)")
        ds, dmm = snap["dist_state"], snap["dist_mm"]
        if ds == "OK":
            self.sv["sensor"].set(f"{dmm / 10:6.1f} cm   ({dmm} mm)")
            self.sl["sensor"].config(fg=TEXT)
        elif ds == "OUT_OF_RANGE":
            self.sv["sensor"].set("ngoài tầm đo / không hợp lệ")
            self.sl["sensor"].config(fg=AMBER)
        elif ds == "SENSOR_ERR":
            self.sv["sensor"].set("không tìm thấy VL53L0X (T,ERR)")
            self.sl["sensor"].config(fg=RED)
        elif ds == "NO_SERIAL":
            self.sv["sensor"].set("tắt (không có serial)")
            self.sl["sensor"].config(fg=MUTED)
        else:
            self.sv["sensor"].set("chưa nhận được dữ liệu")
            self.sl["sensor"].config(fg=MUTED)
        if not core.DIST_VISION_ENABLE:
            self.sv["đo ảnh"].set("tắt (DIST_VISION_ENABLE=False)")
            self.sv["so sánh"].set("-")
            self.sl["đo ảnh"].config(fg=MUTED)
            self.sl["so sánh"].config(fg=MUTED)
        else:
            r1, ok1, r2, ok2 = vision_rows(snap.get("dist_vis") or {}, ds, dmm)
            self.sv["đo ảnh"].set(r1)
            self.sl["đo ảnh"].config(fg=TEXT if ok1 else MUTED)
            self.sv["so sánh"].set(r2)
            self.sl["so sánh"].config(fg=TEXT if ok2 else MUTED)
        if not core.ZOOM_ENABLE:
            self.sv["zoom"].set("tắt (ZOOM_ENABLE=False)")
            self.sl["zoom"].config(fg=MUTED)
        elif snap["zoom_active"]:
            self.sv["zoom"].set(f"x{core.ZOOM_FACTOR:g}   đang zoom")
            self.sl["zoom"].config(fg=ACCENT_DK)
        else:
            self.sv["zoom"].set(f"x1.0   (zoom khi > {core.ZOOM_DIST_THRESHOLD_MM / 10:.0f} cm)")
            self.sl["zoom"].config(fg=TEXT)
        self.sv["serial"].set(self.engine.serial_text)
        self.sl["serial"].config(fg=RED if "KHÔNG" in self.engine.serial_text else TEXT)
        mt = self.engine.mask_text
        self.sv["mask"].set(mt if len(mt) <= 60 else mt[:57] + "...")
        self.sl["mask"].config(fg=RED if mt.startswith("TẮT") else (AMBER if mt.startswith("không") else TEXT))

    def _on_close(self):
        try:
            self.save()
        except OSError:
            pass
        self.engine.stop()
        self.engine.join(timeout=4.0)
        print("[TUNE] giá trị hiện tại:")
        for name, *_ in core.TUNE_PARAMS:
            print(f"    {name} = {self.vals[name]}")
        self.destroy()


def main():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    engine = core.TrackingEngine()
    engine.start()
    App(engine).mainloop()


if __name__ == "__main__":
    main()