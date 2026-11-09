/*
 * Pan-Tilt face tracking - ESP32 DevKit V1 - PID (dạng vận tốc)
 * Nhận từ Python:  E,<error_x>,<error_y>,<has_target>\n   (115200 baud)
 *                  P,<tên_tham_số>,<giá_trị>\n   (chỉnh PID lúc chạy từ GUI, không cần nạp lại code)
 *                  H,<pan>,<tilt>\n               (về vị trí Home)
 * Gửi lên Python:  D,<khoảng_cách_mm>,<valid>\n   (VL53L1X, ~30 Hz; valid=0 nếu ngoài tầm / range_status != RangeValid)
 *                  T,ERR\n                        (không tìm thấy VL53L1X, lặp lại mỗi 1 s)
 *   error_x > 0: mặt nằm bên PHẢI tâm khung
 *   error_y > 0: mặt nằm bên DƯỚI tâm khung
 * Pan  servo -> GPIO18
 * Tilt servo -> GPIO19
 * TOF400C-VL53L1X -> SDA GPIO21, SCL GPIO22 (I2C 0x29), VCC = 3V3, XSHUT/GPIO1 để hở
 * Thư viện: ESP32Servo, VL53L1X (Pololu)
 *
 * Kiến trúc:
 *   - Khi có GÓI MỚI (~fps inference): lọc EMA sai số -> PID -> vận tốc mong muốn vTarget
 *   - Mỗi 5 ms (200 Hz): giới hạn gia tốc -> tích phân vận tốc thành góc -> làm mượt -> servo
 *     => góc đổi liên tục, không bị "bậc thang" theo nhịp gói 30 Hz
 *   - Watchdog: mất mục tiêu / mất gói -> giảm tốc về 0, mất lâu -> reset PID
 */
#include <ESP32Servo.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <Wire.h>
#include <VL53L1X.h>

// ================== CẤU HÌNH ==================
const int PAN_PIN  = 18;
const int TILT_PIN = 19;

const int SERVO_MIN_US = 500;
const int SERVO_MAX_US = 2400;

const float PAN_HOME  = 90.0f;
const float TILT_HOME = 130.0f;
const float PAN_MIN  = 10.0f,  PAN_MAX  = 170.0f;
const float TILT_MIN = 90.0f,  TILT_MAX = 180.0f;

// Đổi +1 <-> -1 nếu gimbal quay NGƯỢC (xa khỏi mặt)
const int PAN_DIR  = -1;
const int TILT_DIR = +1;

const float FRAME_W_PX = 1280.0f;    // phải khớp CAP_WIDTH bên Python

// ---- Cảm biến khoảng cách TOF400C-VL53L1X ----
const int      SDA_PIN = 21;
const int      SCL_PIN = 22;
const uint16_t TOF_MAX_VALID_MM = 4000;     // trên ngưỡng này -> valid=0 (chế độ Long đo tối đa ~4 m). Phải lớn hơn ZOOM_DIST_THRESHOLD_MM bên Python (800)
const uint32_t TOF_BUDGET_US    = 33000;    // timing budget ~33 ms -> ~30 Hz (VL53L1X Long: tối thiểu 20 ms; tăng lên 50-100 ms nếu muốn ít nhiễu hơn)
const uint16_t TOF_PERIOD_MS    = 33;       // chu kỳ đo liên tục, phải >= TOF_BUDGET_US/1000
const unsigned long TOF_POLL_MS = 3;        // chu kỳ hỏi cờ "có mẫu mới" (không chặn)

// ---- Tham số chỉnh được LÚC CHẠY ----
// Giá trị dưới đây chỉ là MẶC ĐỊNH khi ESP32 khởi động (= bộ đã tune trên phần cứng). GUI Python (TUNING_GUI) ghi đè bằng gói
//   P,<tên>,<giá trị>\n
// và gửi lại định kỳ, nên tune xong không cần nạp lại code. Giá trị luôn bị kẹp trong [lo, hi] của bảng params[].
//
// PID đầu ra là vận tốc góc [độ/giây]. Vì góc = tích phân của vận tốc nên P đã triệt tiêu
// sai số tĩnh -> Ki = 0. Kp = omega * DEG_PER_PX, DEG_PER_PX = hfov / FRAME_W_PX,
// omega [rad/s] là băng thông vòng kín. Tăng dần omega từng bước 20-30% cho đến khi bắt đầu
// dao động, rồi lùi về ~60-70% mức đó.
float p_hfov      = 65.0f;   // HFOV webcam ở 1280x720 [độ] (cần ĐO THẬT)
float p_omegaPan  = 4.2f;    // [rad/s]
float p_omegaTilt = 4.2f;    // [rad/s]
float p_kdPan     = 0.0f;   // Kd [(°/s)/(px/s)]
float p_kdTilt    = 0.0f;
float p_deadzone  = 20.0f;   // vùng chết MỀM [px]
float p_maxVel    = 56.0f;   // trần vận tốc khi đã bám ổn định [°/s]
float p_acqVel    = 11.0f;   // trần vận tốc ngay khi VỪA thấy mục tiêu [°/s]
float p_acqRampMs = 650.0f;  // sau khoảng này trần tăng dần từ p_acqVel lên p_maxVel [ms]
float p_maxAccel  = 600.0f;  // giới hạn gia tốc [°/s²]: nhỏ = mượt hơn nhưng trễ hơn
float p_errAlpha  = 0.25f;    // lọc EMA sai số đầu vào, mỗi gói: 1 = tắt lọc, nhỏ = mượt hơn nhưng trễ hơn
float p_dAlpha    = 0.2f;    // lọc D: 0..1, nhỏ = lọc mạnh hơn
float p_holdMs    = 100.0f;  // mất mục tiêu < ngưỡng này: GIỮ vận tốc (chịu rớt detection khi mặt di chuyển nhanh) [ms]

const float JUMP_REJECT_PX = 300.0f;  // sai số nhảy > ngưỡng giữa 2 gói liên tiếp -> nghi detection nhảy sang box khác
const uint8_t JUMP_CONFIRM_PKTS = 3;  // chỉ chấp nhận bước nhảy nếu nó giữ nguyên qua ngần này gói
const float I_LIMIT_DPS    = 10.0f;   // giới hạn đóng góp của thành phần I [°/s]
const float SMOOTH_TAU_S   = 0.01f;   // hằng số thời gian làm mượt đầu ra servo [s]

const unsigned long LOOP_PERIOD_MS = 5;     // chu kỳ cập nhật servo (200 Hz)

// ---- Watchdog ----
const unsigned long PKT_TIMEOUT_MS = 150;   // không nhận được gói nào quá ngưỡng này: giảm tốc về 0
const unsigned long LOST_RESET_MS  = 300;   // mất mục tiêu/gói quá ngưỡng này: reset PID

const float DT_MIN_S = 0.010f;        // kẹp dt giữa 2 gói
const float DT_MAX_S = 0.100f;
const float DT_NOMINAL_S = 0.033f;    // dt dùng cho gói đầu tiên sau khi có mục tiêu
// ===============================================

struct Axis {
  float kp, ki, kd;
  float minA, maxA;
  int   dir;
  // trạng thái PID
  float integ, prevE, dFilt;
  bool  hasPrev;
  // góc
  float cmd;   // góc đích (tích lũy)
  float out;   // góc thực gửi servo (đã làm mượt)
  // chuyển động
  float vTarget;   // vận tốc mong muốn từ PID [°/s] (đã nhân dir)
  float vel;       // vận tốc thực sau giới hạn gia tốc [°/s]
  // lọc sai số đầu vào
  float errF;
  bool  errInit;
};

Servo panServo, tiltServo;
Axis pan  = {0, 0, 0, PAN_MIN,  PAN_MAX,  PAN_DIR,  0, 0, 0, false, PAN_HOME,  PAN_HOME,  0, 0, 0, false};
Axis tilt = {0, 0, 0, TILT_MIN, TILT_MAX, TILT_DIR, 0, 0, 0, false, TILT_HOME, TILT_HOME, 0, 0, 0, false};

VL53L1X tof;
bool    tofOk = false;

char rxBuf[64];
uint8_t rxLen = 0;

volatile bool newPacket = false;
float lastEx = 0, lastEy = 0;
bool  lastHas = false;
unsigned long pktUs = 0;        // thời điểm nhận gói mới nhất
unsigned long prevPktUs = 0;    // thời điểm nhận gói trước đó
bool  hadTargetBefore = false;
unsigned long lastPktMs  = 0;   // lần cuối nhận gói bất kỳ
unsigned long lastSeenMs = 0;   // lần cuối nhận gói có mục tiêu (has=1)
unsigned long acqMs      = 0;   // lúc vừa bắt đầu bám mục tiêu (để tăng trần vận tốc dần)
float accEx = 0, accEy = 0;     // sai số của gói được chấp nhận gần nhất (để phát hiện bước nhảy)
uint8_t jumpCnt = 0;            // số gói liên tiếp bị nghi là bước nhảy

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
static inline int angleToUs(float deg) {
  return (int)(SERVO_MIN_US + (deg / 180.0f) * (SERVO_MAX_US - SERVO_MIN_US));
}
void writeServos() {
  panServo.writeMicroseconds(angleToUs(pan.out));
  tiltServo.writeMicroseconds(angleToUs(tilt.out));
}

void resetPid(Axis &a) {
  a.integ = 0; a.prevE = 0; a.dFilt = 0; a.hasPrev = false;
  a.errInit = false;
}

// Lọc EMA sai số pixel (giảm nhiễu bbox của YOLO trước khi vào PID)
float filterErr(Axis &a, float e) {
  if (!a.errInit) { a.errF = e; a.errInit = true; }
  else            a.errF += p_errAlpha * (e - a.errF);
  return a.errF;
}

// Trả về vận tốc góc [°/s] (chưa nhân dir)
float pidVelocity(Axis &a, float e, float dt, float vmax) {
  // vùng chết mềm: trừ ngưỡng, liên tục tại biên
  float ee = 0.0f;
  if (e >  p_deadzone) ee = e - p_deadzone;
  if (e < -p_deadzone) ee = e + p_deadzone;

  // D (có lọc thông thấp)
  float d = 0.0f;
  if (a.hasPrev) {
    float raw = (ee - a.prevE) / dt;
    a.dFilt += p_dAlpha * (raw - a.dFilt);
    d = a.dFilt;
  } else {
    a.dFilt = 0.0f;
  }
  a.prevE = ee;
  a.hasPrev = true;

  // I có chống windup (kẹp đóng góp của I)
  if (a.ki > 0.0f) {
    a.integ += ee * dt;
    float iMax = I_LIMIT_DPS / a.ki;
    a.integ = clampf(a.integ, -iMax, iMax);
  }

  float v = a.kp * ee + a.ki * a.integ + a.kd * d;
  return clampf(v, -vmax, vmax);
}

// Chạy khi có GÓI MỚI: lọc sai số -> PID -> vận tốc mong muốn (đã nhân dir)
void updateAxis(Axis &a, float err, float dt, float vmax) {
  float v = pidVelocity(a, filterErr(a, err), dt, vmax);
  a.vTarget = a.dir * v;
}

// Chạy mỗi LOOP_PERIOD_MS: giới hạn gia tốc, rồi tích phân vận tốc thành góc
void stepAxis(Axis &a, float dt) {
  float dv    = a.vTarget - a.vel;
  float maxDv = p_maxAccel * dt;
  a.vel += clampf(dv, -maxDv, maxDv);

  float next = a.cmd + a.vel * dt;
  // Chạm giới hạn cơ khí: dừng hẳn và ngừng tích phân để không windup
  if (next <= a.minA || next >= a.maxA) { a.integ = 0; a.vel = 0; }
  a.cmd = clampf(next, a.minA, a.maxA);
}

// Tính lại Kp/Kd từ các tham số runtime
void applyParams() {
  float dpp = p_hfov / FRAME_W_PX;      // độ / pixel
  pan.kp  = p_omegaPan  * dpp;  pan.kd  = p_kdPan;
  tilt.kp = p_omegaTilt * dpp;  tilt.kd = p_kdTilt;
}

// Bảng tham số: tên (khớp TUNE_PARAMS bên Python), con trỏ, giới hạn an toàn
struct Param { const char *name; float *ptr; float lo, hi; };
const Param params[] = {
  {"hfov",       &p_hfov,      20.0f, 120.0f},
  {"omega_pan",  &p_omegaPan,   0.0f,  12.0f},
  {"omega_tilt", &p_omegaTilt,  0.0f,  12.0f},
  {"kd_pan",     &p_kdPan,      0.0f,   0.5f},
  {"kd_tilt",    &p_kdTilt,     0.0f,   0.5f},
  {"deadzone",   &p_deadzone,   0.0f,  50.0f},
  {"max_vel",    &p_maxVel,     1.0f, 150.0f},
  {"acq_vel",    &p_acqVel,     1.0f, 100.0f},
  {"acq_ramp",   &p_acqRampMs,  0.0f, 3000.0f},
  {"max_accel",  &p_maxAccel,  20.0f, 3000.0f},
  {"err_alpha",  &p_errAlpha,   0.05f,  1.0f},
  {"d_alpha",    &p_dAlpha,     0.05f,  1.0f},
  {"hold_ms",    &p_holdMs,     0.0f, 300.0f},
};

// Gói "P,<tên>,<giá trị>": không tin dữ liệu từ serial -> kiểm tra số hợp lệ và kẹp trong [lo, hi]
void parseParam(const char *line) {
  const char *nameStart = line + 2;                 // sau "P,"
  const char *comma = strchr(nameStart, ',');
  if (!comma) return;
  size_t n = (size_t)(comma - nameStart);
  if (n == 0 || n > 15) return;
  char name[16];
  memcpy(name, nameStart, n); name[n] = '\0';
  char *end = nullptr;
  float v = strtof(comma + 1, &end);
  if (end == comma + 1 || !isfinite(v)) return;
  for (size_t i = 0; i < sizeof(params) / sizeof(params[0]); ++i) {
    if (strcmp(name, params[i].name) == 0) {
      *params[i].ptr = clampf(v, params[i].lo, params[i].hi);
      applyParams();
      return;
    }
  }
}

void parseLine(const char *line) {
  if (line[0] == 'P' && line[1] == ',') { parseParam(line); return; }
  if (line[0] == 'H' && line[1] == ',') {          // H,<pan>,<tilt>: về Home
    float hp, ht;
    if (sscanf(line, "H,%f,%f", &hp, &ht) == 2) {
      pan.cmd  = clampf(hp, PAN_MIN,  PAN_MAX);
      tilt.cmd = clampf(ht, TILT_MIN, TILT_MAX);
      pan.vel = tilt.vel = 0;
      pan.vTarget = tilt.vTarget = 0;
      resetPid(pan); resetPid(tilt);
      hadTargetBefore = false;
    }
    return;
  }
  if (line[0] != 'E') return;
  float ex, ey;
  int has;
  if (sscanf(line, "E,%f,%f,%d", &ex, &ey, &has) == 3) {
    lastEx = ex; lastEy = ey; lastHas = (has != 0);
    prevPktUs = pktUs;
    pktUs = micros();
    newPacket = true;     // nhiều gói dồn lại -> chỉ giữ gói MỚI NHẤT
  }
}

void readSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      rxBuf[rxLen] = '\0';
      parseLine(rxBuf);
      rxLen = 0;
    } else if (c != '\r') {
      if (rxLen < sizeof(rxBuf) - 1) rxBuf[rxLen++] = c;
      else rxLen = 0;
    }
  }
}

// Đọc VL53L1X KHÔNG CHẶN: chỉ gọi read(false) khi dataReady() báo có mẫu mới
// (tương đương read(true) nhưng bỏ vòng chờ để không làm đứng vòng servo 200 Hz).
// Lưu ý: read(false) của Pololu KHÔNG tự kiểm tra dataReady() nên bắt buộc phải gọi dataReady() trước.
void pollTof() {
  static unsigned long tPoll = 0, tErr = 0;
  unsigned long now = millis();

  if (!tofOk) {                                   // báo lỗi mỗi 1 s để GUI hiện "T,ERR"
    if (now - tErr >= 1000) { tErr = now; Serial.println("T,ERR"); }
    return;
  }
  if (now - tPoll < TOF_POLL_MS) return;
  tPoll = now;

  if (!tof.dataReady()) return;                    // chưa có mẫu mới
  tof.read(false);                                 // đọc kết quả + xoá cờ ngắt, không chờ
  uint16_t mm = tof.ranging_data.range_mm;

  // VL53L1X có mã trạng thái riêng: chỉ tin khi RangeValid (0). SigmaFail/SignalFail/OutOfBoundsFail... -> valid=0
  bool valid = (tof.ranging_data.range_status == VL53L1X::RangeValid) &&
               (mm > 0 && mm <= TOF_MAX_VALID_MM);
  Serial.print("D,"); Serial.print(mm); Serial.print(',');
  Serial.println(valid ? 1 : 0);
}

void setup() {
  Serial.begin(115200);   // chỉ in dòng D,... và T,ERR (Python bỏ qua dòng lạ); không in debug khác
  applyParams();

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  tof.setTimeout(500);                // mặc định 0 = không timeout -> init() có thể treo nếu mất dây
  tofOk = tof.init();
  if (tofOk) {
    tof.setDistanceMode(VL53L1X::Long);              // Long ~4 m (Short ~1.3 m, Medium ~3 m)
    tof.setMeasurementTimingBudget(TOF_BUDGET_US);
    tof.startContinuous(TOF_PERIOD_MS);              // đối số = chu kỳ đo [ms]
  }

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  panServo.setPeriodHertz(50);
  tiltServo.setPeriodHertz(50);
  panServo.attach(PAN_PIN,  SERVO_MIN_US, SERVO_MAX_US);
  tiltServo.attach(TILT_PIN, SERVO_MIN_US, SERVO_MAX_US);

  writeServos();
  delay(500);
}

void loop() {
  readSerial();
  pollTof();
  unsigned long nowMs = millis();

  // ---- Có gói mới: cập nhật vận tốc mong muốn ----
  if (newPacket) {
    newPacket = false;
    lastPktMs = nowMs;

    if (lastHas) {
      // Lọc bước nhảy: sai số đổi quá lớn giữa 2 gói liên tiếp thường là detection nhảy sang box khác
      bool accept = true;
      if (hadTargetBefore) {
        float jx = lastEx - accEx, jy = lastEy - accEy;
        if (sqrtf(jx * jx + jy * jy) > JUMP_REJECT_PX) {
          if (++jumpCnt < JUMP_CONFIRM_PKTS) accept = false;           // bỏ qua gói này
          else { hadTargetBefore = false; jumpCnt = 0; }               // bước nhảy giữ ổn định: coi là mục tiêu mới
        } else {
          jumpCnt = 0;
        }
      }

      if (accept) {
        float dt;
        if (!hadTargetBefore) {
          // vừa thấy mục tiêu: xoá trạng thái cũ, dùng dt danh định, bắt đầu tăng trần vận tốc từ thấp
          resetPid(pan); resetPid(tilt);
          dt = DT_NOMINAL_S;
          acqMs = nowMs;
        } else {
          dt = clampf((pktUs - prevPktUs) * 1e-6f, DT_MIN_S, DT_MAX_S);
        }

        float vmax = p_maxVel;
        float since = (float)(nowMs - acqMs);
        if (since < p_acqRampMs)
          vmax = p_acqVel + (p_maxVel - p_acqVel) * since / p_acqRampMs;

        updateAxis(pan,  lastEx, dt, vmax);
        updateAxis(tilt, lastEy, dt, vmax);
        accEx = lastEx; accEy = lastEy;
        hadTargetBefore = true;
        lastSeenMs = nowMs;
      }
    }
    // has=0: không làm gì ở đây; watchdog bên dưới tự giảm tốc / reset nếu mất lâu
  }

  // ---- Mỗi LOOP_PERIOD_MS: watchdog + chuyển động + ghi servo ----
  static unsigned long tPrev = 0;
  if (nowMs - tPrev >= LOOP_PERIOD_MS) {
    float dtLoop = clampf((nowMs - tPrev) * 1e-3f, 0.001f, 0.05f);
    tPrev = nowMs;

    if (hadTargetBefore) {
      unsigned long sinceSeen = nowMs - lastSeenMs;
      unsigned long sincePkt  = nowMs - lastPktMs;
      if (sinceSeen > LOST_RESET_MS || sincePkt > LOST_RESET_MS) {
        // mất mục tiêu / mất gói quá lâu: dừng và xoá trạng thái PID
        pan.vTarget = tilt.vTarget = 0;
        resetPid(pan); resetPid(tilt);
        hadTargetBefore = false;
      } else if ((float)sinceSeen > p_holdMs || sincePkt > PKT_TIMEOUT_MS) {
        // mất ngắn: giảm tốc về 0 (không reset để tiếp tục mượt khi mục tiêu quay lại)
        pan.vTarget = tilt.vTarget = 0;
      }
    }

    stepAxis(pan,  dtLoop);
    stepAxis(tilt, dtLoop);

    // Làm mượt đầu ra servo
    float alpha = dtLoop / (SMOOTH_TAU_S + dtLoop);
    pan.out  += alpha * (pan.cmd  - pan.out);
    tilt.out += alpha * (tilt.cmd - tilt.out);
    writeServos();
  }
}
