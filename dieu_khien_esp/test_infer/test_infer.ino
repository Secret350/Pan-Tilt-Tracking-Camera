/*
 * Pan-Tilt face tracking - ESP32 DevKit V1 - PID (dạng vận tốc)
 * Nhận từ Python:  E,<error_x>,<error_y>,<has_target>\n   (115200 baud)
 *   error_x > 0: mặt nằm bên PHẢI tâm khung
 *   error_y > 0: mặt nằm bên DƯỚI tâm khung
 * Pan  servo -> GPIO18
 * Tilt servo -> GPIO19
 * Thư viện: ESP32Servo
 */
#include <ESP32Servo.h>

// ================== CẤU HÌNH ==================
const int PAN_PIN  = 18;
const int TILT_PIN = 19;

const int SERVO_MIN_US = 500;
const int SERVO_MAX_US = 2400;

const float PAN_HOME  = 90.0f;
const float TILT_HOME = 130.0f;
const float PAN_MIN  = 10.0f,  PAN_MAX  = 170.0f;
const float TILT_MIN = 110.0f,  TILT_MAX = 180.0f;

// Đổi +1 <-> -1 nếu gimbal quay NGƯỢC (xa khỏi mặt)
const int PAN_DIR  = -1;
const int TILT_DIR = +1;

// ---- PID: đầu ra là vận tốc góc [độ/giây] ----
//            Kp [(°/s)/px]   Ki [(°/s)/(px·s)]   Kd [(°/s)/(px/s)]
const float KP_PAN  = 0.11f, KI_PAN  = 0.00f, KD_PAN  = 0.08f;
const float KP_TILT = 0.11f, KI_TILT = 0.00f, KD_TILT = 0.08f;

const float DEADZONE_PX    = 25.0f;   // vùng chết MỀM
const float MAX_VEL_DPS    = 60.0f;   // giới hạn vận tốc [°/s]
const float I_LIMIT_DPS    = 10.0f;   // giới hạn đóng góp của thành phần I [°/s]
const float D_FILTER_ALPHA = 0.3f;    // lọc D: 0..1, nhỏ = lọc mạnh hơn
const float SMOOTH_TAU_S   = 0.01f;   // hằng số thời gian làm mượt đầu ra servo [s]

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
};

Servo panServo, tiltServo;
Axis pan  = {KP_PAN,  KI_PAN,  KD_PAN,  PAN_MIN,  PAN_MAX,  PAN_DIR,  0, 0, 0, false, PAN_HOME,  PAN_HOME};
Axis tilt = {KP_TILT, KI_TILT, KD_TILT, TILT_MIN, TILT_MAX, TILT_DIR, 0, 0, 0, false, TILT_HOME, TILT_HOME};

char rxBuf[64];
uint8_t rxLen = 0;

volatile bool newPacket = false;
float lastEx = 0, lastEy = 0;
bool  lastHas = false;
unsigned long pktUs = 0;        // thời điểm nhận gói mới nhất
unsigned long prevPktUs = 0;    // thời điểm nhận gói trước đó
bool  hadTargetBefore = false;

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
}

// Trả về vận tốc góc [°/s] (chưa nhân dir)
float pidVelocity(Axis &a, float e, float dt) {
  // vùng chết mềm: trừ ngưỡng, liên tục tại biên
  float ee = 0.0f;
  if (e >  DEADZONE_PX) ee = e - DEADZONE_PX;
  if (e < -DEADZONE_PX) ee = e + DEADZONE_PX;

  // D (có lọc thông thấp)
  float d = 0.0f;
  if (a.hasPrev) {
    float raw = (ee - a.prevE) / dt;
    a.dFilt += D_FILTER_ALPHA * (raw - a.dFilt);
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
  return clampf(v, -MAX_VEL_DPS, MAX_VEL_DPS);
}

void updateAxis(Axis &a, float err, float dt) {
  float v = pidVelocity(a, err, dt);
  float next = a.cmd + a.dir * v * dt;
  // Chạm giới hạn cơ khí: ngừng tích phân để không windup
  if (next <= a.minA || next >= a.maxA) a.integ = 0;
  a.cmd = clampf(next, a.minA, a.maxA);
}

void parseLine(const char *line) {
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

void setup() {
  Serial.begin(115200);   // không Serial.print debug để khỏi lẫn dữ liệu

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

  if (newPacket) {
    newPacket = false;

    if (lastHas) {
      float dt;
      if (!hadTargetBefore) {
        // vừa thấy mục tiêu: xoá trạng thái cũ, dùng dt danh định
        resetPid(pan); resetPid(tilt);
        dt = DT_NOMINAL_S;
      } else {
        dt = clampf((pktUs - prevPktUs) * 1e-6f, DT_MIN_S, DT_MAX_S);
      }
      updateAxis(pan,  lastEx, dt);
      updateAxis(tilt, lastEy, dt);
      hadTargetBefore = true;
    } else {
      // mất mục tiêu: giữ nguyên góc, xoá trạng thái PID
      resetPid(pan); resetPid(tilt);
      hadTargetBefore = false;
    }
  }

  // Làm mượt đầu ra servo, chạy mỗi 5 ms
  static unsigned long tPrev = 0;
  unsigned long now = millis();
  if (now - tPrev >= 5) {
    float dtLoop = (now - tPrev) * 1e-3f;
    tPrev = now;
    float alpha = dtLoop / (SMOOTH_TAU_S + dtLoop);
    pan.out  += alpha * (pan.cmd  - pan.out);
    tilt.out += alpha * (tilt.cmd - tilt.out);
    writeServos();
  }
}