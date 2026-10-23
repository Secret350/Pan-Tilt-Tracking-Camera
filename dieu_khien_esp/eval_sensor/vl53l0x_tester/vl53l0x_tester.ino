#include <Wire.h>
#include <VL53L0X.h>
#include <ESP32Servo.h>

// ===== CONFIG =====
constexpr int      SDA_PIN           = 21;
constexpr int      SCL_PIN           = 22;
constexpr int      PAN_SERVO_PIN     = 18;     // ĐỔI cho đúng chân servo pan của bạn
constexpr int      SERVO_MIN_US      = 500;
constexpr int      SERVO_MAX_US      = 2400;
constexpr int      SERVO_START_ANGLE = 90;
constexpr uint16_t SERVO_SETTLE_MS   = 250;    // chờ servo ổn định sau mỗi lần quay
constexpr uint32_t SERIAL_BAUD       = 115200;
constexpr uint32_t DEFAULT_BUDGET_US = 33000;
constexpr int      DISCARD_SAMPLES   = 2;      // bỏ các mẫu đầu sau khi đổi góc/chế độ
// ==================

VL53L0X  sensor;
Servo    pan;
uint32_t budgetUs  = DEFAULT_BUDGET_US;
bool     longRange = false;

void applyConfig() {
  sensor.stopContinuous();
  if (longRange) {
    sensor.setSignalRateLimit(0.1);
    sensor.setVcselPulsePeriod(VL53L0X::VcselPeriodPreRange, 18);
    sensor.setVcselPulsePeriod(VL53L0X::VcselPeriodFinalRange, 14);
  } else {
    sensor.setSignalRateLimit(0.25);   // giá trị mặc định của thư viện
    sensor.setVcselPulsePeriod(VL53L0X::VcselPeriodPreRange, 14);
    sensor.setVcselPulsePeriod(VL53L0X::VcselPeriodFinalRange, 10);
  }
  sensor.setMeasurementTimingBudget(budgetUs);
  sensor.startContinuous();
}

void readSamples(long n) {
  n = constrain(n, 1, 1000);
  for (int i = 0; i < DISCARD_SAMPLES; i++) sensor.readRangeContinuousMillimeters();
  for (long i = 0; i < n; i++) {
    uint16_t d = sensor.readRangeContinuousMillimeters();
    if (sensor.timeoutOccurred()) Serial.println(-1);
    else                          Serial.println((int)d);
  }
  Serial.println("END");
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.setTimeout(2000);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  ESP32PWM::allocateTimer(0);
  pan.setPeriodHertz(50);
  pan.attach(PAN_SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
  pan.write(SERVO_START_ANGLE);

  sensor.setTimeout(500);
  if (!sensor.init()) {
    while (true) { Serial.println("ERR,init"); delay(1000); }
  }
  applyConfig();
  Serial.println("READY");
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;
  if (line == "PING") { Serial.println("PONG"); return; }

  char cmd  = line.charAt(0);
  long arg  = (line.length() > 2) ? line.substring(2).toInt() : 0;

  switch (cmd) {
    case 'M': longRange = (arg != 0); applyConfig(); Serial.println("OK"); break;
    case 'B': budgetUs = constrain(arg, 20000, 500000); applyConfig(); Serial.println("OK"); break;
    case 'A': pan.write(constrain((int)arg, 0, 180)); delay(SERVO_SETTLE_MS); Serial.println("OK"); break;
    case 'R': readSamples(arg); break;
    default:  Serial.println("ERR,cmd");
  }
}
