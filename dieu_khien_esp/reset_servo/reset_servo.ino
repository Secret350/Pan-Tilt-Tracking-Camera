#include <ESP32Servo.h>

// ===== Hằng số =====
const int PAN_PIN  = 18;
const int TILT_PIN = 19;

const float PAN_HOME_DEG  = 90.0f;
const float TILT_HOME_DEG = 130.0f;

Servo panServo;
Servo tiltServo;

// Góc hiện tại (dùng chung với PID/tracking sau này)
float panAngle  = PAN_HOME_DEG;
float tiltAngle = TILT_HOME_DEG;

void setup() {
  Serial.begin(115200);

  // SG90: xung 500-2400 us tương ứng 0-180 độ
  panServo.setPeriodHertz(50);
  tiltServo.setPeriodHertz(50);
  panServo.attach(PAN_PIN, 500, 2400);
  tiltServo.attach(TILT_PIN, 500, 2400);

  panServo.write((int)PAN_HOME_DEG);
  tiltServo.write((int)TILT_HOME_DEG);

  Serial.println("Servo da ve vi tri home: pan=90, tilt=130");
}

void loop() {
  // Chưa làm gì, giữ nguyên vị trí
}