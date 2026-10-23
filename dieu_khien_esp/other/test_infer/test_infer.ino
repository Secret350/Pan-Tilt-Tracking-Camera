/*
  ESP32 DevKit V1 + 2 x MG90S — Pan/Tilt head tracking (revised 2026-10)

  Backward-compatible with the supplied Tkinter GUI / infer_webcam.py:
    E,<err_x_px>,<err_y_px>,<has_target>\n
    P,<parameter_name>,<value>\n
    H,<pan_deg>,<tilt_deg>\n

  PAN  signal GPIO18; TILT signal GPIO19; Serial 115200.
  Use an external regulated ~5V servo supply with adequate current capacity,
  and CONNECT ITS GND TO ESP32 GND. Do not drive servo power from GPIO/3V3.

  Changes vs supplied firmware:
  - H protocol implemented. Home motion follows same velocity/acceleration limits.
  - No braking between *valid* target packets just because hold_ms is shorter
    than the camera frame interval. Packet timeout is handled separately.
  - E(...,has=0) is a genuine target-loss indication (no accidental reuse).
  - Deceleration-aware desired velocity, rather than P-only velocity alone.
  - Jump rejection requires successive suspect measurements to be similar.
  - The vertical pixels/degree factor uses estimated VFOV, not horizontal FOV.
  - Time-based servo stepping with bounded dt; finite-value serial validation.
  - Send 1us position increments instead of 2us; preserves 50 Hz PWM.
  - Tuned faster initial velocity/ramp, while retaining braking and acceleration limits.

  IMPORTANT: MG90S generally does NOT report shaft angle/speed to the ESP32.
  The speed/acceleration bounds below constrain COMMANDS, not physical motion.
  GUI saved pid_params.json overrides ALL tuning defaults at runtime.
  Bench-test without load before attaching the camera.
*/
#include <ESP32Servo.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <Wire.h>
#include <VL53L0X.h>


// ---------------- Hardware / mechanics ----------------
// MG90S
static constexpr int PAN_PIN = 18;
static constexpr int TILT_PIN = 19;
static constexpr int SERVO_MIN_US = 500;
static constexpr int SERVO_MAX_US = 2400;
static constexpr int PULSE_CHANGE_MIN_US = 1; // smallest integer microsecond update
// VL53L0x
static constexpr int TOF_SDA_PIN = 21;
static constexpr int TOF_SCL_PIN = 22;
static constexpr uint32_t TOF_PERIOD_MS = 50; 
static constexpr uint32_t TOF_SEND_PERIOD_MS = 50;
static constexpr uint32_t TOF_MAX_VALID_MM = 1200;

static constexpr float PAN_HOME = 90.0f;
static constexpr float TILT_HOME = 130.0f;
static constexpr float PAN_MIN = 10.0f;
static constexpr float PAN_MAX = 170.0f;
static constexpr float TILT_MIN = 110.0f;
static constexpr float TILT_MAX = 180.0f;
static constexpr int PAN_DIR = -1;
static constexpr int TILT_DIR = +1;

static constexpr float FRAME_W_PX = 1280.0f;
static constexpr float FRAME_H_PX = 720.0f;
static constexpr float PI_F = 3.14159265358979323846f;

// ---------------- Tunable GUI parameters ----------------
// Same serial names and GUI constraints as the user's existing program.
float p_hfov = 65.0f;        // horizontal field of view, degrees
float p_omegaPan = 2.4f;     // tracking response; GUI will overwrite values
float p_omegaTilt = 2.2f;
float p_kdPan = 0.0f;
float p_kdTilt = 0.0f;
float p_deadzone = 15.0f;   // soft dead zone, pixels
float p_maxVel = 35.0f;     // deg/s, max allowed command velocity
float p_acqVel = 9.0f;     // deg/s, initial maximum velocity
float p_acqRampMs = 350.0f;
float p_maxAccel = 450.0f;  // deg/s^2, command acceleration limit
float p_errAlpha = 0.40f;   // EMA smoothing, per target packet
float p_dAlpha = 0.20f;
float p_holdMs = 70.0f;    // hold last vTarget after E(...,has=0), not frame timeout

// ---------------- Internal timing and safety ----------------
static constexpr uint32_t SERVO_LOOP_MS = 5;
static constexpr uint32_t PACKET_STALE_MS = 140; // no E packets -> start decel
static constexpr uint32_t TRACK_RESET_MS = 350; // no target -> reset tracking filter
static constexpr float MAX_STEP_DT_S = 0.025f;
static constexpr float MIN_PACKET_DT_S = 0.010f;
static constexpr float MAX_PACKET_DT_S = 0.100f;
static constexpr float NOMINAL_PACKET_DT_S = 0.033f;
static constexpr float ERROR_JUMP_PX = 300.0f;
static constexpr float JUMP_STABILITY_PX = 90.0f;
static constexpr uint8_t JUMP_CONFIRM_COUNT = 3;
static constexpr float OUT_SMOOTH_TAU_S = 0.005f; // minimize extra lag; acceleration limit remains active
static constexpr float HOME_POSITION_TOL_DEG = 0.5f;
static constexpr float HOME_VEL_TOL_DPS = 0.8f;

struct Axis {
  float kp, kd;
  float minAngle, maxAngle;
  int dir;
  float commandAngle, outputAngle;
  float velocity, targetVelocity;
  float filteredErr, prevErr, filteredDerivative;
  bool errInitialized, derivativeInitialized;
};

Servo panServo, tiltServo;
Axis pan = {0, 0, PAN_MIN, PAN_MAX, PAN_DIR, PAN_HOME, PAN_HOME,
            0, 0, 0, 0, 0, false, false};
Axis tilt = {0, 0, TILT_MIN, TILT_MAX, TILT_DIR, TILT_HOME, TILT_HOME,
             0, 0, 0, 0, 0, false, false};

struct Param { const char *name; float *value; float minVal, maxVal; };
Param params[] = {
  {"hfov", &p_hfov, 20, 120},
  {"omega_pan", &p_omegaPan, 0, 12},
  {"omega_tilt", &p_omegaTilt, 0, 12},
  {"kd_pan", &p_kdPan, 0, 0.5f},
  {"kd_tilt", &p_kdTilt, 0, 0.5f},
  {"deadzone", &p_deadzone, 0, 50},
  {"max_vel", &p_maxVel, 1, 150},
  {"acq_vel", &p_acqVel, 1, 100},
  {"acq_ramp", &p_acqRampMs, 0, 3000},
  {"max_accel", &p_maxAccel, 20, 3000},
  {"err_alpha", &p_errAlpha, 0.05f, 1},
  {"d_alpha", &p_dAlpha, 0.05f, 1},
  {"hold_ms", &p_holdMs, 0, 300},
};

static inline float limitf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
static inline float signf32(float x) {
  return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
}
static inline int angleToUs(float angle) {
  return (int)lroundf(SERVO_MIN_US + (limitf(angle, 0, 180) / 180.0f) *
                      (SERVO_MAX_US - SERVO_MIN_US));
}

float panDpp = 65.0f / FRAME_W_PX;
float tiltDpp = 65.0f / FRAME_W_PX;

void applyParams() {
  // The HFOV is for horizontal measurement. Estimate vertical FOV using
  // rectilinear optics and the 1280x720 output aspect ratio.
  const float hfovRad = p_hfov * PI_F / 180.0f;
  const float vfovRad = 2.0f * atanf(tanf(0.5f * hfovRad) * FRAME_H_PX / FRAME_W_PX);
  const float vfovDeg = vfovRad * 180.0f / PI_F;
  panDpp = p_hfov / FRAME_W_PX;
  tiltDpp = vfovDeg / FRAME_H_PX;
  pan.kp = p_omegaPan * panDpp;
  tilt.kp = p_omegaTilt * tiltDpp;
  pan.kd = p_kdPan;
  tilt.kd = p_kdTilt;
}

void resetFilter(Axis &a) {
  a.errInitialized = false;
  a.derivativeInitialized = false;
  a.filteredErr = 0;
  a.prevErr = 0;
  a.filteredDerivative = 0;
}

// Desired speed constrained by how far we are from a stop. This is a simple
// *commanded velocity* braking bound, not a measured servo shaft velocity.
float safeVelocity(float requested, float distanceDeg, float limitSpeed) {
  const float dist = fmaxf(0.0f, distanceDeg);
  const float brakeLimit = sqrtf(2.0f * fmaxf(20.0f, p_maxAccel) * dist);
  return limitf(requested, -fminf(limitSpeed, brakeLimit),
                            fminf(limitSpeed, brakeLimit));
}

float trackingLimitSpeed(uint32_t now) ;

void updateTrackingAxis(Axis &a, float rawErrorPx, float dt, float degPerPx,
                        float limitSpeed) {
  if (!a.errInitialized) {
    a.filteredErr = rawErrorPx;
    a.errInitialized = true;
  } else {
    a.filteredErr += p_errAlpha * (rawErrorPx - a.filteredErr);
  }

  const float e = a.filteredErr;
  const float softError = (fabsf(e) > p_deadzone)
          ? signf32(e) * (fabsf(e) - p_deadzone) : 0.0f;

  float deriv = 0;
  if (a.derivativeInitialized) {
    const float rawDerivative = (softError - a.prevErr) / dt;
    a.filteredDerivative += p_dAlpha * (rawDerivative - a.filteredDerivative);
    deriv = a.filteredDerivative;
  } else {
    a.filteredDerivative = 0;
    a.derivativeInitialized = true;
  }
  a.prevErr = softError;

  float requested = a.dir * (a.kp * softError + a.kd * deriv);
  const float errorDistanceDeg = fabsf(softError) * degPerPx;
  a.targetVelocity = safeVelocity(requested, errorDistanceDeg, limitSpeed);
}

// When returning home, brake to the destination using the same speed and
// acceleration limits as normal tracking. Repeated H commands don't restart it.
void updateHomeAxis(Axis &a, float homeAngle) {
  float distance = homeAngle - a.commandAngle;
  if (fabsf(distance) < HOME_POSITION_TOL_DEG &&
      fabsf(a.velocity) < HOME_VEL_TOL_DPS) {
    a.commandAngle = homeAngle;
    a.velocity = 0;
    a.targetVelocity = 0;
    return;
  }
  const float desired = signf32(distance) * p_maxVel;
  a.targetVelocity = safeVelocity(desired, fabsf(distance), p_maxVel);
}

// Fixed-time motion update. Prevent integrating past angle limits.
void stepAxis(Axis &a, float dt) {
  const float maxDv = p_maxAccel * dt;
  a.velocity += limitf(a.targetVelocity - a.velocity, -maxDv, maxDv);
  const float nextAngle = a.commandAngle + a.velocity * dt;
  if ((nextAngle < a.minAngle && a.velocity < 0) ||
      (nextAngle > a.maxAngle && a.velocity > 0)) {
    a.velocity = 0;
    a.targetVelocity = 0;
  }
  a.commandAngle = limitf(nextAngle, a.minAngle, a.maxAngle);

  const float alpha = dt / (OUT_SMOOTH_TAU_S + dt);
  a.outputAngle += alpha * (a.commandAngle - a.outputAngle);
  a.outputAngle = limitf(a.outputAngle, a.minAngle, a.maxAngle);
}

int lastPanPulse = -1;
int lastTiltPulse = -1;
void writeServos() {
  const int p = angleToUs(pan.outputAngle);
  const int t = angleToUs(tilt.outputAngle);
  if (lastPanPulse < 0 || abs(p - lastPanPulse) >= PULSE_CHANGE_MIN_US) {
    panServo.writeMicroseconds(p);
    lastPanPulse = p;
  }
  if (lastTiltPulse < 0 || abs(t - lastTiltPulse) >= PULSE_CHANGE_MIN_US) {
    tiltServo.writeMicroseconds(t);
    lastTiltPulse = t;
  }
}

enum Mode : uint8_t { IDLE, TRACK, SEARCH, HOME };
Mode mode = IDLE;
uint32_t lastPacketMs = 0;
uint32_t lastTargetMs = 0;
uint32_t firstMissMs = 0;
uint32_t acquisitionMs = 0;
uint32_t lastAcceptedUs = 0;
float acceptedEx = 0, acceptedEy = 0;
float jumpCandidateEx = 0, jumpCandidateEy = 0;
uint8_t jumpCount = 0;
bool seenAcceptedTarget = false;
float goalPan = PAN_HOME;
float goalTilt = TILT_HOME;

void transitionToSearch(uint32_t now) {
  if (mode == SEARCH || mode == HOME) return;
  mode = SEARCH;
  firstMissMs = now;
  jumpCount = 0;
}

void parseParam(char *line) {
  char *name = line + 2;
  char *comma = strchr(name, ',');
  if (!comma) return;
  *comma = '\0';
  if (strlen(name) == 0 || strlen(name) > 15) return;
  char *end = nullptr;
  float val = strtof(comma + 1, &end);
  if (end == comma + 1 || !isfinite(val) || (*end != '\0' && *end != '\r')) return;
  for (size_t i = 0; i < sizeof(params) / sizeof(params[0]); ++i) {
    if (strcmp(name, params[i].name) == 0) {
      *params[i].value = limitf(val, params[i].minVal, params[i].maxVal);
      applyParams();
      return;
    }
  }
}

void parseHome(const char *line, uint32_t now) {
  float p = 0, t = 0;
  if (sscanf(line, "H,%f,%f", &p, &t) != 2 || !isfinite(p) || !isfinite(t)) return;
  goalPan = limitf(p, PAN_MIN, PAN_MAX);
  goalTilt = limitf(t, TILT_MIN, TILT_MAX);
  // Do not zero current velocity; decelerate/reverse with bounded acceleration.
  if (mode != HOME) {
    resetFilter(pan);
    resetFilter(tilt);
    seenAcceptedTarget = false;
    jumpCount = 0;
  }
  mode = HOME;
  lastPacketMs = now;
}

float trackingLimitSpeed(uint32_t now) {
  if (p_acqRampMs < 1.0f) return p_maxVel;
  float fraction = limitf((float)(now - acquisitionMs) / p_acqRampMs, 0, 1);
  const float speed = p_acqVel + (p_maxVel - p_acqVel) * fraction;
  return limitf(speed, 1, p_maxVel);
}

void parseError(const char *line, uint32_t now) {
  float ex = 0, ey = 0;
  int has = 0;
  if (sscanf(line, "E,%f,%f,%d", &ex, &ey, &has) != 3 ||
      !isfinite(ex) || !isfinite(ey) || (has != 0 && has != 1)) return;

  lastPacketMs = now;
  if (!has) {
    // E(...,0) is not a fresh target. Start loss grace only on first miss.
    if (mode == TRACK) transitionToSearch(now);
    return;
  }

  // Reject an abrupt ID swap unless the *new* location persists across
  // successive readings. Ordinary reacquisition after a loss may start anew.
  if (mode == TRACK && seenAcceptedTarget) {
    float dx = ex - acceptedEx, dy = ey - acceptedEy;
    if (sqrtf(dx * dx + dy * dy) > ERROR_JUMP_PX) {
      const float sx = ex - jumpCandidateEx;
      const float sy = ey - jumpCandidateEy;
      if (jumpCount == 0 || sqrtf(sx * sx + sy * sy) > JUMP_STABILITY_PX) {
        jumpCandidateEx = ex;
        jumpCandidateEy = ey;
        jumpCount = 1;
      } else if (jumpCount < JUMP_CONFIRM_COUNT) {
        ++jumpCount;
      }
      if (jumpCount < JUMP_CONFIRM_COUNT) {
        pan.targetVelocity = 0;
        tilt.targetVelocity = 0;
        return;
      }
      // A stable new ID is accepted as a new acquisition.
      resetFilter(pan);
      resetFilter(tilt);
      seenAcceptedTarget = false;
      jumpCount = 0;
    } else {
      jumpCount = 0;
    }
  }

  const uint32_t nowUs = micros();
  const bool newlyAcquired = mode != TRACK || !seenAcceptedTarget;
  const float dt = newlyAcquired ? NOMINAL_PACKET_DT_S :
       limitf((uint32_t)(nowUs - lastAcceptedUs) * 1e-6f,
              MIN_PACKET_DT_S, MAX_PACKET_DT_S);
  if (newlyAcquired) {
    resetFilter(pan);
    resetFilter(tilt);
    acquisitionMs = now;
  }
  // Preserve velocity when reacquiring: smoothly decelerate/reverse instead
  // of teleporting the commanded angle or speed.
  mode = TRACK;
  const float speedLimit = trackingLimitSpeed(now);
  updateTrackingAxis(pan, ex, dt, panDpp, speedLimit);
  updateTrackingAxis(tilt, ey, dt, tiltDpp, speedLimit);
  acceptedEx = ex;
  acceptedEy = ey;
  lastAcceptedUs = nowUs;
  lastTargetMs = now;
  seenAcceptedTarget = true;
}

char rxBuffer[80];
size_t rxSize = 0;
void parseLine(char *line, uint32_t now) {
  if (line[0] == 'P' && line[1] == ',') parseParam(line);
  else if (line[0] == 'H' && line[1] == ',') parseHome(line, now);
  else if (line[0] == 'E' && line[1] == ',') parseError(line, now);
}

void readSerial(uint32_t now) {
  while (Serial.available() > 0) {
    const char c = (char)Serial.read();
    if (c == '\n') {
      rxBuffer[rxSize] = '\0';
      if (rxSize > 0) parseLine(rxBuffer, now);
      rxSize = 0;
    } else if (c != '\r') {
      if (rxSize < sizeof(rxBuffer) - 1) rxBuffer[rxSize++] = c;
      else rxSize = 0; // discard an oversized line
    }
  }
}

void setup() {
  Serial.begin(115200);
  applyParams();
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  panServo.setPeriodHertz(50);
  tiltServo.setPeriodHertz(50);
  panServo.attach(PAN_PIN, SERVO_MIN_US, SERVO_MAX_US);
  tiltServo.attach(TILT_PIN, SERVO_MIN_US, SERVO_MAX_US);
  writeServos();
  delay(500);
}

void loop() {
  const uint32_t now = millis();
  readSerial(now);

  static uint32_t lastStepMs = 0;
  if (lastStepMs == 0) lastStepMs = now;
  const uint32_t elapsedMs = (uint32_t)(now - lastStepMs);
  if (elapsedMs < SERVO_LOOP_MS) return;
  lastStepMs = now;
  const float dt = limitf(elapsedMs * 0.001f, 0.001f, MAX_STEP_DT_S);

  // Distinguish an explicit E(...,0) from a processing stall without E packets.
  if (mode == TRACK && (uint32_t)(now - lastPacketMs) > PACKET_STALE_MS)
    transitionToSearch(now);

  if (mode == SEARCH) {
    if ((uint32_t)(now - firstMissMs) > (uint32_t)p_holdMs) {
      pan.targetVelocity = 0;
      tilt.targetVelocity = 0;
    }
    if ((uint32_t)(now - lastTargetMs) > TRACK_RESET_MS) {
      resetFilter(pan);
      resetFilter(tilt);
      seenAcceptedTarget = false;
    }
  } else if (mode == HOME) {
    updateHomeAxis(pan, goalPan);
    updateHomeAxis(tilt, goalTilt);
  } else if (mode == IDLE) {
    pan.targetVelocity = 0;
    tilt.targetVelocity = 0;
  }

  stepAxis(pan, dt);
  stepAxis(tilt, dt);
  writeServos();
}
