/*
 * ATWFS Hardware Safety Clamp (Arduino)
 *
 * Receives control packets from host over Serial (115200 baud):
 *   Format: "[angle, speed, trust, obstacle_flag]\n"
 *   Example: "[+2.50, 0.800, 0.920, 0]\n"
 *
 * Safety invariants enforced in firmware:
 *   1. Obstacle Emergency Stop: if obstacle_flag != 0, speed = 0.0 immediately.
 *   2. Low-Trust Fallback: if trust < TRUST_MIN (0.40), clamp speed <= LOW_TRUST_SPEED (0.30)
 *      and return steering toward straight (0.0 deg).
 *   3. Minimum Turn Radius vs Speed: lateral acceleration limit a_lat <= A_LAT_MAX
 *      (R_min = v^2 / a_lat_max => |angle| <= atan(L * a_lat_max / v^2)).
 *   4. Maximum Angular Rate Clamp: steering angle cannot change by more than
 *      MAX_DELTA_ANGLE_DEG (5.0 deg) per control cycle.
 *   5. Watchdog Timeout: if no valid packet arrives within WATCHDOG_MS (300 ms),
 *      bring speed to 0.0 and center steering.
 */

#include <Arduino.h>
#include <Servo.h>
#include <math.h>
#include <stdlib.h>

// Pin definitions
const int PIN_STEER_SERVO = 9;
const int PIN_THROTTLE_PWM = 10;
const int PIN_BRAKE_LED = 13;

// Safety constants
const float MAX_ABS_ANGLE_DEG = 30.0f;    // Physical steering limit (deg)
const float MAX_DELTA_ANGLE_DEG = 5.0f;   // Max angular step per cycle (deg/step)
const float TRUST_MIN = 0.40f;            // Minimum trust before low-trust fallback
const float LOW_TRUST_SPEED = 0.30f;      // Fallback creep speed when trust is low
const float WHEELBASE_M = 1.20f;          // Vehicle wheelbase L (m)
const float MAX_SPEED_MPS = 5.0f;         // Full-scale speed corresponding to throttle=1.0
const float MAX_LAT_ACCEL = 2.5f;         // Max lateral acceleration (m/s^2)
const unsigned long WATCHDOG_MS = 300UL;  // Serial timeout (ms)

Servo steerServo;

float currentAngleDeg = 0.0f;
float currentSpeedNorm = 0.0f;
unsigned long lastPacketMs = 0UL;
String rxBuffer = "";

// Clamp helper
float clampVal(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

// Compute maximum safe steering angle (deg) for a given normalized speed [0, 1]
// based on minimum turn radius R_min = v^2 / a_lat_max
float maxAngleForSpeed(float speedNorm) {
  float v = clampVal(speedNorm, 0.0f, 1.0f) * MAX_SPEED_MPS;
  if (v < 0.5f) {
    return MAX_ABS_ANGLE_DEG;
  }
  float rMin = (v * v) / MAX_LAT_ACCEL;
  float maxRad = atanf(WHEELBASE_M / fmaxf(rMin, 0.1f));
  float maxDeg = maxRad * (180.0f / 3.14159265f);
  return clampVal(maxDeg, 3.0f, MAX_ABS_ANGLE_DEG);
}

// Apply all safety clamps to incoming command
void applySafetyClamp(float reqAngle, float reqSpeed, float trust, int obstacleFlag,
                      float &outAngle, float &outSpeed) {
  if (!isfinite(reqAngle) || !isfinite(reqSpeed) || !isfinite(trust)) {
    reqAngle = 0.0f; reqSpeed = 0.0f; trust = 0.0f; obstacleFlag = 1;
  }
  float targetAngle = clampVal(reqAngle, -MAX_ABS_ANGLE_DEG, MAX_ABS_ANGLE_DEG);
  float targetSpeed = clampVal(reqSpeed / MAX_SPEED_MPS, 0.0f, 1.0f);

  // 1. Obstacle stop
  if (obstacleFlag != 0) {
    targetSpeed = 0.0f;
  }

  // 2. Low-trust fallback: slow down and straighten steering
  if (trust < TRUST_MIN) {
    targetSpeed = fminf(targetSpeed, LOW_TRUST_SPEED / MAX_SPEED_MPS);
    targetAngle = 0.0f;
  }

  // 3. Minimum turn radius clamp vs speed
  float speedLimitAngle = maxAngleForSpeed(targetSpeed);
  targetAngle = clampVal(targetAngle, -speedLimitAngle, speedLimitAngle);

  // 4. Max angular change per control step
  float delta = targetAngle - currentAngleDeg;
  delta = clampVal(delta, -MAX_DELTA_ANGLE_DEG, MAX_DELTA_ANGLE_DEG);
  currentAngleDeg = clampVal(currentAngleDeg + delta, -MAX_ABS_ANGLE_DEG, MAX_ABS_ANGLE_DEG);
  currentSpeedNorm = targetSpeed;

  outAngle = currentAngleDeg;
  outSpeed = currentSpeedNorm;
}

// Parse "[angle, speed, trust, obstacle_flag]"
bool parsePacket(const String &line, float &angle, float &speed, float &trust, int &obs) {
  int startIdx = line.indexOf('[');
  int endIdx = line.indexOf(']');
  if (startIdx < 0 || endIdx <= startIdx) {
    return false;
  }
  String body = line.substring(startIdx + 1, endIdx);

  int c1 = body.indexOf(',');
  if (c1 < 0) return false;
  int c2 = body.indexOf(',', c1 + 1);
  if (c2 < 0) return false;
  int c3 = body.indexOf(',', c2 + 1);
  if (c3 < 0) return false;

  String fields[4] = {body.substring(0, c1), body.substring(c1 + 1, c2),
                      body.substring(c2 + 1, c3), body.substring(c3 + 1)};
  float values[4];
  for (int i = 0; i < 4; ++i) {
    fields[i].trim();
    char *end;
    const char *begin = fields[i].c_str();
    values[i] = strtod(begin, &end);
    if (end == begin || *end != '\0' || !isfinite(values[i])) return false;
  }
  if (values[2] < 0.0f || values[2] > 1.0f || (values[3] != 0 && values[3] != 1)) return false;
  angle = values[0]; speed = values[1]; trust = values[2]; obs = (int)values[3];
  return true;
}

void writeActuators(float angleDeg, float speedNorm, int obstacleFlag) {
  // Map [-30, +30] deg to [60, 120] servo degrees (90 = straight)
  int servoPos = (int)roundf(90.0f + angleDeg);
  servoPos = constrain(servoPos, 60, 120);
  steerServo.write(servoPos);

  int pwmVal = (int)roundf(clampVal(speedNorm, 0.0f, 1.0f) * 255.0f);
  analogWrite(PIN_THROTTLE_PWM, pwmVal);
  digitalWrite(PIN_BRAKE_LED, (obstacleFlag != 0 || speedNorm <= 0.01f) ? HIGH : LOW);
}

void setup() {
  Serial.begin(115200);
  steerServo.attach(PIN_STEER_SERVO);
  pinMode(PIN_THROTTLE_PWM, OUTPUT);
  pinMode(PIN_BRAKE_LED, OUTPUT);

  writeActuators(0.0f, 0.0f, 1);
  lastPacketMs = millis();
}

void loop() {
  while (Serial.available() > 0) {
    char ch = (char)Serial.read();
    if (ch == '\n') {
      float reqAngle = 0.0f, reqSpeed = 0.0f, trust = 0.0f;
      int obs = 1;
      if (parsePacket(rxBuffer, reqAngle, reqSpeed, trust, obs)) {
        float safeAngle = 0.0f, safeSpeed = 0.0f;
        applySafetyClamp(reqAngle, reqSpeed, trust, obs, safeAngle, safeSpeed);
        writeActuators(safeAngle, safeSpeed, obs);
        lastPacketMs = millis();

        // Echo clamped state for diagnostics
        Serial.print("ACK [");
        Serial.print(safeAngle, 2);
        Serial.print(", ");
        Serial.print(safeSpeed * MAX_SPEED_MPS, 3);
        Serial.print(", ");
        Serial.print(trust, 3);
        Serial.print(", ");
        Serial.print(obs);
        Serial.println("]");
      }
      rxBuffer = "";
    } else if (ch != '\r') {
      rxBuffer += ch;
      if (rxBuffer.length() > 128) {
        rxBuffer = "";
      }
    }
  }

  // Watchdog safety stop if host link drops
  if (millis() - lastPacketMs > WATCHDOG_MS) {
    float safeAngle = 0.0f, safeSpeed = 0.0f;
    applySafetyClamp(0.0f, 0.0f, 0.0f, 1, safeAngle, safeSpeed);
    writeActuators(safeAngle, safeSpeed, 1);
  }
}
