/*
  signspeak_glove.ino: one sketch for BOTH gloves. Firmware 2.0.

  Set GLOVE_SIDE below and flash. The right glove streams 13 channels
  (5 flex, 3 accel, 3 gyro, 2 contact pads); the left glove streams 11
  (5 flex, 3 accel, 3 gyro, no pads).

  Frame is sampled continuously at 50 Hz in loop() whether or not anything
  is being printed, so the median filter is always warm and the IMU is
  always fresh. Commands only change what gets printed.

  ---- Host protocol (used by record_dataset.py, record_words.py, dual.py) ----
    boot ........ banner line   "GLOVE:RIGHT CHANNELS:13 FW:2.0"
    calibration . prints "PRESS_ENTER: <instruction>" and waits for a newline,
                  twice (flat, fist). Then "Calibration complete!" and
                  "READY FOR DATA COLLECTION".
    STREAM ...... prints "STREAMING", then one CSV frame per 20 ms until
                  ANY byte arrives on serial.
    RAW ......... prints "RAWSTREAM", then 5 raw millivolt values per 20 ms
                  until ANY byte arrives.
    LABEL,REP ... prints "Collecting: ..." then 50 rows
                  "LABEL,REP,sample,<channels>" then "END".
    LABEL,REP,N . same, but N rows (used for word signs, N = 75).
    CAL ......... run calibration again.
    ID .......... reprint the banner.

  ---- Contact pads: ON (ENABLE_TOUCH 1), right glove only ----
  Flex sensors measure how far a finger bends. They cannot measure finger
  spread or where the thumb sits, so A/T (thumb position) and U/V (finger
  spread) are identical to them. The pads sense those contacts directly.
  Wiring:
    middle-finger pad ........ GND (reference, no pin)
    index pad ................ GPIO 13, INPUT_PULLUP
    thumb-tip pad ............ GPIO 4,  INPUT_PULLUP
  Two bits, printed in this order (must match TOUCH in features.py):
    touch_index_mid    index pad reads LOW  (index on the middle pad)
    touch_thumb_mid    thumb pad reads LOW  (thumb on the middle pad)
  To disable: set ENABLE_TOUCH 0 here AND TOUCH = [] in features.py, then
  retrain the letter model.

  Optional third bit (TOUCH_SCAN_THUMB_INDEX 1, off in this build): thumb
  touching index with neither on GND cannot be read against one ground, so
  it is scanned: the index pin is driven LOW for 30 us and the thumb pin is
  read.

  GPIO 4 is a strapping pin: keep the thumb pad off the middle pad while
  flashing. It is irrelevant after boot.
*/

#include <MPU6050_light.h>
#include <Wire.h>

// ============================ SELECT THE GLOVE ============================
#define GLOVE_RIGHT 1
#define GLOVE_LEFT  2
#define GLOVE_SIDE  GLOVE_RIGHT        // <<< change to GLOVE_LEFT for the left board
// ==========================================================================

#define FW_VERSION "2.0"
#define ENABLE_TOUCH 1                 // 1 = 13 channels (contact pads on)
#define TOUCH_SCAN_THUMB_INDEX 0       // only used when ENABLE_TOUCH is 1

#if GLOVE_SIDE == GLOVE_RIGHT
  #define GLOVE_NAME "RIGHT"
  #define HAS_TOUCH ENABLE_TOUCH
#else
  #define GLOVE_NAME "LEFT"
  #define HAS_TOUCH 0
#endif

#if HAS_TOUCH
  #if TOUCH_SCAN_THUMB_INDEX
    #define NUM_TOUCH 3
  #else
    #define NUM_TOUCH 2
  #endif
#else
  #define NUM_TOUCH 0
#endif

// ---------------------------------------------------------------- pins ----
// Thumb, index, middle, ring, pinky. ADC1 only: ADC2 shares the WiFi
// radio, so any flex sensor on an ADC2 pin returns corrupted readings
// whenever WiFi is active.
const int FLEX_PINS[5] = {33, 32, 35, 34, 36};
const int NUM_FLEX = 5;
const int PIN_SDA = 21, PIN_SCL = 22;
const int PIN_TOUCH_THUMB = 4;
const int PIN_TOUCH_INDEX = 13;

// ------------------------------------------------------------ sampling ----
const uint32_t TICK_US = 20000;                  // 50 Hz
const int DEFAULT_SAMPLES_PER_REP = 50;
const int MAX_SAMPLES_PER_REP = 200;
const int MEDIAN_N = 16;
const int CAL_SAMPLES = 150;
const float TRIM = 0.05f;

// --------------------------------------------------------------- state ----
MPU6050 mpu(Wire);
bool mpu_ok = false;

float flex_min[5], flex_max[5];
float med_buf[5][MEDIAN_N];
int med_idx = 0;
bool calibrated = false;

// latest sampled frame
float f_flex[5];
float f_acc[3], f_gyr[3];
int   f_touch[3];

// IMU freeze detector
float last_acc[3] = {0, 0, 0};
int   frozen_ticks = 0;
const int FREEZE_TICKS = 50;                     // 1 s of identical readings

uint32_t next_tick_us = 0;

enum Mode { M_IDLE, M_STREAM, M_RAW, M_COLLECT };
Mode mode = M_IDLE;
String col_label;
int col_rep = 0, col_n = 0, col_i = 0;
uint32_t col_start_us = 0;

// ---------------------------------------------------------- prototypes ---
void banner();
void ready_banner();
void calibrate();
void sample_all();
void print_channels();
void print_raw();
void handle_command(String in);

// ---------------------------------------------------------- utilities ----
void sort_array(float a[], int n) {
  for (int i = 1; i < n; i++) {
    float k = a[i]; int j = i - 1;
    while (j >= 0 && a[j] > k) { a[j + 1] = a[j]; j--; }
    a[j + 1] = k;
  }
}

float trimmed_mean(float a[], int n) {
  sort_array(a, n);
  int t = (int)(n * TRIM);
  float s = 0;
  for (int i = t; i < n - t; i++) s += a[i];
  return s / (n - 2 * t);
}

float read_mv(int pin) { return (float)analogReadMilliVolts(pin); }

void banner() {
  Serial.print("GLOVE:"); Serial.print(GLOVE_NAME);
  Serial.print(" CHANNELS:"); Serial.print(NUM_FLEX + 6 + NUM_TOUCH);
  Serial.print(" FW:"); Serial.println(FW_VERSION);
}

// Wait for the host to send a line. Any line counts, an empty one included.
void wait_for_enter(const char *instruction) {
  while (Serial.available()) Serial.read();
  Serial.print("PRESS_ENTER: "); Serial.println(instruction);
  uint32_t t0 = millis();
  while (true) {
    if (Serial.available()) {
      String s = Serial.readStringUntil('\n');
      // Ignore bytes that were already in flight when the prompt went out
      // (e.g. a host tool nudging "STREAM" too early).
      if (millis() - t0 > 300) break;
    }
    delay(5);
  }
}

// -------------------------------------------------------- calibration ----
void calibrate() {
  float buf[5][CAL_SAMPLES];
  calibrated = false;

  wait_for_enter("hold the hand FLAT, fingers straight, then press Enter");
  Serial.println("sampling flat hand...");
  for (int s = 0; s < CAL_SAMPLES; s++) {
    for (int f = 0; f < 5; f++) buf[f][s] = read_mv(FLEX_PINS[f]);
    delay(10);
  }
  for (int f = 0; f < 5; f++) {
    flex_min[f] = trimmed_mean(buf[f], CAL_SAMPLES);
    Serial.print("Flex "); Serial.print(f + 1);
    Serial.print(" flat: "); Serial.println(flex_min[f], 1);
  }

  wait_for_enter("make a tight FIST, then press Enter");
  Serial.println("sampling fist...");
  for (int s = 0; s < CAL_SAMPLES; s++) {
    for (int f = 0; f < 5; f++) buf[f][s] = read_mv(FLEX_PINS[f]);
    delay(10);
  }
  for (int f = 0; f < 5; f++) {
    flex_max[f] = trimmed_mean(buf[f], CAL_SAMPLES);
    Serial.print("Flex "); Serial.print(f + 1);
    Serial.print(" fist: "); Serial.print(flex_max[f], 1);
    Serial.print("  span: "); Serial.print(flex_max[f] - flex_min[f], 1);
    if (fabsf(flex_max[f] - flex_min[f]) < 100.0f) Serial.print("  WEAK");
    Serial.println();
    if (fabsf(flex_max[f] - flex_min[f]) < 1.0f) flex_max[f] = flex_min[f] + 1.0f;
  }

  // Warm the median filter so the first frames are not zeros.
  for (int i = 0; i < MEDIAN_N; i++) {
    for (int f = 0; f < 5; f++) med_buf[f][i] = read_mv(FLEX_PINS[f]);
    delay(5);
  }
  med_idx = 0;
  calibrated = true;
  Serial.println("Calibration complete!");
}

// ------------------------------------------------------------- sensors ----
void sample_flex() {
  for (int f = 0; f < 5; f++) {
    med_buf[f][med_idx] = read_mv(FLEX_PINS[f]);
    float tmp[MEDIAN_N];
    for (int i = 0; i < MEDIAN_N; i++) tmp[i] = med_buf[f][i];
    sort_array(tmp, MEDIAN_N);
    float med = (tmp[MEDIAN_N / 2 - 1] + tmp[MEDIAN_N / 2]) * 0.5f;
    float v = (med - flex_min[f]) / (flex_max[f] - flex_min[f]);
    f_flex[f] = constrain(v, 0.0f, 1.0f);
  }
  med_idx = (med_idx + 1) % MEDIAN_N;
}

void sample_imu() {
  if (!mpu_ok) { for (int i = 0; i < 3; i++) { f_acc[i] = 0; f_gyr[i] = 0; } return; }
  mpu.update();
  f_acc[0] = mpu.getAccX();  f_acc[1] = mpu.getAccY();  f_acc[2] = mpu.getAccZ();
  f_gyr[0] = mpu.getGyroX(); f_gyr[1] = mpu.getGyroY(); f_gyr[2] = mpu.getGyroZ();

  // Freeze detector: MPU6050_light returns cached values if I2C dies.
  if (f_acc[0] == last_acc[0] && f_acc[1] == last_acc[1] && f_acc[2] == last_acc[2]) {
    if (++frozen_ticks == FREEZE_TICKS) {
      Serial.println("WARN: IMU frozen - reinitialising I2C");
      Wire.begin(PIN_SDA, PIN_SCL);
      Wire.setClock(100000);
      mpu.begin();
      frozen_ticks = 0;
    }
  } else {
    frozen_ticks = 0;
  }
  for (int i = 0; i < 3; i++) last_acc[i] = f_acc[i];
}

void sample_touch() {
#if HAS_TOUCH
  int thumb_low = (digitalRead(PIN_TOUCH_THUMB) == LOW);
  int index_low = (digitalRead(PIN_TOUCH_INDEX) == LOW);
  // Order must match features.TOUCH: index first, then thumb.
  f_touch[0] = index_low;      // index pad on the middle (GND) pad
  f_touch[1] = thumb_low;      // thumb pad on the middle (GND) pad
  f_touch[2] = 0;
  #if TOUCH_SCAN_THUMB_INDEX
  if (!thumb_low && !index_low) {
    // Neither pad is on ground. Pull the index pad low ourselves and see
    // whether the thumb pad follows it: that means thumb-on-index.
    pinMode(PIN_TOUCH_INDEX, OUTPUT);
    digitalWrite(PIN_TOUCH_INDEX, LOW);
    delayMicroseconds(30);
    f_touch[2] = (digitalRead(PIN_TOUCH_THUMB) == LOW);
    pinMode(PIN_TOUCH_INDEX, INPUT_PULLUP);
  }
  #endif
#endif
}

void sample_all() { sample_flex(); sample_imu(); sample_touch(); }

// ------------------------------------------------------------- output ----
void print_channels() {
  for (int f = 0; f < 5; f++) { Serial.print(f_flex[f], 2); Serial.print(','); }
  for (int i = 0; i < 3; i++) { Serial.print(f_acc[i], 2); Serial.print(','); }
  Serial.print(f_gyr[0], 2); Serial.print(','); Serial.print(f_gyr[1], 2);
  Serial.print(','); Serial.print(f_gyr[2], 2);
  for (int i = 0; i < NUM_TOUCH; i++) { Serial.print(','); Serial.print(f_touch[i]); }
  Serial.println();
}

void print_raw() {
  for (int f = 0; f < 5; f++) {
    if (f) Serial.print(',');
    Serial.print(read_mv(FLEX_PINS[f]), 0);
  }
  Serial.println();
}

// ---------------------------------------------------------- commands ----
void stop_mode() {
  mode = M_IDLE;
  while (Serial.available()) Serial.read();
}

void handle_command(String in) {
  in.trim(); in.toUpperCase();
  if (in.length() == 0) return;

  if (in == "STREAM") {
    Serial.println("STREAMING");
    next_tick_us = micros();               // phase-align to the command
    mode = M_STREAM; return;
  }
  if (in == "RAW") {
    Serial.println("RAWSTREAM");
    next_tick_us = micros();
    mode = M_RAW; return;
  }
  if (in == "ID")  { banner(); return; }
  if (in == "CAL") { calibrate(); ready_banner(); return; }

  int c1 = in.indexOf(',');
  if (c1 > 0) {
    String label = in.substring(0, c1);
    String rest  = in.substring(c1 + 1);
    int c2 = rest.indexOf(',');
    int rep = (c2 > 0 ? rest.substring(0, c2) : rest).toInt();
    int n   = (c2 > 0 ? rest.substring(c2 + 1).toInt() : DEFAULT_SAMPLES_PER_REP);
    if (rep <= 0) { Serial.println("Invalid format. Use: LABEL,REP or LABEL,REP,N"); return; }
    n = constrain(n, 1, MAX_SAMPLES_PER_REP);
    col_label = label; col_rep = rep; col_n = n; col_i = 0;
    Serial.print("Collecting: "); Serial.print(label);
    Serial.print(" rep "); Serial.print(rep);
    Serial.print(" ("); Serial.print(n); Serial.println(" samples)");
    Serial.println("Perform the gesture NOW...");
    // Ticks restart now; rows are printed from the first tick at or after
    // 500 ms. Every board that got the command does the same, so two
    // gloves start their rows together. Sampling continues meanwhile,
    // keeping the median filter warm.
    next_tick_us = micros();
    col_start_us = next_tick_us + 500000UL;
    mode = M_COLLECT; return;
  }
  Serial.println("Unknown command. STREAM | RAW | CAL | ID | LABEL,REP[,N]");
}

void ready_banner() {
  Serial.println("========== READY FOR DATA COLLECTION ==========");
  Serial.println("STREAM | RAW | CAL | ID | LABEL,REP[,N]");
}

// ---------------------------------------------------------------- main ----
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);
  delay(500);
  Serial.println();
  banner();

  for (int f = 0; f < 5; f++) pinMode(FLEX_PINS[f], INPUT);
#if HAS_TOUCH
  pinMode(PIN_TOUCH_THUMB, INPUT_PULLUP);
  pinMode(PIN_TOUCH_INDEX, INPUT_PULLUP);
#endif

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(100000);
  Wire.setTimeOut(20);
  byte st = mpu.begin();
  Serial.print("MPU6050 init: "); Serial.println(st);
  mpu_ok = (st == 0);
  if (mpu_ok) {
    Serial.println("hold the hand still: computing IMU offsets");
    mpu.calcOffsets();
    Serial.println("MPU6050 offsets calculated.");
  } else {
    Serial.println("WARNING: MPU6050 failed to initialize! IMU channels will read 0.");
  }

  for (int f = 0; f < 5; f++) { flex_min[f] = 0; flex_max[f] = 3300; }
  calibrate();
  ready_banner();
  next_tick_us = micros();
}

void loop() {
  // Any byte stops STREAM/RAW. In IDLE/COLLECT, lines are commands.
  if (Serial.available()) {
    if (mode == M_STREAM || mode == M_RAW) {
      stop_mode();
      Serial.println("STOPPED");
    } else {
      String in = Serial.readStringUntil('\n');
      handle_command(in);
    }
  }

  uint32_t now = micros();
  if ((int32_t)(now - next_tick_us) < 0) { delayMicroseconds(200); return; }
  next_tick_us += TICK_US;
  if ((int32_t)(micros() - next_tick_us) > (int32_t)TICK_US) next_tick_us = micros();  // fell behind, resync

  sample_all();

  switch (mode) {
    case M_STREAM:
      print_channels();
      break;
    case M_RAW:
      print_raw();
      break;
    case M_COLLECT:
      if ((int32_t)(now - col_start_us) < 0) break;   // still in the 500 ms lead-in
      Serial.print(col_label); Serial.print(','); Serial.print(col_rep);
      Serial.print(','); Serial.print(col_i); Serial.print(',');
      print_channels();
      if (++col_i >= col_n) {
        Serial.println("END");
        Serial.println("Collection complete. Ready for next.");
        mode = M_IDLE;
      }
      break;
    default:
      break;
  }
}
