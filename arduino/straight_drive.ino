// ==================== [핀 설정] ====================
const byte ENC_A_PIN = 2; // 왼쪽 엔코더 (INT0)
const byte ENC_B_PIN = 3; // 오른쪽 엔코더 (INT1)

// 모터 제어 핀 (IN1~IN4 PWM 핀)
const byte MOTOR_A_IN1 = 5; // 왼쪽 모터 정회전
const byte MOTOR_A_IN2 = 6; // 왼쪽 모터 역회전

const byte MOTOR_B_IN3 = 9;  // 오른쪽 모터 정회전
const byte MOTOR_B_IN4 = 10; // 오른쪽 모터 역회전

// ==================== [엔코더 변수] ====================
volatile long posA = 0; // 왼쪽 바퀴 누적 펄스
volatile long posB = 0; // 오른쪽 바퀴 누적 펄스

// ==================== [주행 및 PID 설정] ====================
int baseSpeed = 150; // 기본 직진 속도 (0 ~ 255)

// 직진 동기화용 PID 게인 (필요에 따라 미세 조정)
float Kp = 2.0;
float Ki = 0.01;
float Kd = 0.1;

float errorSum = 0;
float lastError = 0;

unsigned long prevPIDTime = 0;
unsigned long prevPrintTime = 0;

// ==================== [인터럽트 서비스 루틴] ====================
void ISR_EncoderA() { posA++; } // 왼쪽 바퀴 펄스 카운트
void ISR_EncoderB() { posB++; } // 오른쪽 바퀴 펄스 카운트

void setup() {
  Serial.begin(9600);

  pinMode(ENC_A_PIN, INPUT_PULLUP);
  pinMode(ENC_B_PIN, INPUT_PULLUP);

  pinMode(MOTOR_A_IN1, OUTPUT);
  pinMode(MOTOR_A_IN2, OUTPUT);
  pinMode(MOTOR_B_IN3, OUTPUT);
  pinMode(MOTOR_B_IN4, OUTPUT);

  attachInterrupt(digitalPinToInterrupt(ENC_A_PIN), ISR_EncoderA, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_B_PIN), ISR_EncoderB, RISING);

  Serial.println("==========================================");
  Serial.println(" [ 양바퀴 엔코더 직진 동기화 주행 시작 ]");
  Serial.println("==========================================");
}

void loop() {
  unsigned long now = millis();

  // --- [1] 20ms마다 직진 보정 PID 연산 ---
  if (now - prevPIDTime >= 20) {
    float dt = (now - prevPIDTime) / 1000.0;
    prevPIDTime = now;

    // 1. 엔코더 값 원자적 읽기
    long currentPosA, currentPosB;
    noInterrupts();
    currentPosA = posA;
    currentPosB = posB;
    interrupts();

    // 2. 오차 계산 (왼쪽 펄스 - 오른쪽 펄스)
    // 오차가 양수(+)면 왼쪽이 더 많이 돎 -> 왼쪽 줄이고 오른쪽 늘림
    // 오차가 음수(-)면 오른쪽이 더 많이 돎 -> 왼쪽 늘리고 오른쪽 줄임
    float error = currentPosA - currentPosB;

    errorSum += error * dt;
    errorSum = constrain(errorSum, -100, 100); // Windup 방지

    float dError = (error - lastError) / dt;
    lastError = error;

    // 3. 보정값(Correction) 계산
    float correction = (Kp * error) + (Ki * errorSum) + (Kd * dError);

    // 4. 모터별 최종 PWM 계산
    int pwmA = baseSpeed - correction; // 왼쪽 모터
    int pwmB = baseSpeed + correction; // 오른쪽 모터

    // PWM 범위 제한 (0 ~ 255)
    pwmA = constrain(pwmA, 0, 255);
    pwmB = constrain(pwmB, 0, 255);

    // 5. 모터 구동
    driveLeftMotor(pwmA);
    driveRightMotor(pwmB);
  }

  // --- [2] 0.2초마다 모니터링 출력 ---
  if (now - prevPrintTime >= 200) {
    prevPrintTime = now;
    Serial.print("L(posA): "); Serial.print(posA);
    Serial.print(" | R(posB): "); Serial.print(posB);
    Serial.print(" | 오차: "); Serial.print(posA - posB);
    Serial.println();
  }
}

// ==================== [모터 전진 구동 함수] ====================
void driveLeftMotor(int pwm) {
  analogWrite(MOTOR_A_IN1, pwm);
  digitalWrite(MOTOR_A_IN2, LOW);
}

void driveRightMotor(int pwm) {
  analogWrite(MOTOR_B_IN3, pwm);
  digitalWrite(MOTOR_B_IN4, LOW);
}