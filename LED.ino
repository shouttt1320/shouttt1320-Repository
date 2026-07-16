// PWM 제어가 가능한 디지털 핀 정의
const int PIN_RED = 9;
const int PIN_GREEN = 10;
const int PIN_BLUE = 11;

void setup() {
  pinMode(PIN_RED, OUTPUT);
  pinMode(PIN_GREEN, OUTPUT);
  pinMode(PIN_BLUE, OUTPUT);
}

// 💡 RGB 값을 직접 입력받아 빛의 밝기를 섞어주는 편리한 함수 정의
void setColor(int redValue, int greenValue, int blueValue) {
  // analogWrite를 통해 0(꺼짐) ~ 255(최대 밝기) 사이로 세밀하게 전압 제어
  analogWrite(PIN_RED, redValue);
  analogWrite(PIN_GREEN, greenValue);
  analogWrite(PIN_BLUE, blueValue);
}

void loop() {
  // 1. 기본 색상 표현해 보기
  setColor(255, 0, 0);     // 빨간색 (Red 100%)
  delay(10000);
  
  setColor(0, 255, 0);     // 초록색 (Green 100%)
  delay(10000);
  
  setColor(0, 0, 255);     // 파란색 (Blue 100%)
  delay(10000);

  // 2. 색상 혼합해 보기 (빛의 삼원색 조합)
  setColor(255, 255, 0);   // 빨강 + 초록 = 노란색!
  delay(10000);

  setColor(80, 0, 80);     // 빨강 + 파랑 = 은은한 보라색!
  delay(10000);

  setColor(255, 255, 255); // 셋 다 100% 섞으면 흰색(White)!
  delay(10000);

  setColor(0, 0, 0);       // 모두 끄기
  delay(10000);

  // 3. 무지개처럼 자연스럽게 그라데이션하며 바뀌는 연출 (무한 루프의 묘미)
  // 빨강에서 초록으로 서서히 변하는 구간
  for (int i = 0; i < 255; i++) {
    setColor(255 - i, i, 0);
    delay(50); // 부드러운 전환을 위해 10ms씩 대기
  }
  // 초록에서 파랑으로 서서히 변하는 구간
  for (int i = 0; i < 255; i++) {
    setColor(0, 255 - i, i);
    delay(50);
  }
  // 파랑에서 다시 빨강으로 서서히 변하는 구간
  for (int i = 0; i < 255; i++) {
    setColor(i, 0, 255 - i);
    delay(50);
  }
}