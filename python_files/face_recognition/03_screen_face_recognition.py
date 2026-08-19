import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
from mss import mss
from PIL import Image

# 1. 커스텀 CNN 모델 정의
class CustomFaceCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomFaceCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2), nn.Dropout(0.3)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2), nn.Dropout(0.4)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc_layers(self.block3(self.block2(self.block1(x))))

def run_screen_face_recognition():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    resnet_path = "resnet18_face.pth"
    custom_path = "custom_cnn_face.pth"

    print("\n" + "=" * 55)
    print("      [3단계] 모니터 화면 실시간 얼굴 인식 - 모델 선택")
    print("=" * 55)
    print("  [1] ResNet18 전이학습 모델 (resnet18_face.pth)")
    print("  [2] Custom CNN 커스텀 모델 (custom_cnn_face.pth)")

    choice = input("\n사용할 모델 번호를 선택하세요 (1 또는 2 입력, 엔터 누르면 1번): ").strip()

    if choice == "2":
        selected_path = custom_path
        model_type = "CustomCNN"
    else:
        selected_path = resnet_path
        model_type = "ResNet18"

    if not os.path.exists(selected_path):
        print(f"\n⚠️ 선택한 가중치 파일 '{selected_path}'이(가) 없습니다!")
        print(" -> 먼저 02A(커스텀 CNN) 또는 02B(ResNet18) 학습을 실행하여 가중치 파일을 만들어주세요.")
        return

    checkpoint = torch.load(selected_path, map_location=device)
    classes = checkpoint['classes']

    if model_type == "ResNet18":
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(checkpoint['state_dict'])
        print(f"\n✅ ResNet18 전이학습 모델 성공적으로 로드 완료! (클래스: {classes})")
    else:
        model = CustomFaceCNN(num_classes=len(classes))
        model.load_state_dict(checkpoint['state_dict'])
        print(f"\n✅ Custom CNN 커스텀 모델 성공적으로 로드 완료! (클래스: {classes})")

    model = model.to(device)
    model.eval()


    # 이미지 전처리 파이프라인
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # OpenCV Haar Cascade 얼굴 탐지기
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 확신도 임계값 (60% 미만이면 unknown 처리)
    CONFIDENCE_THRESHOLD = 0.60

    print("\n[모니터 화면 실시간 얼굴 인식 실행 중...]")
    print(" - 화면 속 재생 영상에서 얼굴을 감지하여 실시간 추론합니다.")
    print(" - 'q' 키: 프로그램 종료\n")

    sct = mss()
    # 주 모니터 영역 (1번 모니터 전체)
    monitor = sct.monitors[1]

    while True:
        # 모니터 화면 실시간 스크린샷 캡처 (BGRA)
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)

        # BGRA -> BGR 변환
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 화면 속 얼굴 위치 탐지
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            # 여유 공간 확보 크롭
            padding = int(w * 0.1)
            y1 = max(0, y - padding)
            y2 = min(frame_bgr.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(frame_bgr.shape[1], x + w + padding)

            face_roi = frame_bgr[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            # BGR -> RGB 변환 후 PyTorch 전처리
            rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            input_tensor = transform(rgb_roi).unsqueeze(0).to(device)

            # 추론
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)

                top_conf = conf.item()
                top_class_idx = pred.item()

            # Unknown 판단 필터링 (확률 60% 미만이면 unknown)
            if top_conf >= CONFIDENCE_THRESHOLD:
                label_name = classes[top_class_idx]
                color = (0, 255, 0) # 초록색 (인식 성공)
            else:
                label_name = "unknown"
                color = (0, 0, 255) # 빨간색 (미인식/Unknown)

            display_text = f"{label_name} ({top_conf * 100:.1f}%)"

            # 화면 박스 및 라벨 표시
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 3)
            cv2.putText(frame_bgr, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 화면에 표시하기 위한 미리보기 (해상도 반으로 축소해서 표출)
        display_preview = cv2.resize(frame_bgr, (960, 540))
        cv2.putText(display_preview, f"Model: {model_type} | Press 'q' to quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Screen Face Recognition", display_preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_screen_face_recognition()
