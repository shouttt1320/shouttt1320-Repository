# -*- coding: utf-8 -*-
"""
모니터 화면의 특정 영역을 실시간으로 캡처하여 
리센느(RESCENE) 멤버를 분류/예측하는 스크립트

사용 방법:
  1. python 9_rescene_classification/5_realtime_screen_predict.py
  2. 실행 후 모니터에서 감지하고 싶은 영역의 좌표를 지정하거나, 
     OpenCV 화면 창에서 's' 키를 눌러 드래그하여 실시간 캡처 영역을 지정할 수 있습니다.
  3. 'q' 또는 ESC 키를 누르면 종료됩니다.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import cv2
from PIL import Image

# UTF-8 콘솔 출력 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import tensorflow as tf

# 한글 라벨 매핑 디렉터리
MEMBER_KOREAN_NAMES = {
    "woni": "원이 (WONI)",
    "livv": "리브 (LIVV)",
    "minami": "미나미 (MINAMI)",
    "may": "메이 (MAY)",
    "zena": "제나 (ZENA)"
}

# 맴버별 고유 테마 색상 (BGR 포맷)
MEMBER_COLORS = {
    "woni": (255, 150, 50),   # 주황/파랑
    "livv": (255, 100, 200),  # 분홍
    "minami": (50, 200, 255), # 하늘색
    "may": (100, 255, 100),   # 연두
    "zena": (200, 100, 255)   # 보라
}


def load_model_and_classes(model_path="models/rescene_model.keras", class_path="models/class_names.json"):
    """학습된 모델과 클래스 목록을 로드"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 모델 파일('{model_path}')을 찾을 수 없습니다. 먼저 3_train_rescene_cnn.py를 실행하세요.")
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"❌ 클래스 라벨 파일('{class_path}')을 찾을 수 없습니다.")

    print(f"📦 학습된 모델 로드 중: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    with open(class_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    return model, class_names


class ScreenCapturer:
    """mss, pyautogui, PIL ImageGrab을 지원하는 화면 캡처 클래스"""
    def __init__(self):
        self.backend = None
        try:
            import mss
            self.sct = mss.mss()
            self.backend = "mss"
        except Exception:
            try:
                import pyautogui
                self.backend = "pyautogui"
            except Exception:
                from PIL import ImageGrab
                self.backend = "pil"

    def grab(self, bbox):
        """
        bbox: (left, top, width, height)
        반환값: OpenCV BGR 포맷 numpy array
        """
        left, top, width, height = bbox
        
        if self.backend == "mss":
            monitor = {"left": left, "top": top, "width": width, "height": height}
            sct_img = self.sct.grab(monitor)
            # BGRA -> BGR
            img = np.array(sct_img)[:, :, :3]
            return img
        elif self.backend == "pyautogui":
            import pyautogui
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            # RGB -> BGR
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        else:
            from PIL import ImageGrab
            screenshot = ImageGrab.grab(bbox=(left, top, left + width, top + height))
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)


def run_realtime_prediction(bbox=(100, 100, 500, 500), model_path="models/rescene_model.keras", 
                            class_path="models/class_names.json", img_size=(128, 128)):
    """
    모니터 화면 특정 영역 실시간 추론 루프
    """
    model, class_names = load_model_and_classes(model_path, class_path)
    capturer = ScreenCapturer()

    window_name = "RESCENE Real-Time Screen Predictor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 640)

    print("=" * 65)
    print("🚀 실시간 모니터 화면 영역 예측을 시작합니다.")
    print("=" * 65)
    print(f"📌 현재 캡처 영역 (bbox): left={bbox[0]}, top={bbox[1]}, width={bbox[2]}, height={bbox[3]}")
    print("🎮 단축키 안내:")
    print("   - 's' 키: 화면에서 예측할 새 영역(ROI) 마우스로 직접 지정")
    print("   - 'q' 또는 ESC 키: 프로그램 종료")
    print("=" * 65)

    left, top, width, height = bbox
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0.0

    is_paused = False

    while True:
        if not is_paused:
            try:
                # 1. 모니터 영역 캡처 (BGR)
                frame_bgr = capturer.grab((left, top, width, height))
            except Exception as e:
                # 백엔드 오류 시 자동 fallback
                from PIL import ImageGrab
                screenshot = ImageGrab.grab(bbox=(left, top, left + width, top + height))
                frame_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            # 2. 이미지 전처리 (BGR -> RGB -> Resize -> Batch)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb).resize(img_size)
            img_array = np.array(img_pil)
            img_batch = np.expand_dims(img_array, axis=0)

            # 3. 모델 추론
            predictions = model.predict(img_batch, verbose=0)[0]
            top_idx = np.argmax(predictions)
            top_class = class_names[top_idx]
            top_prob = predictions[top_idx] * 100
            top_name = MEMBER_KOREAN_NAMES.get(top_class, top_class)

            # 4. FPS 계산
            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                fps_display = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()

            # 5. UI 오버레이 그리기
            display_frame = frame_bgr.copy()
            h, w, _ = display_frame.shape

            # 상단 오버레이 패널 (반투명 검은색 배경)
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
            # 하단 오버레이 패널 (확률 리스트)
            cv2.rectangle(overlay, (0, h - 130), (w, h), (10, 10, 10), -1)
            cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)

            # 메인 예측 결과 메세지
            theme_color = MEMBER_COLORS.get(top_class, (0, 255, 0))
            result_text = f"PREDICT: {top_class.upper()} ({top_prob:.1f}%)"
            cv2.putText(display_frame, result_text, (15, 40), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, theme_color, 2, cv2.LINE_AA)

            fps_text = f"FPS: {fps_display:.1f} | Area: {w}x{h}"
            cv2.putText(display_frame, fps_text, (15, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            # 하단 5명 멤버 확률 바 오버레이
            y_start = h - 110
            sorted_indices = np.argsort(predictions)[::-1]
            for i, idx in enumerate(sorted_indices):
                c_name = class_names[idx]
                prob = predictions[idx] * 100
                bar_w = int((w - 180) * (prob / 100.0))
                
                y_pos = y_start + (i * 22)
                # 라벨 텍스트
                cv2.putText(display_frame, f"{c_name:<8}", (15, y_pos + 14), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                # 확률 게이지 바
                c_color = theme_color if idx == top_idx else (100, 100, 100)
                cv2.rectangle(display_frame, (110, y_pos + 3), (110 + bar_w, y_pos + 15), c_color, -1)
                # 퍼센트 텍스트
                cv2.putText(display_frame, f"{prob:5.1f}%", (120 + bar_w, y_pos + 14), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display_frame)

        # 단축키 처리
        key = cv2.waitKey(30) & 0xFF
        if key in [27, ord('q')]:  # ESC 또는 q
            print("\n👋 실시간 감지를 종료합니다.")
            break
        elif key == ord('s'):  # s 키 누르면 캡처 영역 마우스 지정 (ROI Selection)
            print("\n🖱️ 모니터 전체 화면 캡처 후 영역 선택창을 엽니다. 마우스로 영역을 드래그하세요!")
            try:
                from PIL import ImageGrab
                full_screen = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2BGR)
                roi = cv2.selectROI("Select Screen Region to Track (Press ENTER to confirm)", full_screen, showCrosshair=True)
                cv2.destroyWindow("Select Screen Region to Track (Press ENTER to confirm)")
                
                if roi[2] > 20 and roi[3] > 20:  # 유효한 크기 설정
                    left, top, width, height = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
                    print(f"✅ 새 캡처 영역 설정 완료: left={left}, top={top}, width={width}, height={height}")
            except Exception as ex:
                print(f"⚠️ ROI 선택 중 오류 발생: {ex}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모니터 화면 실시간 리센느 멤버 분류기")
    parser.add_argument("--left", type=int, default=100, help="캡처 영역 X 좌표 (기본값: 100)")
    parser.add_argument("--top", type=int, default=100, help="캡처 영역 Y 좌표 (기본값: 100)")
    parser.add_argument("--width", type=int, default=500, help="캡처 영역 너비 (기본값: 500)")
    parser.add_argument("--height", type=int, default=500, help="캡처 영역 높이 (기본값: 500)")

    args = parser.parse_args()

    bbox = (args.left, args.top, args.width, args.height)
    run_realtime_prediction(bbox=bbox)
