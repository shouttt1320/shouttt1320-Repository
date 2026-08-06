# -*- coding: utf-8 -*-
"""
학습된 CNN 모델을 이용하여 신규 이미지가 리센느(RESCENE)의 어떤 멤버인지 예측하고 화면에 띄워주는 스크립트

사용법:
  python 9_rescene_classification/4_predict_rescene_member.py --image dataset/woni/woni_0003.jpg
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# 윈도우 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# UTF-8 콘솔 출력 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import tensorflow as tf

# 한글 라벨 매핑 디렉터리
MEMBER_KOREAN_NAMES = {
    "woni": "원희 (WONI)",
    "livv": "리브 (LIVV)",
    "minami": "미나미 (MINAMI)",
    "may": "메이 (MAY)",
    "zena": "제나 (ZENA)"
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


def predict_member(image_path, model_path="models/rescene_model.keras", class_path="models/class_names.json", 
                   img_size=(128, 128), show_window=True, use_cv2=False):
    """
    단일 이미지에 대해 어떤 멤버인지 예측하고 화면에 시각화 창을 띄움
    """
    if not os.path.exists(image_path):
        print(f"❌ 대상 이미지 파일('{image_path}')이 존재하지 않습니다.")
        return

    model, class_names = load_model_and_classes(model_path, class_path)

    # 이미지 전처리
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize(img_size)
    img_array = np.array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 3)

    # 모델 예측
    predictions = model.predict(img_batch, verbose=0)[0]
    top_index = np.argmax(predictions)
    top_class = class_names[top_index]
    top_confidence = predictions[top_index] * 100
    top_korean_name = MEMBER_KOREAN_NAMES.get(top_class, top_class)

    # 결과 콘솔 출력
    print("\n" + "=" * 60)
    print(f"🔍 [이미지 예측 결과] 파일명: {os.path.basename(image_path)}")
    print("=" * 60)
    print(f"🏆 1위 예측 결과: 【 {top_korean_name} 】  (신뢰도: {top_confidence:.2f}%)")
    print("-" * 60)
    print("📊 멤버별 확신도 확률 분포:")

    # 확률 내림차순 정렬
    sorted_indices = np.argsort(predictions)[::-1]
    for idx in sorted_indices:
        c_name = class_names[idx]
        k_name = MEMBER_KOREAN_NAMES.get(c_name, c_name)
        prob = predictions[idx] * 100
        bar_len = int(prob // 5)  # 5%당 한 블록
        bar_str = "█" * bar_len
        print(f"  - {k_name:<15} : {prob:>6.2f}%  [{bar_str:<20}]")

    print("=" * 60)

    # 1) OpenCV 창으로 표시 옵션
    if use_cv2:
        cv_img = cv2.imread(image_path)
        label_text = f"{top_korean_name} ({top_confidence:.1f}%)"
        cv2.putText(cv_img, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        window_title = f"RESCENE Prediction - {os.path.basename(image_path)}"
        print(f"📺 OpenCV 화면 창 '{window_title}'을 출력합니다. (아무 키나 누르면 닫힙니다.)")
        cv2.imshow(window_title, cv_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # 2) Matplotlib 화면 창 출력 (이미지 + 멤버별 확률 차트)
        visualize_prediction(img, class_names, predictions, top_korean_name, top_confidence, show=show_window)


def visualize_prediction(img, class_names, predictions, top_korean_name, top_confidence, 
                         save_path="models/prediction_result.png", show=True):
    """예측 이미지와 멤버별 확률 그래프를 화면에 팝업창으로 띄우고 저장"""
    fig = plt.figure(figsize=(10, 5))

    # 1. 원본 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"예측 결과: {top_korean_name}\n(신뢰도: {top_confidence:.1f}%)", fontsize=14, fontweight="bold", color="darkblue")
    plt.axis("off")

    # 2. 확률 분포 바 차트
    plt.subplot(1, 2, 2)
    y_labels = [MEMBER_KOREAN_NAMES.get(c, c) for c in class_names]
    y_pos = np.arange(len(y_labels))
    probs = predictions * 100

    bars = plt.barh(y_pos, probs, color="skyblue", edgecolor="blue")
    
    # Highest probability bar highlight
    max_idx = np.argmax(predictions)
    bars[max_idx].set_color("dodgerblue")
    bars[max_idx].set_edgecolor("darkblue")

    plt.yticks(y_pos, y_labels, fontsize=11)
    plt.xlabel("신뢰도 확신율 (%)", fontsize=11)
    plt.title("멤버별 예측 확률 분포", fontsize=12, fontweight="bold")
    plt.xlim(0, 100)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # 퍼센트 텍스트 표시
    for i, p in enumerate(probs):
        plt.text(p + 1.5, i, f"{p:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"🖼️ 시각화 결과 이미지가 '{save_path}'에 저장되었습니다.")

    if show:
        print("📺 화면에 예측 결과창을 띄웠습니다. (창을 닫으시면 스크립트가 종료됩니다.)\n")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="리센느 멤버 이미지 구분(인퍼런스)")
    parser.add_argument("--image", type=str, required=False, help="예측할 이미지 파일 경로")
    parser.add_argument("--no_show", action="store_true", help="화면 창 팝업 끄기")
    parser.add_argument("--cv2", action="store_true", help="Matplotlib 대신 OpenCV 창(cv2.imshow)으로 띄우기")
    
    args = parser.parse_args()

    target_image = args.image
    if not target_image:
        print("💡 '--image' 경로가 입력되지 않아 데이터셋 내 샘플 이미지를 테스트합니다.")
        sample_found = None
        for root, dirs, files in os.walk("dataset"):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    sample_found = os.path.join(root, file)
                    break
            if sample_found:
                break
        
        target_image = sample_found

    if target_image:
        predict_member(
            target_image,
            show_window=not args.no_show,
            use_cv2=args.cv2
        )
    else:
        print("❌ 테스트할 이미지가 없습니다. 먼저 1_crawl_rescene_images.py로 이미지를 다운로드하세요.")
