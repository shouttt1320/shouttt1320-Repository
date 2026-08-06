# -*- coding: utf-8 -*-
"""
수집한 리센느 데이터셋을 Keras / TensorFlow로 불러오는 예제 스크립트
"""

import os
import sys

# UTF-8 출력 설정 (윈도우 콘솔 한글 및 이모지 깨짐 방지)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import tensorflow as tf
from tensorflow.keras import layers, models

DATASET_DIR = "dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

def load_dataset_tf():
    """TensorFlow / Keras image_dataset_from_directory 사용 예제"""
    if not os.path.exists(DATASET_DIR):
        print(f"❌ '{DATASET_DIR}' 디렉터리가 없습니다. 먼저 1_crawl_rescene_images.py를 실행하세요.")
        return

    print("=" * 60)
    print("📦 TensorFlow / Keras 데이터셋 불러오기")
    print("=" * 60)
    
    # 1. 학습 데이터셋 (80%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # 2. 검증 데이터셋 (20%)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"\n🏷️ 클래스 목록 (라벨): {class_names}")
    
    # 데이터 성능 최적화
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("✅ 데이터셋 로드 완료!")
    return train_ds, val_ds, class_names

if __name__ == "__main__":
    load_dataset_tf()
