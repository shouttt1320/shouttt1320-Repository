# -*- coding: utf-8 -*-
"""
리센느(RESCENE) 멤버 이미지 분류 CNN 모델 학습 스크립트

핵심 특징:
  1. Data Augmentation (데이터 증강):
     소량 데이터 환경에서의 오버피팅(Overfitting)을 방지하기 위해 
     RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomContrast 적용
  2. CNN 모델 구조:
     - Custom CNN (Conv2D + BatchNorm + MaxPool + Dropout)
     - Transfer Learning (MobileNetV2 기반 사전학습 모델 전이학습) 선택 가능
  3. 학습 결과 시각화 및 모델/클래스 라벨 저장
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt

# UTF-8 콘솔 출력 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# GPU 메모리 동적 할당 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU 메모리 동적 할당 설정 완료")
    except RuntimeError as e:
        print(e)


def create_data_augmentation():
    """
    소량의 이미지 데이터를 위한 Data Augmentation 파이프라인 생성
    (이미지 반전, 회전, 확대/축소, 위치 이동, 대비 조절)
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomContrast(0.15),
    ], name="data_augmentation")


def build_custom_cnn(input_shape, num_classes):
    """
    Custom CNN 모델 아키텍처 구축
    Conv2D + BatchNormalization + MaxPooling2D + Dropout
    """
    data_augmentation = create_data_augmentation()

    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)  # [0, 255] -> [0, 1] 정규화

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Fully Connected Classification Head
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="RESCENE_Custom_CNN")
    return model


def build_transfer_learning_model(input_shape, num_classes):
    """
    MobileNetV2 기반 Transfer Learning (전이학습) 모델 구축
    소량의 이미지 데이터셋에서 최고의 성능을 발휘함
    """
    data_augmentation = create_data_augmentation()

    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    
    # MobileNetV2 전처리 [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # ImageNet으로 사전 학습된 MobileNetV2 로드
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # 사전학습 가중치 동결

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="RESCENE_MobileNetV2_Transfer")
    return model


def train_model(dataset_dir="dataset", img_size=(128, 128), batch_size=16, epochs=30, model_type="custom"):
    """
    데이터셋 로드, 모델 생성, Data Augmentation 적용 및 학습 진행
    """
    if not os.path.exists(dataset_dir):
        print(f"❌ 데이터셋 디렉터리 '{dataset_dir}'가 없습니다. 먼저 1_crawl_rescene_images.py를 실행하세요.")
        return

    print("=" * 60)
    print(f"🚀 리센느 멤버 분류 모델 학습 시작 (방식: {model_type.upper()}, Epochs: {epochs})")
    print("=" * 60)

    # 1. 데이터셋 분할 (80% Train, 20% Validation)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"\n🏷️ 멤버 클래스 라벨 ({num_classes}명): {class_names}")

    # 클래스 이름을 JSON으로 저장 (추후 인퍼런스/예측 시 사용)
    os.makedirs("models", exist_ok=True)
    with open("models/class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 2. 모델 선택 및 구축
    input_shape = (img_size[0], img_size[1], 3)
    if model_type.lower() == "transfer":
        model = build_transfer_learning_model(input_shape, num_classes)
    else:
        model = build_custom_cnn(input_shape, num_classes)

    model.summary()

    # 3. 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4. 콜백 설정 (Overfitting 방지 & 학습률 조정)
    model_save_path = "models/rescene_model.keras"
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        callbacks.ModelCheckpoint(model_save_path, monitor="val_accuracy", save_best_only=True)
    ]

    # 5. 모델 학습
    print("\n🏋️ 모델 학습을 시작합니다...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb_list
    )

    print(f"\n✅ 모델 학습 완료! 최고의 모델이 '{model_save_path}'에 저장되었습니다.")

    # 6. 학습 시각화 그래프 저장
    plot_training_history(history)
    return model, class_names


def plot_training_history(history, save_path="models/training_history.png"):
    """학습 및 검증 손실/정확도 그래프 저장"""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b-o', label="Training Accuracy")
    plt.plot(epochs_range, val_acc, 'r-o', label="Validation Accuracy")
    plt.title("Training & Validation Accuracy (Data Augmentation)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'b-o', label="Training Loss")
    plt.plot(epochs_range, val_loss, 'r-o', label="Validation Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"📊 학습 결과 시각화 그래프가 '{save_path}'에 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="리센느 멤버 이미지 분류 CNN 학습")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="데이터셋 폴더")
    parser.add_argument("--epochs", type=int, default=30, help="학습 에포크 수 (기본값: 30)")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기 (기본값: 16)")
    parser.add_argument("--model_type", type=str, default="transfer", choices=["custom", "transfer"], 
                        help="모델 종류 ('custom': CNN 직접 구축, 'transfer': MobileNetV2 전이학습)")

    args = parser.parse_args()

    train_model(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_type=args.model_type
    )
