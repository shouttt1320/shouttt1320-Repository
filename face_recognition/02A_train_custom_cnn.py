import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchsummary import summary
import numpy as np
import cv2

# 1. 커스텀 CNN 모델 정의 (MiniVGGNet 4블록 개조 버전)
class CustomFaceCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomFaceCNN, self).__init__()

        # Block 1 (Input: 3 x 224 x 224 -> Output: 32 x 112 x 112)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        # Block 2 (Output: 64 x 56 x 56)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )

        # Block 3 (Output: 128 x 28 x 28)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )

        # FC Layers (Flatten -> 128 * 28 * 28 = 100,352)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc_layers(x)
        return x

def check_and_create_unknown_class(data_path):
    """클래스가 1개(woni만 있음)일 경우, unknown 비교군 폴더와 샘플 이미지를 자동으로 생성"""
    if not os.path.exists(data_path):
        return

    subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    if len(subdirs) == 1:
        unknown_dir = os.path.join(data_path, 'unknown')
        os.makedirs(unknown_dir, exist_ok=True)
        print(f"\n[안내] 데이터셋에 '{subdirs[0]}' 1개 클래스만 존재하여, 'unknown' 비교 클래스 폴더를 자동 생성합니다...")

        existing_woni = os.path.join(data_path, subdirs[0])
        woni_files = [f for f in os.listdir(existing_woni) if f.endswith(('.jpg', '.png'))]
        num_unknown = max(50, len(woni_files))

        # 다양한 노이즈 및 랜덤 색상 패턴으로 unknown 샘플 생성
        for i in range(1, num_unknown + 1):
            img_path = os.path.join(unknown_dir, f"unknown_{i:04d}.jpg")
            if not os.path.exists(img_path):
                img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                cv2.imwrite(img_path, img)

        print(f" -> 'unknown' 클래스 샘플 {num_unknown}장 자동 생성 완료!\n")

def train_custom_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    DATA_PATH = "./face_dataset"
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 0.0005

    if not os.path.exists(DATA_PATH):
        print(f"에러: '{DATA_PATH}' 디렉토리가 없습니다. 01_collect_face_from_youtube.py를 먼저 실행해 주세요.")
        return

    # 1개 클래스만 있을 경우 unknown 클래스 자동 생성
    check_and_create_unknown_class(DATA_PATH)

    # 데이터 전처리 및 데이터 증강
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
    num_classes = len(dataset.classes)
    print(f"로드된 클래스 ({num_classes}개): {dataset.classes}")

    # 80% Train, 20% Val 데이터 분리
    if len(dataset) > 5:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    else:
        train_ds = val_ds = dataset

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CustomFaceCNN(num_classes=num_classes).to(device)
    summary(model, input_size=(3, 224, 224))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    print("\n[커스텀 CNN 모델 학습 시작]")
    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / max(total, 1)

        # Validation 평가
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / max(val_total, 1)

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # 모델 가중치 및 클래스 정보 저장
    save_data = {
        'state_dict': model.state_dict(),
        'classes': dataset.classes
    }
    torch.save(save_data, "custom_cnn_face.pth")
    print("\n모델 저장 완료: custom_cnn_face.pth")

    # 학습 그래프 시각화
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='red')
    plt.title('Custom CNN Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', color='blue')
    plt.plot(history['val_acc'], label='Val Acc', color='green')
    plt.title('Custom CNN Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train_custom_cnn()
