import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler # GPU 가속 및 메모리 절약용
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# 1. 장치 설정 (GPU 최우선, 없으면 CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 장치: {DEVICE}")

# 2. 데이터셋 및 증강 (GPU 연산을 위해 배치 사이즈를 4로 상향)
BATCH_SIZE = 4 
EPOCHS = 50 
LR = 0.0001

transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(rotate_limit=20, p=0.5),
    A.Normalize(),
    ToTensorV2(),
])

# [Dataset 클래스는 이전과 동일하므로 생략하거나 기존 것 사용]
# dataset = ToolWearDataset(image_dir='./images', mask_dir='./masks', transform=transform)
# train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. 모델 정의 (MobileNet_V2는 GPU에서도 매우 효율적입니다)
model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation='sigmoid'
).to(DEVICE)

# 4. 손실함수 및 최적화
criterion = smp.losses.DiceLoss(mode='binary')
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = GradScaler() # AMP를 위한 스케일러

# 5. 실제 GPU 학습 루프
def train_gpu(dataloader):
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for images, masks in dataloader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            
            # AMP 적용: 연산 속도 향상 및 메모리 절약
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # 가중치 업데이트
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "tool_wear_gpu_model.pth")
    print("GPU 학습 및 모델 저장 완료")