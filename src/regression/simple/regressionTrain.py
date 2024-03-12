import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 임의의 이미지 특징과 연속적인 각도값 생성 (예시)
# 예제 데이터 생성 (이미지, K값)
X_image = torch.randn(100, 1, 255, 255)  # 100 samples, 1 channel, 127x127 size
k_values = 50 * torch.rand(100, 1)  # K값, 0에서 50 사이의 랜덤 값 사용

# 선형회귀 모델 정의
# 모델 정의
class KRegressionModel(nn.Module):
    def __init__(self):
        super(KRegressionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 31 * 31, 256),  # Adjust the input size based on your architecture
            nn.ReLU(),
            nn.Linear(256, 1),  # Output 1 value for K
            nn.Sigmoid()  # Sigmoid 함수를 이용하여 0에서 1 사이의 값으로 변환
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x * 50  # 0에서 1 사이의 값을 0에서 50 사이로 변환


# 모델, 손실 함수, 최적화 함수 초기화
model = KRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# DataLoader를 사용하여 배치 처리
dataset = TensorDataset(X_image, k_values)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 회귀 모델 학습
epochs = 10
for epoch in range(epochs):
    for inputs, k_values in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, k_values)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


