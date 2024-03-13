import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from IMGseperating import img_sep_main
from torchvision import models, transforms
from PIL import Image

# 임의의 이미지 특징과 연속적인 각도값 생성 (예시)
# 예제 데이터 생성 (이미지, K값)
# X_image = torch.randn(100, 600, 800)  # 100 samples, 1 channel, 800x600 size
# Y_values = img_sep_main(np.array(X_image))  # K값, 0에서 50 사이의 랜덤 값 사용
# Y_values = torch.tensor(Y_values)
# 선형회귀 모델 정의
# 이미지 특성 예측을 위한 이미지 회귀 모델 정의
# 이미지 크기를 224x224로 조정하는 전처리 단계 추가
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 이미지 특성 예측을 위한 이미지 회귀 모델 정의
class ImageRegressionModel(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageRegressionModel, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        return self.resnet(x)

# 예제 데이터 생성
X_image = []
for _ in range(100):
    # 임의의 이미지 생성
    image = Image.fromarray(np.uint8(np.random.rand(600, 800, 3) * 255))
    # 이미지 전처리
    image = preprocess(image)
    X_image.append(image)

# 이미지를 텐서로 변환
X_image = torch.stack(X_image)
Y_values = torch.randn(100, 1)  # 예측할 이미지 특성값

# 데이터셋 및 데이터로더 생성
dataset = TensorDataset(X_image, Y_values)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 생성 및 손실 함수, 옵티마이저 정의
model = ImageRegressionModel(pretrained=True)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 훈련 진행
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

