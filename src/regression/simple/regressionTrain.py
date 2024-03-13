import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)  # 시그모이드 함수를 이용해 0부터 1 사이의 값으로 변환
        x = x * 70  # 값의 범위를 0~70으로 스케일링
        return x

# 예제 데이터 생성
X_image = []
Y_values = []
for _ in range(100):
    # 임의의 이미지 생성
    image = Image.fromarray(np.uint8(np.random.rand(600, 800, 3) * 255))
    # 이미지 전처리
    image = preprocess(image)
    X_image.append(image)
    # 랜덤한 각도 값 생성 (0부터 70까지)
    angle_value = torch.rand(1) * 70
    Y_values.append(angle_value)

# 이미지를 텐서로 변환
X_image = torch.stack(X_image)
Y_values = torch.stack(Y_values)

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

# 모델 저장하기
torch.save(model.state_dict(), 'image_regression_model.pth')