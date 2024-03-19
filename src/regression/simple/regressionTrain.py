
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from IMG_load import IMG_loader, get_IMG_bandingAngle

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
        x = x * 30
        return x

if __name__ == "__main__":
    # 데이터셋 생성
    X_images_tensor = torch.stack(IMG_loader())
    Y_values_tensor = torch.tensor(get_IMG_bandingAngle())

    dataset = TensorDataset(X_images_tensor, Y_values_tensor)

    # 데이터로더 생성
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 모델 생성 및 손실 함수, 옵티마이저 정의
    model = ImageRegressionModel(pretrained=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 훈련 진행
    num_epochs = 100
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # print("---------------------------------------")
            # for a, b in zip(outputs, targets):
            #     print(a, b)
            # print("---------------------------------------")
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    # 모델 저장하기
    torch.save(model.state_dict(), 'image_regression_model.pth')