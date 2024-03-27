
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
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.ELU = nn.ELU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.resnet(x)
        x = self.leaky_relu(x)
        return x

if __name__ == "__main__":
    # 데이터셋 생성
    X_images_tensor = torch.stack(IMG_loader())
    Y_values_tensor = torch.tensor(get_IMG_bandingAngle())

    dataset = TensorDataset(X_images_tensor, Y_values_tensor)

    # 데이터로더 생성
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 모델 생성 및 손실 함수, 옵티마이저 정의
    model = ImageRegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-7)

    # 훈련 진행
    loss_ = []
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0.0  # 전체 손실 초기화
        for inputs, targets in data_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # 각 배치의 손실을 누적
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')
    # 모델 저장하기
    # torch.save(model.state_dict(), 'image_regression_model.pth')


    print("##############학습 완료############")
    #### 저장 이후 출력 테스팅
    model.eval()
    for inputs, targets in data_loader:
        outputs = model(inputs)
        for a, b in zip(outputs, targets):
            print(a.item(), b.item())  # 출력값 및 타겟값을 스칼라로 변환하여 출력
        loss = criterion(outputs, targets)
        print(f'Loss: {loss.item():.4f}')