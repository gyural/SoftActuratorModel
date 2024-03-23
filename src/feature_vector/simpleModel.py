import torch
import torch.nn as nn
import torch.optim as optim
from parsingPoints import get_feature_points
from torch.utils.data import DataLoader, TensorDataset

# 임의의 데이터 생성
feature_points = get_feature_points()
X = torch.tensor(feature_points)  # 입력 데이터, 크기: [100, 33, 2]
Y = 3 * X.mean(dim=(1, 2)).unsqueeze(1) # 타겟 데이터, 크기: [100, 1]
# 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(66, 1)  # 입력 차원: 66, 출력 차원: 1
        self.sigmoid = nn.ReLU()
    def forward(self, x):
        x = x.view(66)  # 입력 데이터의 형태를 [batch_size, 66]으로 변경합니다.
        x = x.sigmoid()
        return self.linear(x)

# 모델 초기화
model = SimpleModel()

# 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 데이터셋 정의
dataset = TensorDataset(X, Y)

if __name__ == "__main__":
    # 훈련
    num_epochs = 10
    for epoch in range(num_epochs):
        # Forward pass
        for i in range(len(dataset)):  # 데이터셋의 길이만큼 순회
            inputs, targets = dataset[i]
            # print(inputs.size(), targets.size())
            outputs = model(inputs)  # 입력 데이터를 그대로 사용
            if (i == len(dataset) - 1):
                print(outputs, targets)
            loss = criterion(outputs, targets)
            # Backward pass 및 경사하강법
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 로그 출력
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')