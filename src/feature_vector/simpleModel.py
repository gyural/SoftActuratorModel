import torch
import torch.nn as nn
import torch.optim as optim

# 임의의 데이터 생성
X = torch.randn(100, 33)  # 입력 데이터, 크기: [100, 33]
Y = 3 * X.mean(dim=1, keepdim=True)   # 타겟 데이터, 크기: [100, 1]

# 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(33, 1)  # 입력 차원: 33, 출력 차원: 1

    def forward(self, x):
        return self.linear(x)

# 모델 초기화
model = SimpleModel()

# 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 훈련
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, Y)
    if epoch == num_epochs - 1:
        print("---------------------------------------")
        for a, b in zip(outputs, Y):
            print(a, b)
        print("---------------------------------------")
    # Backward pass 및 경사하강법
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 로그 출력
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
