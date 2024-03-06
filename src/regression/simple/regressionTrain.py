import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 임의의 이미지 특징과 연속적인 각도값 생성 (예시)
num_samples = 1000
image_features = torch.linspace(-4,4, num_samples).view(-1, 1)  # 임의의 이미지 특징
angles = 3 * image_features + torch.FloatTensor(num_samples, 1).normal_(30, 10)  # 연속적인 각도값

# 선형회귀 모델 정의
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 모델 초기화 및 옵티마이저 설정
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습 과정
num_epochs = 1000

for epoch in range(num_epochs):
    # 모델 예측
    predictions = model(image_features)

    # 손실 계산 및 역전파
    loss = criterion(predictions, angles)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 학습 과정 출력
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 테스트 데이터 생성 (예시)
test_image_features = torch.randn(10, 1)  # 테스트를 위한 이미지 특징
# 학습된 모델을 사용하여 각도값 예측
predicted_angles = model(test_image_features)

# 시각화
plt.scatter(image_features.numpy(), angles.numpy(), label='Original Data')
plt.plot(np.linspace(-4,4, 100),
         model(torch.linspace(-4,4, 100).view(-1, 1)).detach().numpy()
         , label='Predicted Line', color='red')
plt.xlabel('Image Features')
plt.ylabel('Angles')
plt.legend()
plt.show()