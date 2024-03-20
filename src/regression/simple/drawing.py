import torch
import torch.nn as nn
from regressionTrain import ImageRegressionModel
from torch.utils.data import DataLoader, TensorDataset
from IMG_load import IMG_loader, get_IMG_bandingAngle
model_path = "C:\\Users\\Hilal\\pycharmProjects\\SoftActuratorModel\\src\\regression\\simple\\image_regression_model.pth"

# 모델 클래스 초기화
model = ImageRegressionModel()
model.load_state_dict(torch.load(model_path))
#모델을 평가(테스트) 모드로 설정합니다.
model.eval()

# 출력 얻기
imgs = torch.stack(IMG_loader())
Y = torch.tensor(get_IMG_bandingAngle())
# 데이터로더 생성
dataset = TensorDataset(imgs, Y)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# for inputs, targets in data_loader:
#     print(inputs.size())
criterion = nn.MSELoss()
outputs = []
for inputs, targets in data_loader:
    outputs = model(inputs)

    for a, b in zip(outputs, targets):
        print(a, b)
loss = criterion(outputs, targets)
print(f'Loss: {loss.item():.4f}')
# np = output.detach().numpy()
#
# print('모델 예측 결과:', outputs)