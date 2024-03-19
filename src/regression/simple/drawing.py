import torch
from regressionTrain import ImageRegressionModel
from IMG_load import IMG_loader
model_path = "C:\\Users\\Hilal\\pycharmProjects\\SoftActuratorModel\\src\\regression\\simple\\image_regression_model.pth"

# 모델 클래스 초기화
model = ImageRegressionModel()
model.load_state_dict(torch.load(model_path))

#모델을 평가(테스트) 모드로 설정합니다.
model.eval()

# 출력 얻기
imgs = torch.stack(IMG_loader())
with torch.no_grad():
    inputs = imgs
    output = model(inputs)

np = output.detach().numpy()

print('모델 예측 결과:', np)