from regressionTrain import ImageRegressionModel, preprocess
import torch

#### 1. 랜덤한 이미지 생성 ####

#### 2. 이미지 데이터 선형 모델에 대입 ####
# 모델 생성 및 불러오기
model = ImageRegressionModel(pretrained=True)
model.load_state_dict(torch.load('image_regression_model.pth'))
model.eval()

# 이미지를 모델에 입력하여 예측 수행
with torch.no_grad():
    output = model(preprocess(image))

print('모델 예측 결과:', output.item())

