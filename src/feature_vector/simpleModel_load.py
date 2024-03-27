import torch
import torch.nn as nn
# from simpleModel import SimpleModel
from torch.utils.data import TensorDataset
from parsingPoints import get_feature_points
import os
import re

model_path = "C:\\Users\\dgw04\\pycharmProjects\\SoftActuratorModel\\src\\feature_vector\\feature_vector_model.pth"

#이미지 파일명을 통해 bandingangle값 float리스트로 반환
def get_IMG_bandingAngle():
    image_path = "C:\\Users\\dgw04\\PycharmProjects\\SoftActuratorModel\\datas\\afterIMG"
    Y_values = []
    image_filenames = os.listdir(image_path)

    for filename in image_filenames:
        float_number = float(re.findall(r'\d+\.\d+', filename)[0])
        Y_values.append(float_number)
    return Y_values[:100]



if __name__ == "__main__":

    # 모델 클래스 초기화
    model = SimpleModel()
    model.load_state_dict(torch.load(model_path), strict=False)

    #모델을 평가(테스트) 모드로 설정합니다.
    model.eval()

    # 가상 데이터 로드
    feature_points = get_feature_points()
    X = torch.tensor(feature_points)  # 입력 데이터, 크기: [100, 33, 2]
    # Y = 3 * X.mean(dim=(1, 2)).unsqueeze(1) # 타겟 데이터, 크기: [100, 1] torch.Size([100,1])
    Y = torch.tensor(get_IMG_bandingAngle()).unsqueeze(1)


    # 데이터셋 정의
    dataset = TensorDataset(X, Y)
    criterion = nn.MSELoss()
    outputs = []
    with torch.no_grad():
        for inputs, targets in dataset:
            outputs = model(inputs)
            for a, b in zip(outputs, targets):
                print(a, b)
            loss = criterion(outputs, targets)
            print(f'Loss: {loss.item():.4f}')
