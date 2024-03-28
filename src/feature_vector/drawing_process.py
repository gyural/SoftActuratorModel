import torch
import heapq
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.utils.data import TensorDataset
from simpleModel import SimpleModel
from parsingPoints import get_feature_points
import simpleModel_load

my_env = "dgw04"
lab_env = "Hilal"
# model_path = f"C:\\Users\\{my_env}\\pycharmProjects\\SoftActuratorModel\\src\\feature_vector\\feature_vector_model.pth"
model_path = f"C:\\Users\\{my_env}\\pycharmProjects\\SoftActuratorModel\\src\\feature_vector\\feature_vector_model_labtop.pth"

# 데이터로더 생성
X = torch.tensor(get_feature_points())
Y = torch.tensor(simpleModel_load.get_IMG_bandingAngle()).unsqueeze(1)
dataset = TensorDataset(X, Y)

# 불러오기
device = torch.device("cpu")
model = SimpleModel()
# 사용하고자 하는 GPU 장치 번호를 지정합니다.
model.load_state_dict(torch.load(model_path, map_location="cpu"))
# 모델에 사용되는 모든 입력 Tensor들에 대해 input = input.to(device) 을 호출해야 합니다.
X = X.to(device)
model.to(device)

#모델을 평가(테스트) 모드로 설정합니다.
model.eval()
criterion = nn.MSELoss()

# 학습된 모델을 테스팅 하는 코드
def model_test():
    with torch.no_grad():
        outputs = model(X)
        for a, b in zip(outputs, Y):
            print(a.item(), b.item())  # 출력값 및 타겟값을 스칼라로 변환하여 출력
            loss = criterion(a, b)
            print(f'Loss: {loss.item():.4f}')
        # result = [[loss_value, random_points_index], .....]

def get_Top10_design(target_value):
    result = []
    heapq.heappush(result, (-111, 0))
    cycle = 20

    for iter in range(1, cycle+1):
        random_points = torch.tensor(simpleModel_load.get_random_points())
        with torch.no_grad():
            outputs = model(random_points)

        for idx, angle in enumerate(outputs):
            diff = abs(angle.item() - target_value)
            ### 힙의 요소와 비교
            head = heapq.heappop(result)
            if diff < head[0] * -1:
                heapq.heappush(result, (-1 * diff, random_points[idx]))
            if len(result) < 10:
                heapq.heappush(result, head)

        #testing 중간확인
        if(iter % 1 == 0):
            sum_l = 0
            for l, idx in result:
                sum_l += l

            print(f"[{iter}/{cycle}] complete")
            print(f"avg loss: {sum_l/len(result)}")

    return result

def points_drawing(points):
    candi_numpy = np.array([points for _, points in candi])

    # 좌표 추출
    x = candi_numpy[0][:, 0]
    y = candi_numpy[0][:, 1]

    # 그림 그리기
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title('Connected Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

    # 이미지 파일로 저장
    # plt.savefig('connected_points.png')
# ### candi 특징점 추출
candi = get_Top10_design(15)
# candi 출력해보기
# for loss, points in candi:
#     print(f"loss: {loss} points-shape: {np.shape(points)}, ")

### numpy array로 바꾸기
points_drawing(candi)

### model test
# model_test()
