import torch
import heapq
import torch.nn as nn
from torch.utils.data import TensorDataset
from simpleModel import SimpleModel
from parsingPoints import get_feature_points
import simpleModel_load

model_path = "C:\\Users\\Hilal\\pycharmProjects\\SoftActuratorModel\\src\\feature_vector\\feature_vector_model.pth"

# 데이터로더 생성
X = torch.tensor(get_feature_points())
Y = torch.tensor(simpleModel_load.get_IMG_bandingAngle()).unsqueeze(1)
dataset = TensorDataset(X, Y)

# 모델 클래스 초기화
model = SimpleModel()
device = "cpu"
model.load_state_dict(torch.load(model_path))

#모델을 평가(테스트) 모드로 설정합니다.
model.eval()
criterion = nn.MSELoss()

# with torch.no_grad():
#     outputs = model(X)
#     for a, b in zip(outputs, Y):
#         print(a.item(), b.item())  # 출력값 및 타겟값을 스칼라로 변환하여 출력
#         loss = criterion(a, b)
#         print(f'Loss: {loss.item():.4f}')

## result = [[loss_value, random_points_index], .....]
target_value = 15
result = []
heapq.heappush(result, (-111, 0))
cycle = 200

for _ in range(cycle):
    random_points = torch.tensor(simpleModel_load.get_random_points())
    with torch.no_grad():
        outputs = model(random_points)

    for idx, angle in enumerate(outputs):
        diff = abs(angle.item() - target_value)
        ### 힙의 요소와 비교
        head = heapq.heappop(result)
        if diff < head[0] * -1:
            heapq.heappush(result, (-1 * diff, idx))
        if len(result) < 10:
            heapq.heappush(result, head)

print(result)