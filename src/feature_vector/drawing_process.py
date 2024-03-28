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
model_path = f"C:\\Users\\{lab_env}\\pycharmProjects\\SoftActuratorModel\\src\\feature_vector\\feature_vector_model.pth"
# model_path = f"C:\\Users\\{lab_env}\\pycharmProjects\\SoftActuratorModel\\src\\feature_vector\\feature_vector_model_labtop.pth"

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
    cycle = 200

    for iter in range(1, cycle+1):
        random_points_outer, random_points_inner = simpleModel_load.get_random_points()

        with torch.no_grad():
            outputs = model(torch.tensor(random_points_outer))

        for idx, angle in enumerate(outputs):
            diff = abs(angle.item() - target_value)
            ### 힙의 요소와 비교
            head = heapq.heappop(result)
            if diff < head[0] * -1:
                heapq.heappush(result, (-1 * diff, random_points_outer[idx], random_points_inner[idx]))
            if len(result) < 10:
                heapq.heappush(result, head)

        #testing 중간확인
        if(iter % 10 == 0):
            sum_l = 0
            for l, idx, _ in result:
                sum_l += l

            print(f"[{iter}/{cycle}] complete")
            print(f"avg loss: {sum_l/len(result)}")

    return result

def points_drawing(points, target_angle):
    outer_numpy = np.array([outer for _, outer, inner in points])
    inner_list = [inner for _, outer, inner in points]

    #Drawing
    for idx, point_out, point_in in enumerate(zip(outer_numpy, inner_list)):
        # output 좌표 추출
        x_out = point_out[:, 0] + point_out[:1, 0]
        y_out = point_out[:, 1] + point_out[:1, 1]
        # 처음 좌표를 마지막에 추가
        x_out = np.append(x_out, point_out[0][0])
        y_out = np.append(y_out, point_out[0][1])

        # input 좌표 추출
        x_in = point_out[:, 0] + point_in[:1, 0]
        y_in = point_out[:, 1] + point_in[:1, 1]
        # 처음 좌표를 마지막에 추가
        x_in= np.array(x_in)
        y_in= np.array(y_in)
        x_in = np.append(x_in, point_in[0][0])
        y_in = np.append(y_in, point_in[0][1])

        # 그림 그리기
        plt.figure(figsize=(6, 4))
        plt.plot(x_out, y_out, marker='o', linestyle='-')
        plt.plot(x_in, y_in, marker='o', linestyle='-')
        plt.title(f"{target_angle} degrees Actuator design")
        plt.grid(True)

        # 축과 눈금 숨기기
        plt.xticks([])  # x 축 숨기기
        plt.yticks([])  # y 축 숨기기

        # 축 라인 숨기기
        plt.gca().spines['top'].set_visible(False)  # 상단 축 숨기기
        plt.gca().spines['right'].set_visible(False)  # 우측 축 숨기기
        plt.gca().spines['left'].set_visible(False)  # 좌측 축 숨기기
        plt.gca().spines['bottom'].set_visible(False)  # 하단 축 숨기기

        # 이미지 파일로 저장
        lab_user = "Hilal"
        my_user = "dgw04"
        save_path = f"C:\\Users\\{lab_env}\\pycharmProjects\\SoftActuratorModel\\datas\\outputs\\{target_angle}_{idx}.png"
        plt.savefig(save_path)
        # 현재 활성화된 그래프 창을 지웁니다.
        plt.clf()

#####main#####
if __name__ == "__main__":

    # ### candi 특징점 추출
    target_angle = 20
    candi = get_Top10_design(target_angle)
    # # candi 출력해보기
    # for loss, outer, inner in candi:
    #     print(f"loss: {loss} points-outer: {outer[0]} points-inner: {inner[0]}")

    ### numpy array로 바꾸기
    points_drawing(candi, target_angle)
    ### model test
    # model_test()
