import torch
import torch.nn as nn
import torch.nn.functional as F                       # torch 내의 세부적인 기능을 불러옴.
import torch.optim as optim
from parsingPoints import get_feature_points
from torch.utils.data import TensorDataset, DataLoader
import simpleModel_load

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()  # 모델 연산 정의
        self.fc1 = nn.Linear(66, 50, bias=True)  # 입력층(66) -> 은닉층1(50)으로 가는 연산
        self.fc2 = nn.Linear(50, 30, bias=True)  # 은닉층1(50) -> 은닉층2(30)으로 가는 연산
        self.fc6 = nn.Linear(30, 1, bias=True)  # 출력층(30) -> 출력층(1)으로 가는 연산
        self.dropout = nn.Dropout(0.2)  # 연산이 될 때마다 20%의 비율로 랜덤하게 노드를 없앤다.

    def forward(self, x):
        x = x.view(-1, 66)

        x = F.relu(self.fc1(x))  # Linear 계산 후 활성화 함수 ReLU를 적용한다.
        x = self.dropout(F.relu(self.fc2(x)))  # 은닉층2에서 드랍아웃을 적용한다.(즉, 30개의 20%인 6개의 노드가 계산에서 제외된다.)
        x = F.relu(self.fc6(x))  # Linear 계산 후 활성화 함수 ReLU를 적용한다.
        return x

if __name__ == "__main__":
    # 임의의 데이터 생성
    feature_points = get_feature_points()
    X = torch.tensor(feature_points)  # 입력 데이터, 크기: [100, 33, 2]
    # Y = 3 * X.mean(dim=(1, 2)).unsqueeze(1)  # 입력 데이터와 동일한 형태로 타겟 데이터 생성
    Y = torch.tensor(simpleModel_load.get_IMG_bandingAngle()).unsqueeze(1)
    # 모델 초기화
    model = SimpleModel()
    # 손실 함수와 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)
    # 데이터셋 및 데이터로더 정의
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 훈련
    loss_ = []
    num_epochs = 8000
    for epoch in range(1, num_epochs+1):
        total_loss = 0.0  # 전체 손실 초기화
        for inputs, targets in dataloader:  # 데이터로더를 통해 배치 단위로 데이터를 가져옴
            optimizer.zero_grad()  # optimizer의 gradient를 초기화하여 중복 계산을 방지

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # 각 배치의 손실을 누적
            # Backward pass 및 경사하강법

        # 평균 손실 계산 및 출력
        loss_.append(total_loss/len(dataloader))
        if(epoch % 100 == 0):
            print(f'Epoch [{epoch}/{num_epochs}], Average Loss: {total_loss/len(dataloader):.4f}')

    # 모델 저장하기
    # torch.save(model.state_dict(), 'feature_vector_model.pth')
    print("##############학습 완료############")

    #### 저장 이후 출력 테스팅
    model.eval()
    for inputs, targets in dataloader:
        outputs = model(inputs)
        for a, b in zip(outputs, targets):
            print(a.item(), b.item())  # 출력값 및 타겟값을 스칼라로 변환하여 출력
        loss = criterion(outputs, targets)
        print(f'Loss: {loss.item():.4f}')
