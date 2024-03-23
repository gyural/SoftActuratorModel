import ezdxf
import random
import numpy as np

def create_pointy_rectangle_dxf(filename, lower_left, upper_right, num_points, sharpness):
        doc = ezdxf.new('R2010')  # 새로운 DXF 문서 생성
        msp = doc.modelspace()  # 모델 공간 얻기

        # 바깥 면의 레이어 추가
        outer_layer = doc.layers.new(name='Outer', dxfattribs={'color': 1})  # 바깥 면의 색: 1 (빨간색)

        # 안쪽 면의 레이어 추가
        inner_layer = doc.layers.new(name='Inner', dxfattribs={'color': 5})  # 안쪽 면의 색: 5 (파란색)

        # 직사각형의 윗변의 점 생성
        points = []
        small_points = []
        for i in range(num_points + 1):
            x = lower_left[0] + i * (upper_right[0] - lower_left[0]) / num_points
            y = upper_right[1] + random.uniform(-sharpness, sharpness)
            points.append((x, y))
            small_points.append((x, y - thickness))

        points.insert(0, (lower_left[0], lower_left[1]))
        points.append((upper_right[0], lower_left[1]))
        small_points.insert(0, (0, 0))

        new_points = []
        for i in range(1, num_points+2):
            if points[i-1][1] > points[i][1] < points[i+1][1]:
                # Case 1: 포인트 2개 추가
                ratio1 = (points[i][1] - points[i-1][1]) / (sharpness*2)
                ratio2 = (points[i+1][1] - points[i][1]) / (sharpness*2)
                new_x1 = small_points[i][0] + ratio1 * (small_points[i][0] - points[i-1][0])/2
                new_x2 = small_points[i][0] + ratio2 * (points[i+1][0] - small_points[i][0])/2
                new_points.append((new_x1, small_points[i][1]))
                new_points.append((new_x2, small_points[i][1]))
            else:
                # Case 2: 나머지 경우
                vector1_x = points[i-1][0] - points[i][0]
                vector1_y = points[i-1][1] - points[i][1]
                vector2_x = points[i+1][0] - points[i][0]
                vector2_y = points[i+1][1] - points[i][1]
                unitvector1_x = vector1_x / (vector1_x**2 + vector1_y**2)**(1/2)
                unitvector1_y = vector1_y / (vector1_x**2 + vector1_y**2)**(1/2)
                unitvector2_x = vector2_x / (vector2_x**2 + vector2_y**2)**(1/2)
                unitvector2_y = vector2_y / (vector2_x**2 + vector2_y**2)**(1/2)
                sum_unitvector_x = (unitvector1_x + unitvector2_x) / ((unitvector1_x+ unitvector2_x)**2 + (unitvector1_y+ unitvector2_y)**2) ** (1/2)
                sum_unitvector_y = (unitvector1_y + unitvector2_y) / ((unitvector1_x+ unitvector2_x)**2 + (unitvector1_y+ unitvector2_y)**2) ** (1/2)
                #사이각 계산
                vector1 = np.array([unitvector1_x, unitvector1_y, 0])
                vector2 = np.array([unitvector2_x, unitvector2_y, 0])
                dot_product = np.dot(vector1, vector2)
                magnitude1 = np.linalg.norm(vector1)
                magnitude2 = np.linalg.norm(vector2)
                cos_angle = dot_product /(magnitude1 * magnitude2)
                angle = np.arccos(cos_angle)
                distance = thickness / np.sin(angle/2)
                if ((points[i-1][1] + points[i+1][1])/2 > points[i][1] and i != 1 and i != num_points+1) :      #둔각: vector 방향 반대, 길이는 두께로, i != num_points+1 or...
                    new_points.append((points[i][0] - thickness * sum_unitvector_x, points[i][1] - thickness * sum_unitvector_y))
                elif (points[i-1][1] + points[i+1][1])/2 == points[i][1]:
                    pass
                else:                                                       #예각
                    new_points.append((points[i][0] + distance * sum_unitvector_x, points[i][1] + distance * sum_unitvector_y))

        # 좌하단, 우하단의 좌표를 윗변의 끝에 추가하여 직사각형 완성
        new_points.insert(0, (lower_left[0] + thickness, lower_left[1] + thickness))                      #좌측 하단점
        new_points.append((upper_right[0] - thickness, lower_left[1] + thickness))                        #우측 하단점

        # 바깥면 그리기 (빨간색)
        msp.add_lwpolyline(points, close=True, dxfattribs={'layer': 'Outer'})

        # 안쪽 면 그리기 (파란색)
        msp.add_lwpolyline(new_points, close=True, dxfattribs={'layer': 'Inner'})

        # DXF 파일 저장
        doc.saveas(filename)
        #함수 작동시 마다 points 리턴하기
        return points

###################### 수정된 부분 ###########################
points_list = []
# 파일 이름 및 직사각형의 좌표 설정
for j in range (1, 101):
    directory = "C:\\Users\\dgw04\\PycharmProjects\\SoftActuratorModel\\datas\\points\\"
    filename = directory + f"2D_actuator_design7_{j}.dxf"
    lower_left = (0, 0)
    upper_right = (20, 5)
    num_points = 30  # 윗변의 점의 수
    sharpness = 2  # 뾰족함 정도 (양수 또는 음수)
    thickness = 0.3

    # DXF 파일 생성
    p = create_pointy_rectangle_dxf(filename, lower_left, upper_right, num_points, sharpness)
    points_list.append(p)

# Points 데이터 txt로 변환
with open(f"{directory}\\points.txt", "w") as file:
    for points in points_list:
        for point in points:
            file.write(f"({point[0]}, {point[1]}) ")
        file.write(f"\n")
###################### 수정된 부분 ###########################