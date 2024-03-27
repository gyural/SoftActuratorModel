import random
import os
import re
import numpy as np

model_path = "C:\\Users\\dgw04\\pycharmProjects\\SoftActuratorModel\\src\\feature_vector\\feature_vector_model.pth"

#이미지 파일명을 통해 bandingangle값 float리스트로 반환
def get_IMG_bandingAngle():
    image_path = "C:\\Users\\Hilal\\PycharmProjects\\SoftActuratorModel\\datas\\afterIMG"
    Y_values = []
    image_filenames = os.listdir(image_path)

    for filename in image_filenames:
        float_number = float(re.findall(r'\d+\.\d+', filename)[0])
        Y_values.append(float_number)
    return Y_values[:100]

# @Return Random한 Actuator Design의 특징점 추출
def create_pointy_rectangle(lower_left, upper_right, num_points, sharpness, thickness):
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
        for i in range(1, num_points + 2):
            if points[i - 1][1] > points[i][1] < points[i + 1][1]:
                # Case 1: 포인트 2개 추가
                ratio1 = (points[i][1] - points[i - 1][1]) / (sharpness * 2)
                ratio2 = (points[i + 1][1] - points[i][1]) / (sharpness * 2)
                new_x1 = small_points[i][0] + ratio1 * (small_points[i][0] - points[i - 1][0]) / 2
                new_x2 = small_points[i][0] + ratio2 * (points[i + 1][0] - small_points[i][0]) / 2
                new_points.append((new_x1, small_points[i][1]))
                new_points.append((new_x2, small_points[i][1]))
            else:
                # Case 2: 나머지 경우
                vector1_x = points[i - 1][0] - points[i][0]
                vector1_y = points[i - 1][1] - points[i][1]
                vector2_x = points[i + 1][0] - points[i][0]
                vector2_y = points[i + 1][1] - points[i][1]
                unitvector1_x = vector1_x / (vector1_x ** 2 + vector1_y ** 2) ** (1 / 2)
                unitvector1_y = vector1_y / (vector1_x ** 2 + vector1_y ** 2) ** (1 / 2)
                unitvector2_x = vector2_x / (vector2_x ** 2 + vector2_y ** 2) ** (1 / 2)
                unitvector2_y = vector2_y / (vector2_x ** 2 + vector2_y ** 2) ** (1 / 2)
                sum_unitvector_x = (unitvector1_x + unitvector2_x) / (
                        (unitvector1_x + unitvector2_x) ** 2 + (unitvector1_y + unitvector2_y) ** 2) ** (1 / 2)
                sum_unitvector_y = (unitvector1_y + unitvector2_y) / (
                        (unitvector1_x + unitvector2_x) ** 2 + (unitvector1_y + unitvector2_y) ** 2) ** (1 / 2)
                # 사이각 계산
                vector1 = np.array([unitvector1_x, unitvector1_y, 0])
                vector2 = np.array([unitvector2_x, unitvector2_y, 0])
                dot_product = np.dot(vector1, vector2)
                magnitude1 = np.linalg.norm(vector1)
                magnitude2 = np.linalg.norm(vector2)
                cos_angle = dot_product / (magnitude1 * magnitude2)
                angle = np.arccos(cos_angle)
                distance = thickness / np.sin(angle / 2)
                if ((points[i - 1][1] + points[i + 1][1]) / 2 > points[i][
                    1] and i != 1 and i != num_points + 1):  # 둔각: vector 방향 반대, 길이는 두께로, i != num_points+1 or...
                    new_points.append(
                        (points[i][0] - thickness * sum_unitvector_x, points[i][1] - thickness * sum_unitvector_y))
                elif (points[i - 1][1] + points[i + 1][1]) / 2 == points[i][1]:
                    pass
                else:  # 예각
                    new_points.append(
                        (points[i][0] + distance * sum_unitvector_x, points[i][1] + distance * sum_unitvector_y))

        # 좌하단, 우하단의 좌표를 윗변의 끝에 추가하여 직사각형 완성
        new_points.insert(0, (lower_left[0] + thickness, lower_left[1] + thickness))  # 좌측 하단점
        new_points.append((upper_right[0] - thickness, lower_left[1] + thickness))  # 우측 하단점

        # 함수 작동시 마다 points 리턴하기
        return points

def get_random_points():
    points_list = []
    for j in range(1, 101):
        lower_left = (0, 0)
        upper_right = (20, 5)
        num_points = 30  # 윗변의 점의 수
        sharpness = 2  # 뾰족함 정도 (양수 또는 음수)
        thickness = 0.3

        # DXF 파일 생성
        p = create_pointy_rectangle(lower_left, upper_right, num_points, sharpness, thickness)
        points_list.append(p)
    return points_list

