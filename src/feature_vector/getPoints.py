directory = "C:\\Users\\Hilal\\PycharmProjects\\SoftActuratorModel\\datas\\points"
data_list = []

with open(f"{directory}\\points.txt", "r") as file:
    for line in file:
        # 줄을 공백을 기준으로 분할하여 리스트로 변환
        line_data = line.strip(")").split()
        data_list.append(line_data)

print(data_list[:2])
