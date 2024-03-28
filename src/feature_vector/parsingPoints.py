# input sample "(1.xxx, 2.xxx"
# @return Actuator 특징점 리스트
def get_points_by_list(a):
    result = []
    for p in a[:len(a)-1]:
        start = p.find("(")
        comma = p.find(",")

        x = float(p[start+1:comma])
        y = float(p[comma+2:])
        result.append([x, y])
    return result
def read_data_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.rstrip()
            points = get_points_by_list(line.split(")"))
            data.append(points)

    return data
# txt파일로 있는 특징점을 리스트로 반환
def get_feature_points():
    user = "dgw04"
    user_lab = "Hilal"
    base_path = f"C:\\Users\\{user_lab}\\PycharmProjects\\SoftActuratorModel\\"
    points_path = base_path + "datas\\points\\points.txt"

    file_path = points_path  # 파일 경로를 적절히 지정해주세요.
    data = read_data_from_file(file_path)

    # 데이터 출력 (확인용)
    return data

# test용 main
if __name__ == "__main__":
    main()