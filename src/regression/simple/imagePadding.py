import os
from PIL import Image

# 입력 디렉토리와 출력 디렉토리 경로
input_dir = "C:\\Users\\Hilal\\pycharmProjects\\softacturatorModel\\datas\\IMAGE6"
output_dir = "C:\\Users\\Hilal\\pycharmProjects\\softacturatorModel\\datas\\paddingIMG"

# 입력 디렉토리에 있는 모든 이미지 파일 가져오기
image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

# 모든 이미지 파일에 대해 처리
for image_file in image_files:
    # 이미지 파일 경로
    image_path = os.path.join(input_dir, image_file)

    # 이미지 열기
    image = Image.open(image_path)

    # 이미지 크기 확인
    width, height = image.size

    # 새로운 이미지 생성 (800x800 크기, 흰색 배경)
    new_image = Image.new("RGB", (2000, 800), color="white")

    # 이미지를 새 이미지의 중앙에 추가
    x_offset = (2000 - width) // 2
    y_offset = (800 - height) // 2
    new_image.paste(image, (x_offset, y_offset))

    # 출력 디렉토리에 저장
    output_path = os.path.join(output_dir, image_file)
    new_image.save(output_path)

print("전체 이미지 처리가 완료되었습니다.")
