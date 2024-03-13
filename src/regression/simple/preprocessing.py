import os
from torchvision import transforms

from PIL import Image

# 입력 디렉토리와 출력 디렉토리 경로
input_dir = "C:\\Users\\Hilal\\pycharmProjects\\softacturatorModel\\datas\\IMAGE6"
output_dir = "C:\\Users\\Hilal\\pycharmProjects\\softacturatorModel\\datas\\afterIMG"

# 전처리 파이프라인 정의
preprocess = transforms.Compose([
    transforms.Pad(padding=(400, 160), fill=255),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 입력 디렉토리의 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

# 입력 디렉토리의 각 이미지에 대해 전처리 적용 및 저장
for image_file in image_files:
    # 이미지 파일 불러오기
    image_path = os.path.join(input_dir, image_file)
    image = Image.open(image_path)

    # 전처리 적용
    image_tensor = preprocess(image)

    # 출력 디렉토리에 저장
    output_path = os.path.join(output_dir, image_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 이미지 텐서를 PIL 이미지로 변환하여 저장
    image_pil = transforms.ToPILImage()(image_tensor)
    image_pil.save(output_path)

print("전처리가 완료되었습니다.")
