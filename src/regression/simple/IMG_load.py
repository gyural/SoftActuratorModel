import os
import re
from torchvision import transforms
from PIL import Image
image_path = "C:\\Users\\dgw04\\PycharmProjects\\SoftActuratorModel\\datas\\afterIMG"

#전처리된 이미지를 tensor List로 반환
def IMG_loader():
# 이미지 파일명 목록
    image_filenames = os.listdir(image_path)
    # 이미지를 텐서로 변환하여 리스트에 추가
    X_images = []
    for filename in image_filenames:
        # 이미지 불러오기
        image = Image.open(os.path.join(image_path, filename))
        # 이미지를 텐서로 변환
        tensor_image = transforms.ToTensor()(image)
        # 변환된 이미지를 리스트에 추가
        X_images.append(tensor_image)
    return X_images
#이미지 파일명을 통해 bandingangle값 float리스트로 반환
def get_IMG_bandingAngle():
    Y_values = []
    image_filenames = os.listdir(image_path)

    for filename in image_filenames:
        float_number = float(re.findall(r'\d+\.\d+', filename)[0])
        Y_values.append(float_number)
    return Y_values