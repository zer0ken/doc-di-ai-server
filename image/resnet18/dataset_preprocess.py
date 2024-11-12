import random
import zipfile
import json
import os

from PIL import Image
from collections import defaultdict


# 경로 설정
source_data_dir = 'E:\\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\\01.데이터\\1.Training'
label_data_dir = os.path.join(source_data_dir, '라벨링데이터', '단일경구약제 5000종')
image_data_dir = os.path.join(source_data_dir, '원천데이터', '단일경구약제 5000종')
output_base_dir = 'G:\\pill3'

# 모양별 디렉토리에 저장할 최대 이미지 수
max_images_per_shape = 1000

# 알약별 색상 정보 저장을 위한 딕셔너리
color_info = {}
# 각 모양별 이미지 수 카운트
shape_counts = defaultdict(int)

# 기존의 데이터셋 디렉토리에 포함된 이미지 수를 확인하여 카운트 초기화
if os.path.exists(output_base_dir):
    for shape_dir_name in os.listdir(output_base_dir):
        shape_dir_path = os.path.join(output_base_dir, shape_dir_name)
        if os.path.isdir(shape_dir_path):
            shape_counts[shape_dir_name] = len([f for f in os.listdir(shape_dir_path) if f.endswith('.png')])


# 데이터셋 구축 함수
def build_dataset():
    log_step = 0  # 로그 기록용 변수

    # 원천 데이터 ZIP 파일 순회
    for image_zip_name in sorted(os.listdir(image_data_dir)):
        if not image_zip_name.endswith('.zip'):
            continue

        image_zip_path = os.path.join(image_data_dir, image_zip_name)
        label_zip_name = image_zip_name.replace("TS", "TL")
        label_zip_path = os.path.join(label_data_dir, label_zip_name)

        with zipfile.ZipFile(image_zip_path, 'r') as img_zip, zipfile.ZipFile(label_zip_path, 'r') as lbl_zip:
            print(f"Processing ZIP file: {image_zip_name}")

            for folder_name in img_zip.namelist():
                if not folder_name.endswith('/'):
                    continue

                # 알약 ID 추출
                pill_id = folder_name.split('/')[0]
                img_files = [f for f in img_zip.namelist() if f.startswith(folder_name) and f.endswith('.png')]
                lbl_folder = f"{pill_id}_json/"
                lbl_files = [f for f in lbl_zip.namelist() if f.startswith(lbl_folder) and f.endswith('.json')]

                if not img_files or not lbl_files:
                    continue

                # 이미지와 라벨 파일 하나씩 선택
                indices = random.sample(range(len(img_files)), 50)

                for idx in indices:
                    try:
                        img_file, lbl_file = img_files[idx], lbl_files[idx]
                    except IndexError:
                        continue

                    # 라벨 파일 읽기
                    with lbl_zip.open(lbl_file) as f:
                        label_data = json.load(f)
                        shape = label_data['images'][0]['drug_shape']
                        color1 = label_data['images'][0]['color_class1']
                        color2 = label_data['images'][0]['color_class2']
                        chart = label_data['images'][0]['chart']
                        bbox = label_data['annotations'][0]['bbox']  # xywh 형식 bbox

                    if not shape or not color1 or not bbox:
                        continue

                    # 모양별 최대 이미지 수 제한 확인
                    if shape_counts[shape] >= max_images_per_shape and shape in ('원형', '장방형', '타원형'):
                        continue

                    # 색상 정보 저장
                    color_info[os.path.basename(img_file)] = [color1, color2] if color2 else [color1]


                    # 출력 경로 및 파일명 설정
                    shape_dir = os.path.join(output_base_dir, shape)
                    os.makedirs(shape_dir, exist_ok=True)
                    dest_image_path = os.path.join(shape_dir, os.path.basename(img_file))

                    # 동일한 파일명 존재 여부 확인
                    if os.path.exists(dest_image_path):
                        continue  # 이미 존재하는 경우 스킵

                    # 이미지 파일 열기 및 크롭
                    with img_zip.open(img_file) as img_file_obj:
                        img = Image.open(img_file_obj)
                        img_w, img_h = img.size
                        x, y, w, h = bbox
                        x2, y2 = x + w, y + h

                        size = max(w, h)
                        if w < size:
                            diff = size - w
                            x = max(0, x - diff // 2)
                            x2 = min(img_w, x2 + diff // 2)
                        if h < size:
                            diff = size - h
                            y = max(0, y - diff // 2)
                            y2 = min(img_h, y2 + diff // 2)
                        cropped_img = img.crop((x, y, x2, y2))

                        # 크롭된 이미지 저장
                        cropped_img.save(dest_image_path)

                    # 카운트 갱신
                    shape_counts[shape] += 1

                    # 주기적인 로그 출력
                    if log_step % 100 == 0:
                        print(f"Current image counts by shape:")
                        for shp, count in shape_counts.items():
                            print(f"  Shape '{shp}': {count} images")

                    log_step += 1

                    # 모든 모양의 디렉토리가 최대 이미지 수에 도달하면 중단
                    if all(count >= max_images_per_shape for count in shape_counts.values()):
                        print("Dataset fully populated for all shapes.")
                        save_color_info()
                        return

                    if shape in ('원형', '장방형', '타원형'):
                        break

    # 색상 정보 저장
    save_color_info()


# 색상 정보 저장 함수
def save_color_info():
    color_file_path = os.path.join(output_base_dir, "colors.json")

    # 기존 파일이 존재하면 병합
    if os.path.exists(color_file_path):
        with open(color_file_path, 'r', encoding='utf-8') as f:
            existing_colors = json.load(f)
            color_info.update(existing_colors)

    # 병합된 색상 정보를 파일로 저장
    with open(color_file_path, 'w', encoding='utf-8') as f:
        json.dump(color_info, f, ensure_ascii=False, indent=4)
    print(f"Color information saved to {color_file_path}")


if __name__ == '__main__':
    # 주기적인 로그 출력
    print(f"Current image counts by shape:")
    for shp, count in shape_counts.items():
        print(f"  Shape '{shp}': {count} images")
    # 데이터셋 구축 실행
    build_dataset()
