import cv2
import numpy as np


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE를 L 채널에 적용
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # LAB 이미지를 다시 합쳐서 BGR로 변환
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def gray_world_white_balance(image):
    # 각 채널의 평균을 계산
    avg_r = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_b = np.mean(image[:, :, 2])

    # 전체 평균
    avg_gray = (avg_b + avg_g + avg_r) / 3

    # 각 채널을 평균에 맞추어 스케일 조정
    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r

    # 채널별로 조정 적용
    image[:, :, 0] = np.clip(image[:, :, 0] * scale_b, 0, 255)
    image[:, :, 1] = np.clip(image[:, :, 1] * scale_g, 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2] * scale_r, 0, 255)

    return image.astype(np.uint8)


def calculate_gamma(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)

    if mean_brightness < 128:
        gamma = 1 + (128 - mean_brightness) / 128
    else:
        gamma = 1 / (1 + (mean_brightness - 128) / 128)
    return gamma


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def auto_gamma_correction(image):
    gamma = calculate_gamma(image)
    return adjust_gamma(image, gamma)