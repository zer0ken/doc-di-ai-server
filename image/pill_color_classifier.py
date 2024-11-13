from collections import Counter

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from image.speed import speed


class PillColorClassifier:
    def __call__(self, image):
        return self.analyse_color(image)

    @speed
    def analyse_color(self, image):# 100x100으로 이미지를 리사이즈하고 K-means 클러스터링을 적용하는 함수
        # 100x100으로 리사이즈
        resized_img = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
        center_x, center_y = 50, 50  # 100x100 이미지에서 중앙

        # 리사이즈된 이미지의 픽셀을 1D로 변환
        pixels = resized_img.reshape(-1, 3).astype(np.float32)

        # K-means 클러스터링 (k=3)
        kmeans = KMeans(n_clusters=3, random_state=0)
        kmeans.fit(pixels)
        labels = kmeans.labels_.reshape(100, 100)  # 클러스터 라벨을 이미지 크기에 맞게 재구성

        # 클러스터 중심을 이미지로 재구성
        segmented_img = kmeans.cluster_centers_[labels].reshape(100, 100, 3)
        segmented_img = np.uint8(segmented_img)  # 0-255 범위로 변환
        # segmented_img = segmented_img[center_y - 15:center_y + 15, center_x - 15:center_x + 15]  # 0-255 범위로 변환

        # 각 클러스터의 중심 색상
        cluster_centers = kmeans.cluster_centers_

        # 30x30 크기의 중앙 영역 추출
        cropped_area = labels[center_y - 15:center_y + 15, center_x - 15:center_x + 15].reshape(-1)

        # 각 클러스터의 비중 계산 (중앙 영역에서 각 클러스터 라벨의 등장 횟수)
        cluster_counts = Counter(cropped_area)
        total_pixels = sum(cluster_counts.values())
        cluster_weights = {label: count / total_pixels for label, count in cluster_counts.items()}
        cluster_weights = {label: weight for label, weight in cluster_weights.items() if weight > 0.1}

        # 비중이 가장 적은 클러스터 제외
        if len(cluster_weights) > 2:
            outer_label = min(cluster_weights, key=cluster_weights.get)
            del cluster_weights[outer_label]
        pill_colors = [tuple(map(int, cluster_centers[label])) for label in cluster_weights.keys()]

        # # 이미지 표시
        # fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        # axes[0].imshow(resized_img)  # 원본 리사이즈 이미지
        # axes[0].set_title(f"Resized Image (100x100)\nMost Common Cluster Colors")
        # axes[0].axis('off')
        #
        # # 세그멘테이션 이미지 표시
        # axes[1].imshow(segmented_img)  # K-means 세그멘테이션 이미지
        # axes[1].set_title("K-means Segmentation")
        # axes[1].axis('off')

        # pill_color = pill_colors
        # # 중앙에 가장 많이 포함된 클러스터 색상 순서대로 표시
        # if len(pill_color) == 2:
        #     color_patch = np.ones((50, 100, 3), dtype=np.uint8)
        #     color_patch[:, :50] = pill_color[0]  # 좌측 색상
        #     color_patch[:, 50:] = pill_color[1]  # 우측 색상
        # else:
        #     color_patch = np.ones((100, 100, 3), dtype=np.uint8)
        #     color_patch[:, :] = pill_color[0]  # 좌측 색상
        # axes[2].imshow(color_patch)  # 색상 패치
        # axes[2].axis('off')
        #
        # plt.show()
        return pill_colors