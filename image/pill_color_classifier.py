import cv2
import numpy as np
import colorsys

from sklearn.cluster import KMeans

from image.speed import speed


class PillColorClassifier:
    def __init__(self):
        self.color_labels = ["노랑", "주황", "분홍", "빨강", "갈색", "연두",
                             "초록", "청록", "파랑", "남색", "자주", "보라"]

        self.color_values = np.array([
            [255, 255, 0],  # 노랑
            [255, 165, 0],  # 주황
            [255, 192, 203],  # 분홍
            [255, 0, 0],  # 빨강
            [139, 69, 19],  # 갈색
            [144, 238, 144],  # 연두
            [0, 128, 0],  # 초록
            [0, 206, 209],  # 청록
            [0, 0, 255],  # 파랑
            [0, 0, 139],  # 남색
            [128, 0, 128],  # 자주
            [128, 0, 128],  # 보라
        ])
        self.hsv_values = np.array([colorsys.rgb_to_hsv(r / 255, g / 255, b / 255) for r, g, b in self.color_values])

    def __call__(self, image):
        return self.classify_color(image)

    @speed
    def classify_color(self, image):
        image = cv2.resize(image, (50, 50))

        crop_size = 20
        height, width, _ = image.shape
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        center_crop = image[top:top + crop_size, left:left + crop_size]

        center_rgb = center_crop.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, n_init='auto')
        kmeans.fit(center_rgb)
        centers = kmeans.cluster_centers_

        colors = []
        for center in centers:
            colors.extend(self.get_names_of_color(center))

        return list(set(colors))

    def get_names_of_color(self, rgb):
        hue, sat, val = colorsys.rgb_to_hsv(*(rgb / 255))

        closest_colors = ["투명"]

        if sat < 0.25:
            if val > 0.6:
                closest_colors.append("하양")
            if 0.8 > val > 0.2:
                closest_colors.append("회색")
            if 0.3 > val:
                closest_colors.append("검정")
        if sat > 0.15:
            hue_distances = np.min(np.stack([
                    (self.hsv_values[:, 0] - hue) ** 2,
                    (self.hsv_values[:, 0] - hue + 1) ** 2,
                    (self.hsv_values[:, 0] - hue - 1) ** 2
                ]), axis=0)

            sat_distances = (self.hsv_values[:, 1] - sat) ** 2

            hue_closest_indices = np.argsort(hue_distances)

            sat_distances[hue_closest_indices[6:]] = 10.0
            total_distances = hue_distances * sat_distances

            total_closest_indices = np.argsort(total_distances)

            closest_colors += [self.color_labels[i] for i in total_closest_indices][:3]

        return closest_colors
