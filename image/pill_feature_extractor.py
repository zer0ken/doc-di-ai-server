import colorsys

import cv2
import numpy as np

from image.colors_ import name_to_color, confusing_reds, confusing_yellows, similar_reds, similar_greens, similar_blues, \
    similar_purples
from image.pill_color_classifier import PillColorClassifier
from image.pill_detector import PillDetector
from image.resnet18.resnet18classifier import ResNet18Classifier
from image.speed import speed
from image.thread_with_return import ThreadWithReturn


class PillFeatureExtractor:
    _INSTANCE = None

    @classmethod
    def get_instance(cls):
        if cls._INSTANCE is None:
            cls._INSTANCE = cls()
        return cls._INSTANCE

    def __init__(self):
        self.shapes = {0: '기타-8자형', 1: '기타-강낭콩형', 2: '기타-구형', 3: '기타-나뭇잎형', 4: '기타-나비형', 5: '기타-눈물형', 6: '기타- 도넛형',
                       7: '기타-레몬형', 8: '기타-방패형', 9: '기타-볼록삼각형', 10: '기타-사과형', 11: '기타-십각형', 12: '기타-치아 형',
                       13: '기타-클로버형', 14: '기타-튜브형', 15: '기타-하트형', 16: '마름모형', 17: '반원형', 18: '사각형', 19: '삼각형',
                       20: '오각형', 21: '원형', 22: '육각형', 23: '장방형', 24: '타원형', 25: '팔각형'}

        self.pill_detector = PillDetector()
        self.pill_shape_classifier = ResNet18Classifier.from_pretrained(
            'image/resnet18/models/resnet18_best.pth',
            num_classes=26,
            idx_to_class=self.shapes
        )
        self.pill_color_classifier = PillColorClassifier()

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

    def __call__(self, image_path):
        return self.extract(image_path)

    def extract_with_path(self, image_path, **kwargs):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.extract(image, **kwargs)

    @speed
    def extract(self, image):
        cropped_images = self.pill_detector(image)

        shapes = self.pill_shape_classifier.predict(cropped_images)
        colors = []

        for i, cropped_image in enumerate(cropped_images):
            thread = ThreadWithReturn(target=self.pill_color_classifier, args=(cropped_image,))
            thread.start()
            colors.append(thread)

        for i, color in enumerate(colors):
            colors[i] = color.join()

        colors_ = self.cleanup_colors(colors)
        shapes_ = self.cleanup_shapes(shapes)
        similar_colors = self.get_similar_colors(colors_)

        for i, cropped_image in enumerate(cropped_images):
            print(f"{i}: {colors_[i]} {shapes_[i]} {similar_colors[i]}")

        # for i, cropped_image in enumerate(cropped_images):
        #     cv2.imshow(f'{i}-{colors_[i]}-{shapes_[i]}', cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return [{'color': color, 'shape': shape, 'similar_color': similar_colors}
                for color, shape, similar_colors in zip(colors_, shapes_, similar_colors)]

    @speed
    def cleanup_shapes(self, shapes: list[str]):
        shapes_ = []
        for shape in shapes:
            if shape.startswith('기타'):
                shapes_.append(['기타'])
            elif shape == '기타-튜브형':
                shapes_.append(['원형'])
            # elif shape == '타원형':
            #     shapes_.append(['타원형', '장방형'])
            # elif shape == '장방형':
            #     shapes_.append(['장방형', '타원형'])
            else:
                shapes_.append([shape])

        return shapes_

    @speed
    def get_similar_colors(self, colorsets: list[list[tuple]]):
        colorsets_ = []
        for colorset in colorsets:
            colorset_ = []
            for color in colorset:
                similar_colors = set()
                if color in similar_reds:
                    similar_colors.update(similar_reds)
                elif color in similar_greens:
                    similar_colors.update(similar_greens)
                elif color in similar_blues:
                    similar_colors.update(similar_blues)
                elif color in similar_purples:
                    similar_colors.update(similar_purples)
                colorset_.append(list(similar_colors))
            colorsets_.append(colorset_)
        return colorsets_

    @speed
    def cleanup_colors(self, colorsets: list[list[tuple]]):

        def hue_dist(rgb1, rgb2):
            rgb1 = np.array(rgb1) / 255.0
            rgb2 = np.array(rgb2) / 255.0

            h1, s1, v1 = colorsys.rgb_to_hsv(*rgb1)
            h2, s2, v2 = colorsys.rgb_to_hsv(*rgb2)

            hue_diff = min(abs(h1 - h2), 1 - abs(h1 - h2)) * 360
            return hue_diff

        def rgb_dist(rgb1, rgb2):
            return np.linalg.norm(np.array(rgb1) - np.array(rgb2))

        def grayscale(c):
            std_dev = np.std(c)  # R, G, B 값의 표준편차 계산
            brightness = np.mean(c)  # 밝기는 R, G, B 값의 평균
            if std_dev < 15 and brightness > 180:
                return '흰색'
            elif std_dev < 12.5 and brightness > 70:
                return '회색'
            elif std_dev < 10:
                return '검정색'
            return None

        for i, color_pair in enumerate(colorsets):
            for j, color in enumerate(color_pair):
                grayscale_closest = grayscale(color)
                colorsets[i][j] = grayscale_closest

                if grayscale_closest is not None:
                    continue
                hue_closest = min(name_to_color.keys(), key=lambda name: hue_dist(name_to_color[name], color))
                colorsets[i][j] = hue_closest

                if hue_closest not in confusing_reds.keys():
                    continue
                red_closest = min(confusing_reds.keys(), key=lambda name: rgb_dist(confusing_reds[name], color))
                colorsets[i][j] = red_closest

                if red_closest not in confusing_yellows.keys():
                    continue
                yellow_closest = min(confusing_yellows.keys(),
                                     key=lambda name: rgb_dist(confusing_yellows[name], color))
                colorsets[i][j] = yellow_closest

            colorsets[i] = list(set(color_pair))

        return colorsets
