import cv2

from pill_color_classifier import PillColorClassifier
from pill_detector import PillDetector
from resnet18.resnet18classifier import ResNet18Classifier
from speed import speed
from thread_with_return import ThreadWithReturn


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
            './resnet18/models/resnet18_best.pth',
            num_classes=26,
            idx_to_class=self.shapes
        )
        self.pill_color_classifier = PillColorClassifier()

    def __call__(self, image_path):
        return self.extract(image_path)

    def extract_with_path(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.extract(image)

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

        for i, cropped_image in enumerate(cropped_images):
            print(f"{i}: {colors[i]} {shapes[i]}")
        #     cv2.imshow(f'{i}', cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        shapes = self.cleanup_shape(shapes)
        return [{'color': color, 'shape': shape} for color, shape in zip(colors, shapes)]

    @speed
    def cleanup_shape(self, shapes: list[str]):
        shapes_ = []
        for shape in shapes:
            if shape == '타원형':
                shapes_.append(['타원형', '장방형'])
            elif shape == '장방형':
                shapes_.append(['장방형', '타원형'])
            elif shape.startswith('기타'):
                shapes_.append(['기타'])
            else:
                shapes_.append([shape])

        return shapes_


if __name__ == '__main__':
    extractor = PillFeatureExtractor()
    pills = extractor(r'F:\zer0ken\how-does-pill-look-like\images\img_2.png')
    print(pills)
