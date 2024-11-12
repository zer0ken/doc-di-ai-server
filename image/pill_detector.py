from threading import Thread

from ultralytics import YOLO

from image.speed import speed


class PillDetector:
    def __init__(self):
        self.model = YOLO('models/yolov11_best.pt')

    def __call__(self, image_path):
        return self.crop_pill(image_path)

    @speed
    def crop_pill(self, image):
        results = self.model(image, conf=0.60, imgsz=1024)
        image_h, image_w, _ = image.shape

        threads = []
        cropped_images = []

        def _crop_pill(_box):
            x1, y1, x2, y2 = _box.xyxy[0]
            w, h = x2-x1, y2-y1
            size = max(w, h)
            if w < size:
                diff = size - w
                x1 = max(0, x1 - diff / 2)
                x2 = min(x2 + diff / 2, image_w)
            if h < size:
                diff = size - h
                y1 = max(0, y1 - diff / 2)
                y2 = min(y2 + diff / 2, image_h)
            cropped_img = image[int(y1):int(y2), int(x1):int(x2)]
            cropped_images.append(cropped_img)

        for box in results[0].boxes:
            thread = Thread(target=_crop_pill, args=(box,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        return cropped_images
