import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms

from resnet18.custom_dataset import CustomDataset
from resnet18.resnet18classifier import ResNet18Classifier

if __name__ == '__main__':
    BATCH_SIZE = 32
    NUM_CLASSES = 26
    EPOCHS = 10
    LEARNING_RATE = 0.001

    DATASET_ROOT = r'G:\pill3'
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 수평 뒤집기
        transforms.RandomRotation(15),  # -15도에서 15도 사이로 회전
        transforms.ColorJitter(brightness=0.2,  # 밝기, 대비, 채도, 색조 변화
                               contrast=0.2,
                               saturation=0.2,
                               hue=0.1),
        transforms.ToTensor(),
    ])
    PREDICT_TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),  # numpy 이미지를 PIL로 변환
        transforms.Resize((224, 224)),  # ResNet 입력 크기
        transforms.ToTensor(),  # Tensor로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 모델 학습에 사용된 평균과 표준편차
    ])
    DATASET_SPLIT = {'train_weight': 8, 'val_weight': 1, 'test_weight': 1}

    logger = TensorBoardLogger("logs/", name="resnet18-classifier")

    checkpoint_callback = ModelCheckpoint(
        dirpath='./models',
        filename='resnet18_best',
        monitor="val_acc",  # 검증 정확도를 기준으로 최고 모델 저장
        mode="max",  # 검증 정확도가 높을수록 좋은 모델로 간주
        save_top_k=1,  # 성능이 최고일 때만 저장
        save_weights_only=True  # 전체 모델이 아닌 state_dict만 저장
    )

    dataset = CustomDataset(root=DATASET_ROOT, batch_size=BATCH_SIZE, transform=TRAIN_TRANSFORM)
    train, val, test = dataset.split(**DATASET_SPLIT)
    idx_to_class = dataset.idx_to_class
    print('Dataset loaded.')

    model = ResNet18Classifier(num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE, idx_to_class=idx_to_class)
    trainer = Trainer(
        max_epochs=EPOCHS,
        devices=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback],
        logger=logger
    )
    print('Model loaded.')

    # 모델 학습
    trainer.fit(model, train, val)

    # 모델 테스트
    trainer.test(model, test)

    # 모델 저장
    torch.save(model.state_dict(), r'./models/resnet18_last.pth')

    best = ResNet18Classifier.from_pretrained(r'./models/resnet18_best.ckpt', num_classes=NUM_CLASSES)
    torch.save(best.state_dict(), r'./models/resnet18_best.pth')
