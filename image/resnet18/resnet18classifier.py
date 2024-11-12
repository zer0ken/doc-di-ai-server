import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import seaborn as sns


class ResNet18Classifier(pl.LightningModule):
    def __init__(self, num_classes=11, learning_rate=1e-4, idx_to_class=None):
        super(ResNet18Classifier, self).__init__()
        self.save_hyperparameters()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        self.predictions = []
        self.labels = []

        self.idx_to_class = idx_to_class

    def forward(self, images, *args, **kwargs):
        return self.model(images, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)

        self.predictions.extend(preds.cpu().numpy())
        self.labels.extend(y.cpu().numpy())

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        self.predictions = np.array(self.predictions)
        self.labels = np.array(self.labels)
        class_correct = [0] * self.hparams.num_classes
        class_total = [0] * self.hparams.num_classes

        for i in range(len(self.labels)):
            label = self.labels[i]
            pred = self.predictions[i]
            if label == pred:
                class_correct[label] += 1
            class_total[label] += 1

        class_accuracies = {
            f"{i if self.idx_to_class is None else self.idx_to_class[i]} accuracy": class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            for i in range(self.hparams.num_classes)
        }

        for class_name, accuracy in class_accuracies.items():
            self.log(class_name, accuracy, prog_bar=True)

            # confusion matrix 계산
            cm = confusion_matrix(self.labels, self.predictions, labels=np.arange(self.hparams.num_classes))

            # confusion matrix 시각화
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(self.hparams.num_classes),
                        yticklabels=np.arange(self.hparams.num_classes))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')

            plt.show()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @classmethod
    def from_pretrained(cls, state_dict_path, *args, **kwargs):
        model = cls(*args, **kwargs)
        state_dict = torch.load(state_dict_path, weights_only=True)
        model.load_state_dict(state_dict)
        return model

    def predict(self, images, transform=None):
        self.eval()

        if transform is None:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 모델 학습에 사용된 평균과 표준편차
            ])

        all_preds = []

        with torch.no_grad():
            for img in images:
                img_tensor = transform(img).unsqueeze(0)

                logits = self.forward(img_tensor)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds[0])

        if self.idx_to_class is not None:
            return [self.idx_to_class[pred] for pred in all_preds]

        return all_preds
