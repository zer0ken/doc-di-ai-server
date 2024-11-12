from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder


class CustomDataset:
    def __init__(self, root, batch_size=32, transform=None):
        self.dataset = ImageFolder(root=root, transform=transform)
        self.batch_size = batch_size
        self.class_to_idx = self.dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def split(self, train_weight=8, val_weight=1, test_weight=1):
        total = train_weight + val_weight + test_weight

        train_ratio = train_weight / total
        val_ratio = val_weight / total

        dataset_size = len(self.dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size

        train_set, val_set, test_set = random_split(self.dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_set, batch_size=self.batch_size, num_workers=4, persistent_workers=True, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, num_workers=4, persistent_workers=True, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, num_workers=4, persistent_workers=True, shuffle=False)

        return train_loader, val_loader, test_loader
