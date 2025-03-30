import os
import json

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = []
        self.class_to_idx = {}  # Для перетворення назв класів у числові індекси

        # --- Зчитуємо всі підпапки у dataset ---
        for class_idx, class_name in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            annotation_path = os.path.join(class_path, "annotations.json")

            if not os.path.isdir(class_path) or not os.path.exists(annotation_path):
                continue  # Пропускаємо, якщо це не папка або немає анотацій

            # --- Додаємо клас у словник ---
            self.class_to_idx[class_name] = class_idx

            # --- Завантажуємо анотації ---
            with open(annotation_path, "r") as f:
                annotations = json.load(f)

            for ann in annotations:
                self.annotations.append({"img_path": ann["image"], "label": class_idx})

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = Image.open(ann["img_path"]).convert("RGB")
        label = ann["label"]  # Індекс класу

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
