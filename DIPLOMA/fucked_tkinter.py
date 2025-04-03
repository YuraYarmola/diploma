import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_PATH = "resnet50_custom.pth"

# --- Визначаємо, чи доступний GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Трансформації ---
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Змінюємо розмір для ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # Стандартна нормалізація
])

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
                img_path = os.path.join(class_path, ann["image"])
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

# --- Завантажуємо датасет ---
dataset_path = "dataset"
dataset = CustomDataset(root_dir=dataset_path, transform=transform)

# --- Перевіряємо, чи датасет не пустий ---
if len(dataset) == 0:
    raise ValueError("❌ Помилка: датасет порожній. Переконайтеся, що дані коректні.")

# --- Створюємо індекси для кожного класу ---
class_indices = {class_idx: [] for class_idx in dataset.class_to_idx.values()}
for idx, (_, label) in enumerate(dataset):
    class_indices[label.item()].append(idx)

# --- Визначаємо мінімальну кількість зображень у класі ---
min_class_size = min(len(indices) for indices in class_indices.values())

# --- Створюємо індекси для тренувальної та валідаційної вибірки ---
train_indices = []
val_indices = []

for indices in class_indices.values():
    np.random.shuffle(indices)
    split = int(0.2 * min_class_size)
    train_indices.extend(indices[:split])
    val_indices.extend(indices[split:min_class_size])

# --- Створюємо самплери ---
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# --- Створюємо завантажувачі даних ---
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=2)
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

if __name__ == "__main__":
    for inputs, labels in val_loader:
        for label in labels:
            class_name = idx_to_class[label.item()]  # Отримуємо назву класу за індексом
            print(f"Label Index: {label.item()}, Class Name: {class_name}")
