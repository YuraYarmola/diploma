import os
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image

# --- Гіперпараметри ---
BATCH_SIZE = 32
EPOCHS = 10
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


# --- Кастомний датасет, який обробляє всі підпапки ---
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


# --- Створюємо індекси для тренувальної та валідаційної вибірки ---
indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=[dataset[idx][1].item() for idx in indices])

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_labels = [dataset[idx][1].item() for idx in train_indices]
val_labels = [dataset[idx][1].item() for idx in val_indices]

print("Розподіл тренувальних класів:", Counter(train_labels))
print("Розподіл валідаційних класів:", Counter(val_labels))


# --- Створюємо завантажувачі даних ---
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=2)
# --- Завантажуємо модель ResNet-50 ---
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_classes = len(dataset.class_to_idx)  # Визначаємо кількість класів
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Оновлюємо останній шар

model = model.to(device)

# --- Функції втрат та оптимізатор ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# --- Функція тренування ---
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        val_acc = evaluate_model(model, val_loader)
        print(
            f"🌀 Епоха {epoch + 1}/{epochs} | Втрата: {running_loss:.4f} | Тренувальна точність: {train_acc:.2f}% | Валідаційна точність: {val_acc:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Модель збережена в {MODEL_PATH}")


# --- Функція оцінки ---
def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    return accuracy


# --- Запуск тренування ---
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
