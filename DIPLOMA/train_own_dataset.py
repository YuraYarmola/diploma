import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler

from custom_dataset_loader import CustomDataset
from model_get import get_model
from normalizer import compute_mean_std

# --- Гіперпараметри ---
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
IMG_SIZE = 32

MODEL_TYPE = "mobilenet_v3_small"
MODEL_PATH = f"{MODEL_TYPE}_custom_with_metadata.pth"

DATASET_PATH = r"D:\LPNU\DIPLOMA\DIPLOMA\dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



transform = transforms.Compose([
    transforms.ToTensor()
])


def make_normilize_transform(dataset):
    # --- Завантажуємо датасет ---

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    if len(dataset) == 0:
        raise ValueError("❌ Помилка: датасет порожній. Переконайтеся, що дані коректні.")

    mean, std = compute_mean_std(dataloader)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform, mean, std


# --- Створюємо індекси для тренувальної та валідаційної вибірки ---
def train(dataset):
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2,
                                                  stratify=[dataset[idx][1].item() for idx in indices])

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # --- Створюємо завантажувачі даних ---
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=2)
    num_classes = len(dataset.class_to_idx)  # Визначаємо кількість класів

    # 🔹 Завантаження моделі


    # 🔹 Ініціалізація моделі
    model = get_model(MODEL_TYPE, num_classes).to(DEVICE)

    # --- Функції втрат та оптимізатор ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def evaluate_model(model, dataloader):
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total * 100
        return accuracy

    # --- Функція тренування ---
    def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct, total = 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

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

        metadata = {
            "normalization_mean": list(mean),
            "normalization_std": list(std),
            "img_size": IMG_SIZE,
            "model_type": MODEL_TYPE,
            "class_names": list(dataset.class_to_idx.keys())
        }
        torch.save({
            "model_state_dict": model.state_dict(),
            "metadata": metadata
        }, MODEL_PATH)
        print(f"✅ Модель збережена в {MODEL_PATH}")

    train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)


# --- Запуск тренування ---
if __name__ == "__main__":
    dataset = CustomDataset(root_dir=DATASET_PATH, transform=transform)

    transform, mean, std = make_normilize_transform(dataset)
    dataset.transform = transform
    train(dataset)
