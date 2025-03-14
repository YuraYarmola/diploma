import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models

# --- Гіперпараметри ---
BATCH_SIZE = 64
MODEL_PATH = r"D:\LPNU\DIPLOMA\DIPLOMA\resnet50_custom.pth"  # Шлях до збереженої моделі

# --- Визначаємо, чи доступний GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Завантаження тестового датасету CIFAR-10 ---
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])    # Стандартна нормалізація
])

test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- Ініціалізація моделі ---
def load_model(model_path, num_classes=10):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Оновлюємо останній шар
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # ВАЖЛИВО: Переключення в режим оцінки
    return model

# --- Функція оцінки моделі ---
def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            print(predicted)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Додатковий вивід (перевіряємо правильність передбачень)
            # if total <= 10:  # Друкуємо тільки перші 10
            print(f"🎯 Реальні мітки: {labels.tolist()}")
            print(f"🤖 Передбачення: {predicted.tolist()}")

    accuracy = correct / total * 100
    print(f"✅ Точність моделі на CIFAR-10: {accuracy:.2f}%")


# --- Запуск перевірки ---
if __name__ == "__main__":
    model = load_model(MODEL_PATH, 3)
    evaluate_model(model, test_loader)
