import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision import models
import torch.nn as nn

# --- Налаштування ---
MODEL_PATH = "resnet50_custom.pth"  # Завантажуємо модель
IMAGE_FOLDER = "dataset/cat"  # Папка з тестовими зображеннями
CLASS_NAMES = ["car", "cat", "dog"] # Назви класів

# --- Завантаження моделі ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)  # Використовуємо ту саму архітектуру
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))  # Встановлюємо правильний вихідний шар
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()  # Переводимо в режим оцінки

# --- Підготовка зображень ---
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Той самий розмір, що й при навчанні
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # Стандартна нормалізація

])

# --- Перевірка всіх зображень ---
correct = 0
total = 0

for filename in os.listdir(IMAGE_FOLDER):
    if filename.endswith((".jpg", ".png", ".jpeg")):  # Перевіряємо тільки зображення
        image_path = os.path.join(IMAGE_FOLDER, filename)
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)  # Додаємо batch dimension

        # --- Передбачення ---
        with torch.no_grad():
            output = model(image)
            predicted_class = torch.argmax(output, dim=1).item()

        # --- Перевірка правильності ---
        actual_class = CLASS_NAMES.index(IMAGE_FOLDER.split('/')[1])  # Припускаємо, що ім'я файлу містить "dog" або "cat"
        is_correct = predicted_class == actual_class
        total += 1
        correct += int(is_correct)

        print(f"{filename}: Передбачено - {CLASS_NAMES[predicted_class]}, Реальність - {CLASS_NAMES[actual_class]}, {'✅' if is_correct else '❌'}")

# --- Вивід загальної точності ---
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"\nЗагальна точність: {accuracy:.2f}%")
