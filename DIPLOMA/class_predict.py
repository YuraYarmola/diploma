import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision import models
import torch.nn as nn

# --- Налаштування ---
MODEL_PATH = "resnet50_custom.pth"  # Завантажуємо останню збережену модель
CLASS_NAMES = ["car", "cat", "dog"]  # Назви класів
IMAGE_PATH = r"C:\Users\uarmo\Downloads\car2.jpeg" # Шлях до тестового зображення

# --- Завантаження моделі ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)  # Використовуємо ту саму архітектуру
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))  # Встановлюємо правильний вихідний шар
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()  # Переводимо в режим оцінки

# --- Підготовка зображення ---
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Той самий розмір, що й при навчанні
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # Стандартна нормалізація

])

image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image).unsqueeze(0).to(device)  # Додаємо batch dimension

# --- Передбачення ---
with torch.no_grad():
    output = model(image)
    predicted_class = torch.argmax(output, dim=1).item()

print(f"Передбачений клас: {CLASS_NAMES[predicted_class]}")
