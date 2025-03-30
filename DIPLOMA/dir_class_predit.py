import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision import models
import torch.nn as nn

# --- Налаштування ---
MODEL_PATH = "mobilenet_v3_small_custom_with_metadata.pth"  # Завантажуємо модель
IMAGE_FOLDER = "dataset/dog"  # Папка з тестовими зображеннями

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)
metadata = checkpoint['metadata']
CLASS_NAMES = metadata['class_names']
num_classes = len(CLASS_NAMES)

model = getattr(models, metadata.get('model_type', 'resnet50'))(pretrained=False)
if hasattr(model, "classifier"):
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
elif hasattr(model, "fc"):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()  # Переводимо в режим оцінки

# --- Підготовка зображень ---
transform = transforms.Compose([
    transforms.Resize((metadata.get("img_size", 32), metadata.get("img_size", 32))),
    transforms.ToTensor(),
    transforms.Normalize(mean=metadata.get("normalization_mean"),
                         std=metadata.get("normalization_std"))

])

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
        actual_class = CLASS_NAMES.index(
            IMAGE_FOLDER.split('/')[1])  # Припускаємо, що ім'я файлу містить "dog" або "cat"
        is_correct = predicted_class == actual_class
        total += 1
        correct += int(is_correct)

        print(
            f"{filename}: Передбачено - {CLASS_NAMES[predicted_class]}, Реальність - {CLASS_NAMES[actual_class]}, {'✅' if is_correct else '❌'}")

# --- Вивід загальної точності ---
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"\nЗагальна точність: {accuracy:.2f}%")
