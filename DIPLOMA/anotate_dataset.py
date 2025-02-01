import cv2
import os
from ultralytics import YOLO

# Завантажуємо преднавчену модель (можна змінити на іншу)
model = YOLO("yolov8n.pt")

dataset_path = "dataset"
output_labels = "labels"

os.makedirs(output_labels, exist_ok=True)

# Проходимося по всіх класах у датасеті
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    # Проходимося по всіх фото у класі
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        # Виконуємо детекцію об'єктів на фото
        results = model(img)

        # Формуємо .txt-файл для кожного зображення
        label_file = os.path.join(output_labels, f"{os.path.splitext(img_name)[0]}.txt")

        with open(label_file, "w") as f:
            for result in results:
                for box in result.boxes.xywhn:  # Отримуємо координати у нормалізованому форматі
                    x, y, w, h = box.tolist()
                    f.write(f"{class_name} {x} {y} {w} {h}\n")
