import os

dataset_path = "dataset"
images_path = os.path.join(dataset_path, "images", "train")
labels_path = os.path.join(dataset_path, "labels", "train")

# Створюємо папки, якщо вони відсутні
os.makedirs(images_path, exist_ok=True)
os.makedirs(labels_path, exist_ok=True)

# Створюємо пусті файли анотацій, якщо їх немає
for image_file in os.listdir(images_path):
    label_file = image_file.replace(".jpg", ".txt").replace(".png", ".txt")
    label_path = os.path.join(labels_path, label_file)

    if not os.path.exists(label_path):
        with open(label_path, "w") as f:
            f.write("")  # Порожній файл анотації, щоб уникнути помилки

print("Перевірка завершена: структура папок виправлена!")
