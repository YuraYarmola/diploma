import cv2
import torch
import tkinter as tk
from tkinter import simpledialog, messagebox
import numpy as np
import threading
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

# Глобальні змінні
selected_roi = None
tracker = None
capture_mode = False
capture_images = {}
model = None
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

# Простий датасет для навчання
class ObjectDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1) / 255.0, self.labels[idx]

# Базова модель нейронної мережі
class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Захоплення та трекінг об'єкта
def capture_and_track_object():
    global selected_roi, tracker, capture_mode, capture_images
    capture_mode = True

    cap = cv2.VideoCapture(0)

    def select_roi(event, x, y, flags, param):
        global selected_roi, tracker
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_roi = cv2.selectROI("Video", frame, fromCenter=False, showCrosshair=True)
            if selected_roi != (0, 0, 0, 0):
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, selected_roi)
                # Введення назви об'єкта
                object_name = input("Введіть назву об'єкта: ")
                if object_name:
                    capture_images.setdefault(object_name, [])
                    messagebox.showinfo("Трекінг", f"Об'єкт '{object_name}' вибрано для трекінгу")

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", select_roi)

    while capture_mode:
        ret, frame = cap.read()
        if not ret:
            break

        if tracker is not None and selected_roi is not None:
            success, box = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = list(capture_images.keys())[-1]
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Збереження зображення
                cropped_image = frame[y:y+h, x:x+w]
                if cropped_image.size > 0:  # Перевірка на порожній кадр
                    capture_images[label].append(cv2.resize(cropped_image, (256, 256)))

        cv2.imshow("Video", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Режим перевірки
def detect_objects():
    global model
    if model is None:
        messagebox.showwarning("Помилка", "Спочатку натренуйте модель!")
        return

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        input_image = cv2.resize(frame, (256, 256))
        input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(device)

        # Model inference
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        # Draw bounding box and label
        label = f"Object: {predicted.item()}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)  # Example bounding box

        cv2.imshow("Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# Навчання або донавчання моделі
def train_or_retrain_model():
    global model, capture_images
    if not capture_images:
        messagebox.showwarning("Помилка", "Немає зображень для навчання!")
        return

    images = []
    labels = []
    label_map = {}
    current_label = 0

    for object_name, object_images in capture_images.items():
        if object_name not in label_map:
            label_map[object_name] = current_label
            current_label += 1

        images.extend(object_images)
        labels.extend([label_map[object_name]] * len(object_images))

    dataset = ObjectDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    if model is None:
        model = SimpleModel(num_classes=len(label_map)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Навчання
    for epoch in range(5):
        model.train()
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Епоха {epoch+1}, Втрата: {loss.item()}")

    torch.save(model.state_dict(), "object_model.pth")
    messagebox.showinfo("Успіх", "Модель натреновано та збережено.")

# Функція для запуску в окремих потоках
def run_in_thread(func):
    thread = threading.Thread(target=func)
    thread.daemon = True
    thread.start()

# Графічний інтерфейс
root = tk.Tk()
root.title("Object Detection & Tracking")

btn_capture = tk.Button(root, text="Захопити та відстежувати об'єкт", command=lambda: run_in_thread(capture_and_track_object))
btn_capture.pack(pady=10)

btn_detect = tk.Button(root, text="Режим перевірки", command=lambda: run_in_thread(detect_objects))
btn_detect.pack(pady=10)

btn_train = tk.Button(root, text="Навчити модель", command=lambda: run_in_thread(train_or_retrain_model))
btn_train.pack(pady=10)

root.mainloop()
