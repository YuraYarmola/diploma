import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
import threading
from tkinter import Tk, Button, Label

# Шляхи до моделі
model_path = "model.pth"

# Перевірка наявності CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Використовується пристрій: {device}")


# Завантаження натренованої моделі (MobileNet)
class CustomModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomModel, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        self.base_model.classifier[1] = nn.Linear(self.base_model.last_channel, num_classes)

    def forward(self, x):
        return self.base_model(x)


model = CustomModel(num_classes=10).to(device)

# Завантаження моделі
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Модель завантажена!")
    except Exception as e:
        print(f"Помилка завантаження моделі: {e}")
else:
    print("Модель не знайдена, буде створена нова!")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Параметри
buffer_data = []
buffer_labels = []
buffer_limit = 10
training_active = False

# Трансформації для входу
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Обробка ROI
def preprocess_roi(roi):
    return transform(roi).unsqueeze(0).to(device)


# Донавчання моделі
lock = threading.Lock()


def train_model():
    global buffer_data, buffer_labels
    with lock:
        while training_active:
            if len(buffer_data) >= buffer_limit:
                X_train = torch.stack(buffer_data).to(device)
                y_train = torch.tensor(buffer_labels, dtype=torch.long).to(device)
                dataset = TensorDataset(X_train, y_train)
                dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

                model.train()
                for epoch in range(2):  # Менше епох для швидшого донавчання
                    for batch_data, batch_labels in dataloader:
                        optimizer.zero_grad()
                        outputs = model(batch_data)
                        loss = criterion(outputs, batch_labels)
                        loss.backward()
                        optimizer.step()

                torch.save(model.state_dict(), model_path)
                buffer_data.clear()
                buffer_labels.clear()
                print("Модель збережена. Буфер очищено.")

                # Завантаження оновленої моделі
                model.load_state_dict(torch.load(model_path, map_location=device))
                print("Оновлена модель завантажена для використання.")


# Прогнозування за допомогою моделі
def predict_with_model(frame, bbox):
    x, y, w, h = [int(v) for v in bbox]
    roi = frame[y:y + h, x:x + w]
    roi_tensor = preprocess_roi(roi)
    model.eval()
    with torch.no_grad():
        outputs = model(roi_tensor)
        _, predicted_label = torch.max(outputs, 1)
    return predicted_label.item()


# Інтерфейс керування
def start_training():
    global training_active
    training_active = True
    threading.Thread(target=train_model).start()
    print("Навчання запущено.")


def stop_training():
    global training_active
    training_active = False
    print("Навчання зупинено.")


# Основний цикл
def main():
    global buffer_data, buffer_labels

    video = cv2.VideoCapture(0)
    trackers = []

    def on_close():
        stop_training()
        video.release()
        cv2.destroyAllWindows()
        root.destroy()

    root = Tk()
    root.title("Керування навчанням")
    root.protocol("WM_DELETE_WINDOW", on_close)

    start_button = Button(root, text="Start Training", command=start_training)
    start_button.pack()

    stop_button = Button(root, text="Stop Training", command=stop_training)
    stop_button.pack()

    label = Label(root, text="Використовуйте 'n' для виділення об'єктів")
    label.pack()

    def video_loop():
        ok, frame = video.read()
        if not ok:
            root.quit()
            return

        for tracker, label in trackers[:]:
            ok, bbox = tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in bbox]
                predicted_label = predict_with_model(frame, bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Pred: {predicted_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0),
                            2)
            else:
                trackers.remove((tracker, label))
                print(f"Трекер для мітки {label} видалено.")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            bbox = cv2.selectROI("Select ROI", frame, False)
            cv2.destroyWindow("Select ROI")
            if sum(bbox) > 0:
                label = int(input("Введіть номер класу: "))
                roi = frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                buffer_data.append(preprocess_roi(roi).squeeze(0))
                buffer_labels.append(label)

                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, bbox)
                trackers.append((tracker, label))

        cv2.imshow("Object Detection", frame)
        root.after(10, video_loop)

    video_loop()
    root.mainloop()


if __name__ == "__main__":
    main()
