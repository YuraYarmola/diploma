from ultralytics import YOLO


def train_yolo():
    # Завантажуємо YOLO11M (неофіційна версія, але припустимо, що вона існує)
    model = YOLO("yolo11m.pt")

    # Тренуємо модель
    model.train(
        data="dataset/data.yaml",  # Вказуємо файл конфігурації
        epochs=50,
        batch=16,
        imgsz=640,
        device="cuda"
    )

if __name__ == "__main__":
    train_yolo()