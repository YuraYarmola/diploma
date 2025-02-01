import cv2
from ultralytics import YOLO

# Завантажуємо попередньо навчену модель YOLOv8
model = YOLO(r"D:\LPNU\DIPLOMA\DIPLOMA\yolo11m.pt")  # nano-версія (найшвидша)

# Відкриваємо вебкамеру (0 - основна, 1 - додаткова)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy  # Отримуємо координати
        class_ids = result.boxes.cls  # Отримуємо ID класів
        confidences = result.boxes.conf  # Отримуємо точність

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)  # Координати рамки
            class_id = int(class_ids[i])  # Індекс класу
            confidence = confidences[i]  # Точність розпізнавання

            # Отримуємо ім'я класу (завантажене з моделі)
            label = model.names[class_id]
            text = f"{label} ({confidence:.2f})"

            # Малюємо рамку та підпис
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Відображаємо результат
    cv2.imshow("YOLOv8 Live Detection", frame)

    # Натисніть "q", щоб вийти
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Закриваємо відеопотік
cap.release()
cv2.destroyAllWindows()
