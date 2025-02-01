import cv2
import os
import time
from tkinter import Tk, Label


def save_image_and_label(frame, bbox, class_id, image_id):
    """Зберігає зображення та відповідну анотацію у YOLO-форматі"""
    x, y, w, h = [int(v) for v in bbox]

    # Створюємо необхідні директорії
    images_dir = os.path.join("dataset", "images", "train")
    labels_dir = os.path.join("dataset", "labels", "train")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Генеруємо унікальну назву файлу
    image_filename = f"{image_id}.jpg"
    image_path = os.path.join(images_dir, image_filename)

    # Зберігаємо зображення
    cv2.imwrite(image_path, frame)

    # Нормалізація координат для YOLO
    img_height, img_width, _ = frame.shape
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    # Зберігаємо анотацію у YOLO-форматі
    label_filename = f"{image_id}.txt"
    label_path = os.path.join(labels_dir, label_filename)

    with open(label_path, "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    print(f"Збережено: {image_path} та {label_path}")


def main():
    video = cv2.VideoCapture(0)
    trackers = []
    class_map = {}  # Відображення назв класів у індекси
    last_save_time = time.time()
    image_counter = 0  # Лічильник для файлів

    def on_close():
        """Закриває всі вікна при виході"""
        video.release()
        cv2.destroyAllWindows()
        root.destroy()

    root = Tk()
    root.title("Відстеження об'єктів")
    root.protocol("WM_DELETE_WINDOW", on_close)

    label = Label(root, text="Натисніть 'n' для вибору об'єкта")
    label.pack()

    def video_loop():
        nonlocal last_save_time, image_counter
        ok, frame = video.read()
        if not ok:
            root.quit()
            return

        for tracker, class_id in trackers[:]:
            ok, bbox = tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                if time.time() - last_save_time >= 0.5:
                    save_image_and_label(frame, bbox, class_id, image_counter)
                    last_save_time = time.time()
                    image_counter += 1
            else:
                trackers.remove((tracker, class_id))
                print(f"Трекер для класу {class_id} видалено.")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            bbox = cv2.selectROI("Select ROI", frame, False)
            cv2.destroyWindow("Select ROI")
            if sum(bbox) > 0:
                class_name = input("Введіть назву класу: ")

                # Призначаємо класу унікальний ID
                if class_name not in class_map:
                    class_map[class_name] = len(class_map)

                class_id = class_map[class_name]

                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, bbox)
                trackers.append((tracker, class_id))

        cv2.imshow("Object Tracking", frame)
        root.after(10, video_loop)

    video_loop()
    root.mainloop()

    # Зберігаємо мапу класів у `classes.txt` для майбутнього використання
    with open("dataset/classes.txt", "w") as f:
        for class_name, class_id in class_map.items():
            f.write(f"{class_name}\n")


if __name__ == "__main__":
    main()
