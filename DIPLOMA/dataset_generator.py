import threading
import uuid

import cv2
import os
import random
import numpy as np
import json
from tkinter import Tk, filedialog, simpledialog
import tkinter as tk


# --- ФУНКЦІЯ ДЛЯ ВИБОРУ ВІДЕО ---
def select_video():
    video_path = filedialog.askopenfilename(title="Виберіть відеофайл",
                                            filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if not video_path:
        print("Відео не вибрано.")
        quit(-1)
    return video_path


# --- ФУНКЦІЯ ДЛЯ ВИБОРУ ФІЛЬТРІВ ---
def select_settings():
    root = tk.Tk()
    root.withdraw()  # Приховуємо головне вікно

    settings = {}

    def on_submit():
        # Зберігаємо значення у словник
        settings["rotate"] = bool(rotate_var.get())
        settings["flip"] = bool(flip_var.get())
        settings["blur"] = bool(blur_var.get())
        settings["brightness"] = bool(brightness_var.get())
        settings["contrast"] = bool(contrast_var.get())
        settings["noise"] = bool(noise_var.get())
        settings["noise_level"] = float(noise_level_var.get())
        settings["brightness_level"] = float(brightness_level_var.get())
        settings['width'] = int(width_var.get())
        settings['height'] = int(height_var.get())
        settings['stop_at'] = int(stop_at.get())
        filter_window.destroy()  # Закриваємо вікно

    # Створюємо нове вікно для вибору фільтрів
    filter_window = tk.Toplevel(root)
    filter_window.title("Вибір фільтрів")
    filter_window.grab_set()

    # Ініціалізуємо змінні (замість BooleanVar використовуємо IntVar)
    rotate_var = tk.IntVar(value=0)

    flip_var = tk.IntVar(value=0)
    blur_var = tk.IntVar(value=0)
    brightness_var = tk.IntVar(value=0)
    contrast_var = tk.IntVar(value=0)
    noise_var = tk.IntVar(value=0)
    noise_level_var = tk.DoubleVar(value=0.1)
    brightness_level_var = tk.DoubleVar(value=1.2)
    width_var = tk.IntVar(value=128)
    height_var = tk.IntVar(value=128)
    stop_at = tk.IntVar(value=0)
    rewrite = tk.IntVar(value=1)

    # Додаємо чекбокси
    tk.Checkbutton(filter_window, text="Застосовувати повороти?", variable=rotate_var,
                   onvalue=1, offvalue=0).pack(anchor='w')
    tk.Checkbutton(filter_window, text="Застосовувати дзеркальне відображення?", variable=flip_var,
                   onvalue=1, offvalue=0).pack(anchor='w')
    tk.Checkbutton(filter_window, text="Застосовувати розмиття?", variable=blur_var,
                   onvalue=1, offvalue=0).pack(anchor='w')
    tk.Checkbutton(filter_window, text="Змінювати яскравість?", variable=brightness_var,
                   onvalue=1, offvalue=0).pack(anchor='w')
    tk.Scale(filter_window, label="Рівень яскравості", variable=brightness_level_var, from_=0.5, to=2.0, resolution=0.1,
             orient='horizontal').pack(anchor='w')
    tk.Checkbutton(filter_window, text="Змінювати контраст?", variable=contrast_var,
                   onvalue=1, offvalue=0).pack(anchor='w')
    tk.Checkbutton(filter_window, text="Додавати шум?", variable=noise_var,
                   onvalue=1, offvalue=0).pack(anchor='w')
    tk.Scale(filter_window, label="Рівень шуму", variable=noise_level_var, from_=0, to=50, resolution=1,
             orient='horizontal').pack(anchor='w')
    tk.Label(filter_window, text="Розмір вихідного зображення:").pack(anchor='w')
    tk.Entry(filter_window, textvariable=width_var).pack(anchor='w')
    tk.Entry(filter_window, textvariable=height_var).pack(anchor='w')

    tk.Label(filter_window, text="Зупинити на кількості (0 - не зупиняти)").pack(anchor='w')
    tk.Entry(filter_window, textvariable=stop_at).pack(anchor='w')
    tk.Checkbutton(filter_window, text="Перезаписувати?", variable=rewrite,
                   onvalue=1, offvalue=0).pack(anchor='w')
    # Кнопка підтвердження
    tk.Button(filter_window, text="Підтвердити", command=on_submit).pack()

    root.wait_window(filter_window)  # Очікуємо закриття вікна
    root.destroy()

    return settings  # Повертаємо значення


frame = None
temp_frame = None
roi_selected = False
roi = (0, 0, 0, 0)
start_point = None
end_point = None
roi_choose = False
images_processed = 0

def draw_roi(event, x, y, flags, param):
    global roi_selected, roi, start_point, end_point, temp_frame, frame, roi_choose

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_choose = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and start_point is not None:
        end_point = (x, y)
        if roi_choose:
            temp_frame = frame.copy()  # Оновлюємо тимчасове зображення
            cv2.rectangle(temp_frame, start_point, end_point, (0, 255, 0), 2)  # Малюємо прямокутник

    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        x_min, y_min = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
        x_max, y_max = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])

        # Обмежуємо координати в межах кадру
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)

        # Перевіряємо, чи ROI має допустимий розмір
        if x_max > x_min and y_max > y_min:
            roi_selected = True
            roi = (x_min, y_min, x_max - x_min, y_max - y_min)
        else:
            roi_selected = False
        roi_choose = False
        temp_frame = frame.copy()  # Оновлюємо тимчасове зображення
        cv2.rectangle(temp_frame, start_point, end_point, (0, 255, 0), 2)


def process_video(video_path, settings):
    pause = True
    if not video_path:
        return

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FPS, 1000)
    class_name = simpledialog.askstring("Клас об'єкта", "Введіть назву класу об'єкта (напр., car, person)")
    save_path = f"dataset/{class_name}"
    os.makedirs(save_path, exist_ok=True)
    annotations = []

    global roi_selected, roi, frame, temp_frame
    cv2.namedWindow("Трекінг об'єкта")
    cv2.setMouseCallback("Трекінг об'єкта", draw_roi)

    ret, frame = cap.read()
    if not ret:
        print("Не вдалося завантажити відео.")
        return

    tracker = None
    frame_count = 0

    while cap.isOpened():
        # Якщо ROI вибрано, ініціалізуємо трекер
        if roi_selected and tracker is None:
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, roi)
            roi_selected = True

        # Оновлення трекера
        if tracker and not pause:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [max(0, min(int(v), frame.shape[i % 2])) for i, v in enumerate(bbox)]
                w, h = (w if x+w <= frame.shape[1] else frame.shape[1] - x, h if y+h <= frame.shape[0] else frame.shape[0] - y)
                # Перевірка коректності координат
                if w > 0 and h > 0 and x >= 0 and y >= 0:
                    # Нормалізація координат та збереження
                    obj_img = frame[y:y + h, x:x + w]
                    obj_img = cv2.resize(obj_img, (settings['width'], settings['height']))

                    # Збереження кадрів і анотацій у окремому потоці
                    thread = threading.Thread(target=save_augmented_images, args=(
                        obj_img, settings, save_path, frame_count, x, w, y, h, frame, class_name, annotations), daemon=True)
                    thread.start()

                    cv2.rectangle(temp_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if temp_frame is not None:
            display_frame = temp_frame.copy()
        else:
            display_frame = frame.copy()
        # Відображення кадру
        cv2.imshow("Трекінг об'єкта", display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):  # Натискання Esc виходить із циклу
            break

        if key == ord(" "):
            if not roi_selected and pause:
                print(f"ROI {roi_selected} PAUSE {pause}")
                print("ROI не вибрано.")
                continue
            roi_selected = True
            tracker = None
            pause = not pause

        if not pause:
            ret, frame = cap.read()
            if not ret:
                break
            temp_frame = frame.copy()
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    write_data = None
    if settings["rewrite"]:
        write_data = annotations
    else:
        with open(f"{save_path}/annotations.json", "r") as f:
            write_data = json.load(f)
            write_data.extend(annotations)
    with open(f"{save_path}/annotations.json", "w") as f:
        json.dump(write_data, f, indent=4)


def save_augmented_images(obj_img, settings, save_path, frame_count, x, w, y, h, frame, class_name, annotations):
    global images_processed
    augmented_images = apply_augmentations(obj_img, settings)

    for i, img in enumerate(augmented_images):
        images_processed += 1
        img_name = (f"{save_path}/img_{frame_count:04d}_{i}"
                    f"{'_' + str(uuid.uuid4().hex) if not settings['rewrite'] else ''}.jpg")
        cv2.imwrite(img_name, img)
        img_height, img_width, _ = frame.shape
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        annotations.append(
            {"image": img_name, "bbox": [x_center, y_center, w_norm, h_norm], "class": class_name})

        if images_processed >= settings['stop_at'] and settings['stop_at'] != 0:
            cv2.destroyAllWindows()
            print(f"Зупинено на {images_processed} зображеннях.")
            quit(0)


def apply_augmentations(img, settings):
    augmented_images = [img]
    if settings.get("rotate", False):
        angle = random.randint(-15, 15)
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        augmented_images.append(rotated)
    if settings.get("flip", False):
        flipped = cv2.flip(img, 1)
        augmented_images.append(flipped)
    if settings.get("blur", False):
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        augmented_images.append(blurred)
    if settings.get("brightness", False):
        bright = cv2.convertScaleAbs(img, alpha=settings["brightness_level"], beta=30)
        augmented_images.append(bright)
    if settings.get("contrast", False):
        contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        augmented_images.append(contrast)
    if settings.get("noise", False):
        noise = np.random.randint(0, settings["noise_level"], img.shape, dtype='uint8')
        noisy = cv2.add(img, noise)
        augmented_images.append(noisy)
    return augmented_images


if __name__ == "__main__":
    # cv2.cuda.setDevice(0)
    cv2.setUseOptimized(True)
    cv2.CAP_PROP_BUFFERSIZE = 1
    video_file = select_video()
    selected_settings = select_settings()
    process_video(video_file, selected_settings)
