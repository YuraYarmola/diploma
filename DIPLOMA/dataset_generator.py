import cv2
import os
import random
import numpy as np
import json
from datetime import datetime
from torchvision import transforms
from tkinter import Tk, filedialog, simpledialog
import tkinter as tk

# --- ФУНКЦІЯ ДЛЯ ВИБОРУ ВІДЕО ---
def select_video():
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(title="Виберіть відеофайл",
                                            filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if not video_path:
        print("Відео не вибрано.")
        return None
    return video_path

# --- ФУНКЦІЯ ДЛЯ ВИБОРУ ФІЛЬТРІВ ---
def select_filters():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    filters = {}

    def on_submit():
        filters["rotate"] = rotate_var.get()
        filters["flip"] = flip_var.get()
        filters["blur"] = blur_var.get()
        filters["brightness"] = brightness_var.get()
        filters["contrast"] = contrast_var.get()
        filters["noise"] = noise_var.get()
        filters["noise_level"] = noise_level_var.get()
        filters["brightness_level"] = brightness_level_var.get()
        root.quit()
        root.destroy()

    # Create a new window for the checkboxes
    filter_window = tk.Toplevel(root)
    filter_window.title("Вибір фільтрів")

    rotate_var = tk.BooleanVar()
    flip_var = tk.BooleanVar()
    blur_var = tk.BooleanVar()
    brightness_var = tk.BooleanVar()
    contrast_var = tk.BooleanVar()
    noise_var = tk.BooleanVar()
    noise_level_var = tk.DoubleVar(value=0.1)
    brightness_level_var = tk.DoubleVar(value=1.2)

    tk.Checkbutton(filter_window, text="Застосовувати повороти?", variable=rotate_var).pack(anchor='w')
    tk.Checkbutton(filter_window, text="Застосовувати дзеркальне відображення?", variable=flip_var).pack(anchor='w')
    tk.Checkbutton(filter_window, text="Застосовувати розмиття?", variable=blur_var).pack(anchor='w')
    tk.Checkbutton(filter_window, text="Змінювати яскравість?", variable=brightness_var).pack(anchor='w')
    tk.Scale(filter_window, label="Рівень яскравості", variable=brightness_level_var, from_=0.5, to=2.0, resolution=0.1, orient='horizontal').pack(anchor='w')
    tk.Checkbutton(filter_window, text="Змінювати контраст?", variable=contrast_var).pack(anchor='w')
    tk.Checkbutton(filter_window, text="Додавати шум?", variable=noise_var).pack(anchor='w')
    tk.Scale(filter_window, label="Рівень шуму", variable=noise_level_var, from_=0, to=50, resolution=1, orient='horizontal').pack(anchor='w')

    tk.Button(filter_window, text="Підтвердити", command=on_submit).pack()

    root.mainloop()

    return filters

# --- ФУНКЦІЯ ДЛЯ ОБРОБКИ ЗОБРАЖЕНЬ ---
def apply_augmentations(img, filters):
    augmented_images = [img]
    if filters["rotate"]:
        angle = random.randint(-15, 15)
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        augmented_images.append(rotated)
    if filters["flip"]:
        flipped = cv2.flip(img, 1)
        augmented_images.append(flipped)
    if filters["blur"]:
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        augmented_images.append(blurred)
    if filters["brightness"]:
        bright = cv2.convertScaleAbs(img, alpha=filters["brightness_level"], beta=30)
        augmented_images.append(bright)
    if filters["contrast"]:
        contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        augmented_images.append(contrast)
    if filters["noise"]:
        noise = np.random.randint(0, filters["noise_level"], img.shape, dtype='uint8')
        noisy = cv2.add(img, noise)
        augmented_images.append(noisy)
    return augmented_images

# --- ГОЛОВНА ФУНКЦІЯ ---
def process_video(video_path, filters):
    if not video_path:
        return

    cap = cv2.VideoCapture(video_path)
    class_name = simpledialog.askstring("Клас об'єкта", "Введіть назву класу об'єкта (напр., car, person)")
    save_path = f"dataset/{class_name}"
    os.makedirs(save_path, exist_ok=True)
    annotations = []

    # Вибір трекера
    tracker = cv2.TrackerCSRT_create()
    ret, frame = cap.read()
    bbox = cv2.selectROI("Виберіть об'єкт", frame, False)
    tracker.init(frame, bbox)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            obj_img = frame[y:y + h, x:x + w]
            obj_img = cv2.resize(obj_img, (128, 128))
            augmented_images = apply_augmentations(obj_img, filters)
            for i, img in enumerate(augmented_images):
                img_name = f"{save_path}/img_{frame_count:04d}_{i}.jpg"
                cv2.imwrite(img_name, img)
                img_height, img_width, _ = frame.shape
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                annotations.append({"image": img_name, "bbox": [x_center, y_center, w_norm, h_norm], "class": class_name})
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Трекінг об'єкта", frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

    # Збереження анотацій
    with open(f"{save_path}/annotations.json", "w") as f:
        json.dump(annotations, f, indent=4)

if __name__ == "__main__":
    video_file = select_video()
    selected_filters = select_filters()
    process_video(video_file, selected_filters)