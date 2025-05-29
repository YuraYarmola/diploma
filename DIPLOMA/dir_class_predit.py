import time

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision import models
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from functools import wraps


def evaluate_time_stats_per_func():
    stats_dict = {}

    def decorator(func):
        stats = stats_dict.setdefault(func.__name__, {'calls': 0, 'total_time': 0.0, 'max_time': 0.0, 'min_time': 0.0})

        @wraps(func)
        def wrapper(*args, **kwargs):
            stats['calls'] += 1
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            stats['max_time'] = max(stats['max_time'], elapsed)
            stats['min_time'] = min(stats['min_time'], elapsed) if stats['calls'] > 1 else elapsed
            stats['total_time'] += elapsed
            avg_time = stats['total_time'] / stats['calls']
            print(f"[{func.__name__}] Call #{stats['calls']}: Time = {elapsed:.4f}s | Avg = {avg_time:.4f}s | Total = {stats['total_time']:.4f}s")
            return result
        return wrapper

    decorator.stats_dict = stats_dict
    return decorator

time_decorator = evaluate_time_stats_per_func()

@time_decorator
def load_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    return checkpoint, checkpoint['metadata']

@time_decorator
def build_model(metadata, num_classes):
    model = getattr(models, metadata.get('model_type', 'resnet50'))(pretrained=False)
    if hasattr(model, "classifier"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model

@time_decorator
def get_transform(metadata):
    return transforms.Compose([
        transforms.Resize((metadata.get("img_size", 32), metadata.get("img_size", 32))),
        transforms.ToTensor(),
        transforms.Normalize(mean=metadata.get("normalization_mean"),
                             std=metadata.get("normalization_std"))
    ])

@time_decorator
def predict_image(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(device))
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

@time_decorator
def evaluate_folder(model, image_folder, class_names, transform, device):
    y_true = []
    y_pred = []
    actual_class = class_names.index(os.path.basename(image_folder))
    for filename in os.listdir(image_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)
            predicted_class = predict_image(model, image_tensor, device)
            y_true.append(actual_class)
            y_pred.append(predicted_class)
            print(
                f"{filename}: Передбачено - {class_names[predicted_class]}, Реальність - {class_names[actual_class]}, {'✅' if predicted_class == actual_class else '❌'}"
            )
    accuracy = (np.array(y_true) == np.array(y_pred)).mean() * 100 if len(y_true) > 0 else 0
    print(f"\nЗагальна точність: {accuracy:.2f}%")

    # Друк матриці неточностей та метрик
    if y_true:
        print("\nМатриця неточностей:")
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        print(cm)
        print("\nЗвіт про класифікацію:")
        print(classification_report(
            y_true, y_pred, target_names=class_names, labels=range(len(class_names)), zero_division=0
        ))

    return accuracy

@time_decorator
def main(model_path, image_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, metadata = load_checkpoint(model_path, device)
    class_names = metadata['class_names']
    num_classes = len(class_names)
    model = build_model(metadata, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    transform = get_transform(metadata)
    return evaluate_folder(model, image_folder, class_names, transform, device)


if __name__ == "__main__":
    MODEL_PATH = "mobilenet_v3_small_custom_with_metadata.pth"
    IMAGE_FOLDER = "dataset128/dog"
    main(MODEL_PATH, IMAGE_FOLDER)

    print("\n--- Summary ---")
    for func, stats in time_decorator.stats_dict.items():
        avg = stats['total_time'] / stats['calls'] if stats['calls'] else 0
        print(f"{func}: calls={stats['calls']}, total={stats['total_time']:.4f}s, avg={avg:.4f}s, max={stats['max_time']:.4f}s, min={stats['min_time']:.4f}s")