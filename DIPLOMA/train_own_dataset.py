import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datetime import datetime
from custom_dataset_loader import CustomDataset
from model_get import get_model
from normalizer import compute_mean_std
import numpy as np
import time

# --- Гіперпараметри ---
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
IMG_SIZE = 128
STOP_CRITERION = 99  # Валідаційна точність для зупинки навчання. -1 - не враховувати
MODEL_TYPE = "resnet50"
MODEL_PATH = f"{MODEL_TYPE}_custom_with_metadata.pth"
DATASET_PATH = r"D:\LPNU\DIPLOMA\DIPLOMA\dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- Початкові трансформації ---
transform = transforms.Compose([
    transforms.ToTensor()
])

def make_normalize_transform(dataset):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    if len(dataset) == 0:
        raise ValueError("❌ Помилка: датасет порожній. Переконайтеся, що дані коректні.")
    mean, std = compute_mean_std(dataloader)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform, mean, std

def train(dataset, mean, std, show_plot=False):
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2,
                                                  stratify=[dataset[idx][1].item() for idx in indices])
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=2)
    num_classes = len(dataset.class_to_idx)
    model = get_model(MODEL_TYPE, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def evaluate_model(model, dataloader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return correct / total * 100

    def plot_training_results(train_losses, train_accuracies, val_accuracies, epochs, save_path=f"training_plot {date}.png"):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), train_losses, label="Втрати (Train)", marker="o")
        plt.plot(range(1, epochs + 1), train_accuracies, label="Точність (Train)", marker="s")
        plt.plot(range(1, epochs + 1), val_accuracies, label="Точність (Validation)", marker="^")
        plt.xlabel("Епохи")
        plt.ylabel("Значення")
        plt.title("Динаміка навчання моделі")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()
        print(f"📊 Графік навчання збережено у {save_path}")

    def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
        train_losses, train_accuracies, val_accuracies = [], [], []
        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            train_losses.append(running_loss / len(train_loader))
            train_accuracies.append(correct / total * 100)
            val_acc = evaluate_model(model, val_loader)
            val_accuracies.append(val_acc)
            print(f"🌀 Епоха {epoch + 1}/{epochs} | Втрата: {train_losses[-1]:.4f} | "
                  f"Тренувальна точність: {train_accuracies[-1]:.2f}% | Валідаційна точність: {val_acc:.2f}%")
            if val_acc >= STOP_CRITERION and STOP_CRITERION != -1:
                print(f"🔴 Зупинка навчання: досягнуто критерію зупинки. Валідаційна точність {val_acc}")
                break
        if show_plot:
            plot_training_results(train_losses, train_accuracies, val_accuracies, len(train_losses))
        torch.save({
            "model_state_dict": model.state_dict(),
            "metadata": {
                "normalization_mean": list(mean),
                "normalization_std": list(std),
                "img_size": IMG_SIZE,
                "model_type": MODEL_TYPE,
                "class_names": list(dataset.class_to_idx.keys())
            }
        }, MODEL_PATH)
        print(f"✅ Модель збережена в {MODEL_PATH}")

    train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

def validate_model(model, dataloader, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    print(f"📊 Загальна точність: {accuracy:.2f}%")
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="viridis", xticks_rotation="vertical")
    plt.title("Матриця неточностей")
    plt.show()
    print(f"🔹 Звіт про класифікацію:\n{classification_report(all_labels, all_preds, target_names=class_names)}")

if __name__ == "__main__":
    dataset = CustomDataset(root_dir=DATASET_PATH, transform=transform)
    transform, mean, std = make_normalize_transform(dataset)
    dataset.transform = transform
    train(dataset, mean, std)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = get_model(MODEL_TYPE, len(checkpoint["metadata"]["class_names"])).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    validate_model(model, DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4),
                   checkpoint["metadata"]["class_names"])
