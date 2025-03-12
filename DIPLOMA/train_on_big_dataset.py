import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# --- Гіперпараметри ---
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_CLASSES = 10  # ImageNet (1000 класів), змініть на 10 для CIFAR-10
DATA_DIR = "./data"  # Локальна директорія для датасету
IMAGE_SIZE = 32  # ImageNet зазвичай використовує 224x224

# Визначаємо пристрій
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Підготовка датасетів ---
def create_dataloaders(image_size=IMAGE_SIZE, dataset="imagenet"):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    if dataset == "imagenet":
        dataset_train = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
        dataset_val = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
    elif dataset == "cifar10":
        dataset_train = datasets.CIFAR10(DATA_DIR, train=True, transform=transform, download=True)
        dataset_val = datasets.CIFAR10(DATA_DIR, train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader, len(dataset_train.classes)


# --- Ініціалізація моделі ---
def initialize_model(num_classes=NUM_CLASSES):
    model = models.resnet50(pretrained=True)

    # Розморожуємо всі шари
    for param in model.parameters():
        param.requires_grad = True

    # Оновлюємо останній шар
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


# --- Функція навчання ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)


        epoch_loss = running_loss / len(train_loader.dataset)
        accuracy = correct / total * 100
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return model


if __name__ == "__main__":
    train_loader, val_loader, num_classes = create_dataloaders(dataset="cifar10")  # або "cifar10"
    model = initialize_model(num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

    print("Навчання моделі...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer)

    torch.save(model.state_dict(), "resnet50_trained2.pth")
    print("Модель збережена як resnet50_trained.pth")
