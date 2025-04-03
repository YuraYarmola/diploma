import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# --- Гіперпараметри ---
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
DATA_DIR = r"D:\LPNU\DIPLOMA\DIPLOMA\dataset"
IMAGE_SIZE = 128

# Визначаємо, чи доступний GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Підготовка датасетів ---
def create_dataloaders(image_size=IMAGE_SIZE):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader, len(dataset.classes)


# --- Ініціалізація моделі ---
def initialize_model(num_classes, feature_extract=True):
    model = models.resnet18(pretrained=True)  # Використання ResNet18
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False  # Заморожуємо всі параметри, крім останнього шару
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Оновлення останнього шару
    return model.to(device)


# --- Функція навчання ---
def train_model(model, dataloader, criterion, optimizer, num_epochs=EPOCHS):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            if isinstance(criterion, nn.BCEWithLogitsLoss):
                labels = labels.view(-1, 1).float()  # Для бінарної класифікації
            loss = criterion(outputs, labels)

            # Оновлення ваг
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Обчислення точності
            if isinstance(criterion, nn.CrossEntropyLoss):
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
            else:
                predicted = torch.sigmoid(outputs) > 0.5
                correct += (predicted.int() == labels.int()).sum().item()

            total += labels.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        accuracy = correct / total * 100
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return model


# --- Навчання на класі "dog" ---
def train_on_dog():
    dataloader, _ = create_dataloaders()
    model = initialize_model(num_classes=1)  # 1 клас: dog (бінарна класифікація)

    criterion = nn.BCEWithLogitsLoss()  # Бінарна класифікація
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    print("Навчання моделі на класі dog...")
    model = train_model(model, dataloader, criterion, optimizer)

    torch.save(model.state_dict(), "model_dog.pth")
    print("Модель збережена як model_dog.pth")
    return model


# --- Довчання на класі "cat" ---
def fine_tune_with_cat():
    dataloader, _ = create_dataloaders()
    model = models.resnet18(pretrained=True)  # Використовуємо попередньо навчений ResNet18

    # Завантажуємо збережену модель, але без останнього шару (fc)
    checkpoint = torch.load("model_dog.pth", map_location=device)
    model.load_state_dict({k: v for k, v in checkpoint.items() if "fc" not in k}, strict=False)

    # Створюємо новий повністю підключений шар для 2 класів
    model.fc = nn.Linear(model.fc.in_features, 2)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    print("Довчання моделі з класом cat...")
    model = train_model(model, dataloader, criterion, optimizer)

    torch.save(model.state_dict(), "model_dog_cat.pth")
    print("Модель збережена як model_dog_cat.pth")
    return model


if __name__ == "__main__":
    # Етап 1: Навчання на класі dog
    train_on_dog()

    # Етап 2: Довчання з класом cat
    fine_tune_with_cat()
