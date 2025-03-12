import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms, models

# --- Hyperparameters ---
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
ALPHA = 0.7  # Weight for distillation loss
TEMPERATURE = 2.0
NUM_CLASSES = 11  # 10 classes + new class
NEW_CLASS_NAME = "bicycle"
DATA_DIR = "./data"
PRETRAINED_MODEL_PATH = "resnet50_trained3.pth"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-100 dataset
def load_cifar100(new_class_name):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    ])
    cifar100_train = datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=transform)
    cifar100_test = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=transform)
    class_idx = cifar100_train.classes.index(new_class_name)
    train_indices = [i for i, label in enumerate(cifar100_train.targets) if label == class_idx]
    test_indices = [i for i, label in enumerate(cifar100_test.targets) if label == class_idx]
    train_dataset = Subset(cifar100_train, train_indices)
    test_dataset = Subset(cifar100_test, test_indices)
    return train_dataset, test_dataset

# Initialize model
def initialize_model(pretrained_path, num_classes):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load the pre-trained model
    state_dict = torch.load(pretrained_path, map_location=device)

    # Remove the final layer's weights from the state dictionary
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)

    # Load the state dictionary into the model
    model.load_state_dict(state_dict, strict=False)

    return model.to(device)

# Generate pseudo-labels
def generate_pseudo_labels(model, dataloader):
    model.eval()
    pseudo_labels = []
    inputs_list = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            inputs_list.append(inputs.cpu())
            pseudo_labels.append(outputs.cpu())
    return torch.cat(inputs_list), torch.cat(pseudo_labels)

# Distillation loss
def distillation_loss(outputs, old_outputs, temperature):
    soft_labels = torch.nn.functional.softmax(old_outputs / temperature, dim=1)
    soft_predictions = torch.nn.functional.log_softmax(outputs / temperature, dim=1)
    loss = torch.nn.functional.kl_div(soft_predictions, soft_labels, reduction="batchmean") * (temperature ** 2)
    return loss

# Weighted CrossEntropyLoss
def weighted_cross_entropy_loss(outputs, labels, class_weights):
    return nn.CrossEntropyLoss(weight=class_weights.to(device))(outputs, labels)

# Training with combined old and new data
def train_model_with_distillation(model, new_loader, old_loader, criterion, optimizer, alpha, temperature, num_epochs, class_weights):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for (new_inputs, new_labels), (old_inputs, old_logits) in zip(new_loader, old_loader):
            new_inputs, new_labels = new_inputs.to(device), new_labels.to(device)
            old_inputs, old_logits = old_inputs.to(device), old_logits.to(device)

            optimizer.zero_grad()

            # Forward pass for new data
            new_outputs = model(new_inputs)
            loss_new = weighted_cross_entropy_loss(new_outputs, new_labels, class_weights)

            # Forward pass for old data
            old_outputs = model(old_inputs)
            loss_old = distillation_loss(old_outputs, old_logits, temperature)

            # Combine losses
            loss = (1 - alpha) * loss_new + alpha * loss_old
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}")
    return model

# Main function
if __name__ == "__main__":
    print("Loading new class...")
    train_dataset, test_dataset = load_cifar100(NEW_CLASS_NAME)
    for i in range(len(train_dataset)):
        train_dataset.dataset.targets[train_dataset.indices[i]] = 10
    for i in range(len(test_dataset)):
        test_dataset.dataset.targets[test_dataset.indices[i]] = 10
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Initializing model...")
    model = initialize_model(pretrained_path=PRETRAINED_MODEL_PATH, num_classes=NUM_CLASSES)

    # Freeze old model for generating pseudo-labels
    old_model = initialize_model(pretrained_path=PRETRAINED_MODEL_PATH, num_classes=NUM_CLASSES)
    old_model.eval()

    print("Generating pseudo-labels...")
    old_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    old_inputs, pseudo_labels = generate_pseudo_labels(old_model, old_data_loader)
    pseudo_dataset = TensorDataset(old_inputs, pseudo_labels)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Compute class weights to balance the loss
    new_class_targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
    class_counts = torch.bincount(torch.tensor(new_class_targets), minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts.float() + 1e-8)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("Training model with distillation...")
    model = train_model_with_distillation(
        model, train_loader, pseudo_loader, criterion, optimizer, ALPHA, TEMPERATURE, EPOCHS, class_weights
    )

    torch.save(model.state_dict(), "resnet50_finetuned.pth")
    print("Model saved as resnet50_finetuned.pth")
