import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision import models
import torch.nn as nn

def load_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    return checkpoint, checkpoint['metadata']

def build_model(metadata, num_classes):
    model = getattr(models, metadata.get('model_type', 'resnet50'))(pretrained=False)
    if hasattr(model, "classifier"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model

def get_transform(metadata):
    return transforms.Compose([
        transforms.Resize((metadata.get("img_size", 32), metadata.get("img_size", 32))),
        transforms.ToTensor(),
        transforms.Normalize(mean=metadata.get("normalization_mean"),
                             std=metadata.get("normalization_std"))
    ])

def predict_image(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(device))
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

def get_actual_class(class_names, image_folder):
    # Assumes folder name is the class
    return class_names.index(os.path.basename(image_folder))

def evaluate_folder(model, image_folder, class_names, transform, device):
    correct = 0
    total = 0
    actual_class = get_actual_class(class_names, image_folder)
    for filename in os.listdir(image_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)
            predicted_class = predict_image(model, image_tensor, device)
            is_correct = predicted_class == actual_class
            total += 1
            correct += int(is_correct)
            print(
                f"{filename}: Передбачено - {class_names[predicted_class]}, Реальність - {class_names[actual_class]}, {'✅' if is_correct else '❌'}"
            )
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nЗагальна точність: {accuracy:.2f}%")
    return accuracy

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