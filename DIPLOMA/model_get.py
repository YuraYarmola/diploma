import torchvision.models as models
import torch.nn as nn


def get_model(model_name, num_classes):
    model = getattr(models, model_name)(weights=None)  # ⚡ Завантажуємо попередньо натреновану модель

    # 🔹 Замінюємо класифікатор під нашу кількість класів
    if hasattr(model, "classifier"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model