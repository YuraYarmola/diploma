import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from custom_dataset_loader import CustomDataset


# Обчислюємо середнє та стандартне відхилення
def compute_mean_std(dataloader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0

    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, 3, -1)

        mean += images.mean(dim=[0, 2]) * batch_samples
        std += images.std(dim=[0, 2]) * batch_samples
        n_samples += batch_samples

    mean /= n_samples
    std /= n_samples
    return mean, std


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = CustomDataset(root_dir=r"D:\LPNU\DIPLOMA\DIPLOMA\dataset", transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    mean, std = compute_mean_std(dataloader)
    print(f"Mean: {mean}")
    print(f"Std: {std}")
