import os
import zipfile
import urllib.request
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Tiny ImageNet-200 RGB Normalization Constants
RED_MEAN, GREEN_MEAN, BLUE_MEAN = 0.485, 0.456, 0.406
RED_STD, GREEN_STD, BLUE_STD = 0.229, 0.224, 0.225

def download_tiny_imagenet(data_dir: str = './data'):
    """
    Downloads and extracts Tiny ImageNet-200 dataset if not already present.

    Args:
        data_dir (str): Directory to store the dataset. Defaults to './data'.
    """
    dataset_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    extract_path = os.path.join(data_dir, "tiny-imagenet-200")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(extract_path):
        print(f"Downloading Tiny ImageNet-200 from {dataset_url}...")
        urllib.request.urlretrieve(dataset_url, zip_path)
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)
        print("Dataset ready.")
    else:
        print("Tiny ImageNet-200 already present.")

def get_dataloaders(data_dir: str = './data/tiny-imagenet-200', batch_size: int = 64, seed: int = 42):
    """
    Prepares dataloaders for Target training, Surrogate training, and Validation.

    Args:
        data_dir (str): Path to extracted dataset.
        batch_size (int): Batch size for dataloaders.
        seed (int): Random seed for reproducibility.
    Returns:
        tuple: (target_loader, surrogate_loader, val_loader)
    """
    train_dir = os.path.join(data_dir, 'train')
    
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[RED_MEAN, GREEN_MEAN, BLUE_MEAN],
            std=[RED_STD, GREEN_STD, BLUE_STD]
        )
    ])

    full_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    
    # Split into Target training (40k), Surrogate training (40k), and Validation (20k)
    # Note: Tiny ImageNet has 100k training images (500 per class x 200 classes)
    import torch
    generator = torch.Generator().manual_seed(seed)
    target_ds, surrogate_ds, val_ds = random_split(full_dataset, [40000, 40000, 20000], generator=generator)
    
    target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=True)
    surrogate_loader = DataLoader(surrogate_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return target_loader, surrogate_loader, val_loader

class Denormalize:
    """
    Inverse transformation for Normalize transform to allow plotting.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def get_inverse_transform():
    """
    Returns the Denormalize transform for Tiny ImageNet constants.

    Returns:
        Denormalize: Inverse normalization transform.
    """
    return Denormalize(
        mean=[RED_MEAN, GREEN_MEAN, BLUE_MEAN],
        std=[RED_STD, GREEN_STD, BLUE_STD]
    )
