import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

class WeatherDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Sử dụng ImageFolder để load dữ liệu từ folders.
        root_dir: thư mục gốc chứa 11 sub-folders (mỗi folder là một class).
        """
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def get_dataloaders(root_dir, batch_size=32):
    # Transforms: resize ảnh về 224x224, normalize cho RGB
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = WeatherDataset(root_dir, transform=transform)
    num_samples = len(full_dataset)
    
    # Chia split: 70% train, 15% val, 15% test
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    # DataLoaders: không dùng collator (collate_fn=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=None)
    
    # Lấy class names (từ folders)
    class_names = full_dataset.dataset.classes
    num_classes = len(class_names)
    
    return train_loader, val_loader, test_loader, num_classes, class_names