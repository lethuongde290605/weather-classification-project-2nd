import torch
import torch.optim as optim
import yaml
from src.data.dataloader import get_dataloaders
from src.models.model import WeatherCNN
from src.losses.loss import WeatherLoss
from trainer import Trainer  # Import Trainer mới

def main():
    # Load config từ YAML
    with open('configs/baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    root_dir = config['data']['root_dir']
    batch_size = config['data']['batch_size']
    lr = config['training']['lr']
    epochs = config['training']['epochs']
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    checkpoint_path = config['checkpoint_path']
    
    # Load data
    train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(root_dir, batch_size)
    print(f"Loaded {num_classes} classes: {class_names}")
    
    # Model, loss, optimizer
    model = WeatherCNN(num_classes).to(device)
    criterion = WeatherLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Init Trainer và train
    trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, test_loader, device, class_names, epochs, checkpoint_path)
    trainer.train()

if __name__ == "__main__":
    main()