import torch
from utils import save_checkpoint, load_checkpoint, evaluate

class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, test_loader, device, class_names, epochs, checkpoint_path):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.start_epoch = self.load_checkpoint()
    
    def load_checkpoint(self):
        return load_checkpoint(self.model, self.optimizer, path=self.checkpoint_path)
    
    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
            
            # Validate
            print("Validation metrics:")
            evaluate(self.model, self.val_loader, self.device, self.class_names)
            
            # Save checkpoint
            save_checkpoint(self.model, self.optimizer, epoch, path=self.checkpoint_path)
        
        # Test final model
        print("Test metrics:")
        evaluate(self.model, self.test_loader, self.device, self.class_names, is_test=True)