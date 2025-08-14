import torch.nn as nn

class WeatherLoss(nn.Module):
    def __init__(self):
        super(WeatherLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()  # Cross Entropy cho classification
    
    def forward(self, outputs, labels):
        return self.criterion(outputs, labels)