import torch
import torch.nn as nn

class ConvMLP(nn.Module):
    def __init__(self, input_size=256):
        super(ConvMLP, self).__init__()
        
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            sample_input = torch.randn(1, 1, input_size)
            self.flat_features = self.audio_encoder(sample_input).shape[1]

        self.mlp_layers = nn.Sequential(
            nn.Linear(self.flat_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.audio_encoder(x)
        x = self.mlp_layers(x)
        return x
