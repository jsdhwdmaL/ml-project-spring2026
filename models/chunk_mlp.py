import torch
import torch.nn as nn
from torchvision import models


class BehavioralCloningPolicy(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=256, n_frames=2, horizon=50):
        super().__init__()
        self.n_frames = n_frames
        self.action_dim = action_dim
        self.horizon = horizon
        
        # Load Pre-trained ResNet18
        resnet = models.resnet18(weights="DEFAULT")
        
        # Modify the first layer to accept stacked frames (6 channels instead of 3)
        self.input_channels = 3 * n_frames
        resnet.conv1 = nn.Conv2d(
            self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        vision_feature_dim = 512 
        
        # Encoder for stacked states (4 dims if n_frames=2)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim * n_frames, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(vision_feature_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim)
        )

    def forward(self, image, state):
        # image shape: (B, 6, 96, 96) | state shape: (B, 4)
        img_features = self.vision_backbone(image).view(image.size(0), -1) 
        state_features = self.state_encoder(state) 
        combined_features = torch.cat([img_features, state_features], dim=1) 
        pred = self.action_head(combined_features)
        return pred.view(image.size(0), self.horizon, self.action_dim)