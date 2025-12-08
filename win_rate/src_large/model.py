import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class ValueNetLarge(nn.Module):
    def __init__(self, num_blocks=10, channels=128, input_planes=2):
        super(ValueNetLarge, self).__init__()
        
        # Initial convolution
        # Input shape: [Batch, 2, 19, 19] -> [Batch, 128, 19, 19]
        self.conv_input = nn.Conv2d(input_planes, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)
        
        # Residual Tower
        self.blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_blocks)
        ])
        
        # Value Head
        # Conv 1x1 -> 1 Channel
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        
        # Fully Connected
        self.fc1 = nn.Linear(19 * 19, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Input Block
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x)
        
        # Residual Tower
        for block in self.blocks:
            x = block(x)
            
        # Value Head
        x = self.value_conv(x) # [B, 1, 19, 19]
        x = self.value_bn(x)
        x = F.relu(x)
        
        x = x.view(-1, 19 * 19)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        # Output is Logit (Raw Score), not Probability.
        # Use BCEWithLogitsLoss during training, or Sigmoid for inference.
        return torch.sigmoid(x)
