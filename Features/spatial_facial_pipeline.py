import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from typing import Tuple, Dict, Any

# ==========================================
# 1. ADVANCED BUILDING BLOCKS (SE + Residual)
# ==========================================

class SqueezeExcite2D(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super(SqueezeExcite2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpecializedSpatialBlock(nn.Module):
    """
    Custom residual block supporting asymmetric kernels for wide regions (mouth)
    and fine-grained isotropic kernels for localized regions (eyes).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Any = 3, stride: int = 1, use_se: bool = True):
        super(SpecializedSpatialBlock, self).__init__()
        
        # Handle tuple or int padding/kernel sizes
        if isinstance(kernel_size, tuple):
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.use_se = use_se
        if self.use_se:
            self.se = SqueezeExcite2D(out_channels)
            
        # Shortcut connection for residual learning
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        
        if self.use_se:
            out = self.se(out)
            
        out += residual
        out = F.relu(out, inplace=True)
        return out


# ==========================================
# 2. FEATURE EXTRACTORS & DUAL-STREAM MODEL
# ==========================================

class SpecializedFeatureExtractor(nn.Module):
    def __init__(self, crop_type: str = "eye"):
        super(SpecializedFeatureExtractor, self).__init__()
        self.crop_type = crop_type.lower()
        
        if self.crop_type == "eye":
            # Eyes need high-resolution fine detail capture
            self.initial_conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            self.block1 = SpecializedSpatialBlock(32, 64, kernel_size=3, stride=2)   # e.g., 32x32 -> 16x16
            self.block2 = SpecializedSpatialBlock(64, 128, kernel_size=3, stride=2)  # 16x16 -> 8x8
            self.block3 = SpecializedSpatialBlock(128, 256, kernel_size=3, stride=2) # 8x8 -> 4x4
            
        elif self.crop_type == "mouth":
            # Mouths are wider horizontally; initial asymmetric filter captures lip line shape
            self.initial_conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            self.block1 = SpecializedSpatialBlock(32, 64, kernel_size=(3, 3), stride=2)
            self.block2 = SpecializedSpatialBlock(64, 128, kernel_size=(3, 3), stride=2)
            self.block3 = SpecializedSpatialBlock(128, 256, kernel_size=(3, 3), stride=2)
        else:
            raise ValueError("crop_type must be 'eye' or 'mouth'")

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return x


class SpatialFacialFusionModel(nn.Module):
    """
    Fuses left eye, right eye, and mouth feature representations 
    for downstream classification (e.g., Drowsiness, Engagement, Emotion).
    """
    def __init__(self, num_classes: int = 2):
        super(SpatialFacialFusionModel, self).__init__()
        
        self.left_eye_net = SpecializedFeatureExtractor(crop_type="eye")
        self.right_eye_net = SpecializedFeatureExtractor(crop_type="eye")
        self.mouth_net = SpecializedFeatureExtractor(crop_type="mouth")
        
        # 256 (left) + 256 (right) + 256 (mouth) = 768 total channels
        combined_dim = 256 * 3
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, left_eye: torch.Tensor, right_eye: torch.Tensor, mouth: torch.Tensor) -> torch.Tensor:
        feat_l = self.left_eye_net(left_eye)
        feat_r = self.right_eye_net(right_eye)
        feat_m = self.mouth_net(mouth)
        
        concatenated = torch.cat((feat_l, feat_r, feat_m), dim=1)
        logits = self.classifier(concatenated)
        return logits


# ==========================================
# 3. CUSTOM DATASET & PIPELINE UTILITIES
# ==========================================

class FacialCropsDataset(Dataset):
    """
    Expects data samples structured to yield crops for left eye, right eye, mouth, and a target label.
    """
    def __init__(self, data_records: list, transform: Any = None):
        self.records = data_records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        
        # Mock tensors or load from disk / image arrays using PIL / OpenCV
        left_eye = record["left_eye"]   # Shape: [3, H, W]
        right_eye = record["right_eye"] # Shape: [3, H, W]
        mouth = record["mouth"]         # Shape: [3, H, W]
        label = record["label"]         # Int scalar

        if self.transform:
            left_eye = self.transform(left_eye)
            right_eye = self.transform(right_eye)
            mouth = self.transform(mouth)

        return left_eye, right_eye, mouth, torch.tensor(label, dtype=torch.long)


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for left_eye, right_eye, mouth, labels in dataloader:
        left_eye = left_eye.to(device)
        right_eye = right_eye.to(device)
        mouth = mouth.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(left_eye, right_eye, mouth)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * left_eye.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ==========================================
# 4. EXECUTION & TRAINING DEMO
# ==========================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate dummy records to test training mechanics
    dummy_records = []
    for _ in range(100):
        dummy_records.append({
            "left_eye": torch.randn(3, 32, 32),
            "right_eye": torch.randn(3, 32, 32),
            "mouth": torch.randn(3, 32, 64),
            "label": torch.randint(0, 2, (1,)).item()
        })

    transform = T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2),
    ])

    dataset = FacialCropsDataset(dummy_records, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model, loss, and optimizer
    model = SpatialFacialFusionModel(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Quick Training Loop Smoke Test
    print("Starting smoke test training epoch...")
    loss, acc = train_one_epoch(model, dataloader, criterion, optimizer, device)
    print(f"Epoch Complete! Loss: {loss:.4f} | Accuracy: {acc * 100:.2f}%")
