"""
LeNet 模型定义 - 适配 CIFAR-10 (32x32 RGB)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    LeNet 模型，适配 CIFAR-10 (32x32x3)
    
    结构 (改进版 + Dropout):
    - Conv1 (3->32, 5x5) -> BN -> ReLU -> MaxPool (2x2)
    - Conv2 (32->64, 5x5) -> BN -> ReLU -> MaxPool (2x2)
    - FC1 (64*5*5 -> 256) -> BN -> ReLU -> Dropout(0.5)
    - FC2 (256 -> 128) -> BN -> ReLU -> Dropout(0.5)
    - FC3 (128 -> 10)
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(LeNet, self).__init__()
        
        # 卷积层 (扩大卷积核数量: 6->16 改为 32->64)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 全连接层 (调整以适应新的特征图大小)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout 层（防止过拟合）
        self.dropout = nn.Dropout(dropout_rate)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化 - Kaiming 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1 -> BN -> ReLU -> Pool
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        # Conv2 -> BN -> ReLU -> Pool
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        # Flatten
        x = x.view(x.size(0), -1)
        # FC layers with BN and Dropout
        x = self.dropout(F.relu(self.bn3(self.fc1(x))))
        x = self.dropout(F.relu(self.bn4(self.fc2(x))))
        x = self.fc3(x)
        return x


def create_model(num_classes: int = 10, device: str = "cuda") -> nn.Module:
    """创建模型并移动到指定设备"""
    model = LeNet(num_classes=num_classes)
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
