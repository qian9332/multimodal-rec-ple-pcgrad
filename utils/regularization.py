"""
正则化模块
包含 DropBlock、Early Stop、Label Smoothing、R-Drop 等
"""
import math
from typing import Dict, List, Tuple, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加父目录到路径
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import RegularizationConfig


class DropBlock2D(nn.Module):
    """
    DropBlock 正则化
    在特征图上丢弃连续区域，比 Dropout 更适合卷积网络
    """
    
    def __init__(self, block_size: int = 7, drop_prob: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图 [batch_size, channels, height, width]
        
        Returns:
            正则化后的特征图
        """
        if not self.training or self.drop_prob == 0:
            return x
        
        gamma = self._compute_gamma(x)
        
        # 生成随机掩码
        mask = (torch.rand_like(x[:, :1, :, :]) < gamma).float()
        
        # 扩展掩码到块大小
        mask = F.max_pool2d(
            mask,
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=(self.block_size // 2, self.block_size // 2)
        )
        
        mask = 1 - mask
        
        # 归一化
        out = x * mask
        out = out * mask.numel() / (mask.sum() + 1e-8)
        
        return out
    
    def _compute_gamma(self, x: torch.Tensor) -> float:
        """计算 gamma 值"""
        return self.drop_prob / (self.block_size ** 2)


class EarlyStopping:
    """
    Early Stopping
    当验证损失不再改善时停止训练
    """
    
    def __init__(self, 
                 patience: int = 5, 
                 min_delta: float = 0.001,
                 mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该停止
        
        Args:
            score: 当前验证分数
        
        Returns:
            是否应该停止
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        """重置状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing 损失
    防止模型过度自信
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 预测值 [batch_size]
            targets: 目标值 [batch_size]
        
        Returns:
            平滑后的损失
        """
        # 平滑标签
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # 计算 BCE 损失
        loss = F.binary_cross_entropy_with_logits(inputs, targets_smooth)
        
        return loss


class RDropRegularization(nn.Module):
    """
    R-Drop 正则化
    通过一致性正则化增强泛化能力
    """
    
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, 
                output1: torch.Tensor, 
                output2: torch.Tensor) -> torch.Tensor:
        """
        计算两次输出的 KL 散度
        
        Args:
            output1: 第一次前向传播输出
            output2: 第二次前向传播输出
        
        Returns:
            R-Drop 正则化损失
        """
        # 转换为概率
        p1 = torch.sigmoid(output1)
        p2 = torch.sigmoid(output2)
        
        # 双向 KL 散度
        kl_1_2 = F.kl_div(
            torch.log(p1 + 1e-8), 
            p2, 
            reduction='batchmean'
        )
        kl_2_1 = F.kl_div(
            torch.log(p2 + 1e-8), 
            p1, 
            reduction='batchmean'
        )
        
        return self.alpha * (kl_1_2 + kl_2_1) / 2


class DropBlockResNet(nn.Module):
    """
    带有 DropBlock 的残差块
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1,
                 dropblock: Optional[DropBlock2D] = None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropblock = dropblock
        
        # 下采样
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        
        if self.dropblock is not None:
            out = self.dropblock(out)
        
        out = self.bn2(self.conv2(out))
        
        if self.dropblock is not None:
            out = self.dropblock(out)
        
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


class WeightDecayRegularizer:
    """
    权重衰减正则化
    """
    
    def __init__(self, weight_decay: float = 1e-4, norm_type: int = 2):
        self.weight_decay = weight_decay
        self.norm_type = norm_type
    
    def __call__(self, model: nn.Module) -> torch.Tensor:
        """
        计算权重衰减损失
        
        Args:
            model: 模型
        
        Returns:
            权重衰减损失
        """
        if self.weight_decay == 0:
            return torch.tensor(0.0)
        
        reg_loss = torch.tensor(0.0)
        for param in model.parameters():
            if param.requires_grad:
                reg_loss += torch.norm(param, p=self.norm_type)
        
        return self.weight_decay * reg_loss


class GradientNoise:
    """
    梯度噪声
    在梯度中添加噪声以增强泛化能力
    """
    
    def __init__(self, noise_factor: float = 0.1):
        self.noise_factor = noise_factor
    
    def __call__(self, model: nn.Module):
        """
        在梯度中添加噪声
        
        Args:
            model: 模型
        """
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.noise_factor
                param.grad += noise


def create_regularization_layers(config: RegularizationConfig) -> Dict[str, nn.Module]:
    """
    创建正则化层
    
    Args:
        config: 正则化配置
    
    Returns:
        正则化层字典
    """
    layers = {}
    
    if config.use_dropblock:
        layers['dropblock'] = DropBlock2D(
            block_size=config.dropblock_block_size,
            drop_prob=config.dropblock_prob
        )
    
    layers['label_smoothing'] = LabelSmoothingLoss(
        smoothing=config.label_smoothing
    )
    
    if config.use_rdrop:
        layers['rdrop'] = RDropRegularization(
            alpha=config.rdrop_alpha
        )
    
    layers['weight_decay'] = WeightDecayRegularizer(
        weight_decay=config.weight_decay
    )
    
    return layers


def create_early_stopping(config: RegularizationConfig) -> EarlyStopping:
    """创建 Early Stopping"""
    return EarlyStopping(
        patience=config.early_stop_patience,
        min_delta=config.early_stop_min_delta
    )
