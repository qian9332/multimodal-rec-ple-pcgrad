"""
损失函数模块
包含 Focal Loss、逆频率降权、难例挖掘等
"""
import math
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加父目录到路径
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import LossConfig


class FocalLoss(nn.Module):
    """
    Focal Loss
    用于处理类别不平衡问题
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 预测值 [batch_size]
            targets: 目标值 [batch_size]
        
        Returns:
            损失值
        """
        # 计算 BCE 损失
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算 p_t
        p_t = torch.exp(-bce_loss)
        
        # 计算 alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 计算 Focal Loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class InverseFrequencyWeightedLoss(nn.Module):
    """
    逆频率加权损失
    用于抑制高频样本主导
    """
    
    def __init__(self, smoothing: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.class_counts = None
    
    def update_class_counts(self, class_counts: Dict[int, int]):
        """更新类别计数"""
        self.class_counts = class_counts
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 预测值 [batch_size]
            targets: 目标值 [batch_size]
        
        Returns:
            损失值
        """
        # 计算基础损失
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算权重
        if self.class_counts is not None:
            total = sum(self.class_counts.values())
            weights = torch.zeros(2, device=inputs.device)
            
            for cls, count in self.class_counts.items():
                weights[cls] = total / (self.smoothing + count)
            
            # 归一化权重
            weights = weights / weights.sum()
            
            # 应用权重
            sample_weights = weights[targets.long()]
            bce_loss = bce_loss * sample_weights
        
        if self.reduction == 'mean':
            return bce_loss.mean()
        elif self.reduction == 'sum':
            return bce_loss.sum()
        else:
            return bce_loss


class HardNegativeMiningLoss(nn.Module):
    """
    难负例挖掘损失
    """
    
    def __init__(self, hard_negative_ratio: float = 0.3, reduction: str = 'mean'):
        super().__init__()
        self.hard_negative_ratio = hard_negative_ratio
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 预测值 [batch_size]
            targets: 目标值 [batch_size]
        
        Returns:
            损失值
        """
        # 计算所有样本的损失
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 分离正负样本
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        pos_loss = bce_loss[pos_mask]
        neg_loss = bce_loss[neg_mask]
        
        # 难负例挖掘：选择损失最大的负样本
        num_hard_negatives = int(len(neg_loss) * self.hard_negative_ratio)
        
        if num_hard_negatives > 0 and len(neg_loss) > num_hard_negatives:
            # 选择损失最大的负样本
            hard_neg_loss, _ = torch.topk(neg_loss, num_hard_negatives)
        else:
            hard_neg_loss = neg_loss
        
        # 合并正样本损失和难负例损失
        total_loss = torch.cat([pos_loss, hard_neg_loss])
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


class RDropLoss(nn.Module):
    """
    R-Drop 正则化损失
    通过一致性正则化增强泛化能力
    """
    
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, 
                pred1: torch.Tensor, 
                pred2: torch.Tensor) -> torch.Tensor:
        """
        计算两次预测之间的 KL 散度
        
        Args:
            pred1: 第一次预测 [batch_size]
            pred2: 第二次预测 [batch_size]
        
        Returns:
            R-Drop 损失
        """
        # 转换为概率
        p1 = torch.sigmoid(pred1)
        p2 = torch.sigmoid(pred2)
        
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


class MultimodalLoss(nn.Module):
    """
    多模态综合损失
    """
    
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        
        # 主损失函数
        if config.task_loss_type == "focal":
            self.main_loss = FocalLoss(
                alpha=config.focal_alpha,
                gamma=config.focal_gamma
            )
        else:
            self.main_loss = nn.BCEWithLogitsLoss()
        
        # 逆频率加权
        if config.use_inverse_freq_weight:
            self.freq_weighted_loss = InverseFrequencyWeightedLoss(
                smoothing=config.freq_weight_smoothing
            )
        else:
            self.freq_weighted_loss = None
        
        # 难例挖掘
        if config.use_hard_negative_mining:
            self.hard_negative_loss = HardNegativeMiningLoss(
                hard_negative_ratio=config.hard_negative_ratio
            )
        else:
            self.hard_negative_loss = None
        
        # R-Drop
        self.rdrop_loss = RDropLoss(alpha=0.1)
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: torch.Tensor,
                class_counts: Dict[int, int] = None) -> Dict[str, torch.Tensor]:
        """
        计算综合损失
        
        Args:
            predictions: 预测字典，包含 image_pred, text_pred, final_pred
            targets: 目标值
            class_counts: 类别计数（用于逆频率加权）
        
        Returns:
            损失字典
        """
        losses = {}
        
        # 更新类别计数
        if self.freq_weighted_loss is not None and class_counts is not None:
            self.freq_weighted_loss.update_class_counts(class_counts)
        
        # 图像任务损失
        image_pred = predictions["image_pred"]
        if self.hard_negative_loss is not None:
            losses["image_loss"] = self.hard_negative_loss(image_pred, targets)
        else:
            losses["image_loss"] = self.main_loss(image_pred, targets)
        
        # 文本任务损失
        text_pred = predictions["text_pred"]
        if self.hard_negative_loss is not None:
            losses["text_loss"] = self.hard_negative_loss(text_pred, targets)
        else:
            losses["text_loss"] = self.main_loss(text_pred, targets)
        
        # 最终预测损失
        final_pred = predictions["final_pred"]
        losses["final_loss"] = self.main_loss(final_pred, targets)
        
        # 稀疏度损失
        losses["sparsity_loss"] = (
            predictions["image_sparsity_loss"] + predictions["text_sparsity_loss"]
        ) * self.config.gate_sparsity_lambda if hasattr(self.config, 'gate_sparsity_lambda') else 0
        
        # 总损失
        losses["total_loss"] = (
            losses["image_loss"] + 
            losses["text_loss"] + 
            losses["final_loss"] + 
            losses["sparsity_loss"]
        )
        
        return losses


class MLMAuxiliaryLoss(nn.Module):
    """
    MLM 辅助任务损失
    用于文本侧的自监督学习
    """
    
    def __init__(self, vocab_size: int = 30522, mlm_probability: float = 0.15):
        super().__init__()
        self.vocab_size = vocab_size
        self.mlm_probability = mlm_probability
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, 
                masked_tokens: torch.Tensor,
                predictions: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        计算 MLM 损失
        
        Args:
            masked_tokens: 被掩盖的真实token [batch_size, seq_len]
            predictions: 模型预测 [batch_size, seq_len, vocab_size]
            mask: 掩盖位置掩码 [batch_size, seq_len]
        
        Returns:
            MLM 损失
        """
        # 只计算被掩盖位置的损失
        masked_tokens = masked_tokens[mask]
        predictions = predictions[mask]
        
        if len(masked_tokens) == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        return self.ce_loss(predictions, masked_tokens)
