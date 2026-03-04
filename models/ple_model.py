"""
PLE (Progressive Layered Extraction) 多任务学习模型
实现图文梯度解耦结构
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
from configs.config import ModelConfig


class ExpertNetwork(nn.Module):
    """专家网络"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        current_size = input_size
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GateNetwork(nn.Module):
    """门控网络（带稀疏度约束）"""
    
    def __init__(self, input_size: int, num_experts: int, hidden_size: int = 128,
                 temperature: float = 1.0, top_k: int = 2):
        super().__init__()
        
        self.temperature = temperature
        self.top_k = top_k
        self.num_experts = num_experts
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入特征 [batch_size, input_size]
        
        Returns:
            gate_weights: 门控权重 [batch_size, num_experts]
            sparsity_loss: 稀疏度损失
        """
        logits = self.network(x)
        
        # Gumbel-Softmax 用于可微分的 Top-K 选择
        gate_weights = F.softmax(logits / self.temperature, dim=-1)
        
        # Top-K 稀疏化
        if self.top_k < self.num_experts:
            top_k_values, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
            
            # 创建稀疏掩码
            sparse_mask = torch.zeros_like(gate_weights)
            sparse_mask.scatter_(1, top_k_indices, 1.0)
            
            # 应用掩码并重新归一化
            gate_weights = gate_weights * sparse_mask
            gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 计算稀疏度损失（鼓励权重集中）
        sparsity_loss = -torch.mean(torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=-1))
        
        return gate_weights, sparsity_loss


class PLELayer(nn.Module):
    """
    PLE 单层结构
    包含共享专家和任务特定专家
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 num_shared_experts: int = 2,
                 num_task_experts: int = 2,
                 expert_hidden_size: int = 256,
                 gate_hidden_size: int = 128,
                 num_expert_layers: int = 2,
                 dropout: float = 0.2,
                 temperature: float = 1.0,
                 top_k: int = 2):
        super().__init__()
        
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        self.total_experts = num_shared_experts + num_task_experts
        
        # 共享专家
        self.shared_experts = nn.ModuleList([
            ExpertNetwork(input_size, expert_hidden_size, output_size, 
                         num_expert_layers, dropout)
            for _ in range(num_shared_experts)
        ])
        
        # 任务特定专家
        self.task_experts = nn.ModuleList([
            ExpertNetwork(input_size, expert_hidden_size, output_size, 
                         num_expert_layers, dropout)
            for _ in range(num_task_experts)
        ])
        
        # 门控网络
        self.gate = GateNetwork(
            input_size, 
            self.total_experts, 
            gate_hidden_size,
            temperature, 
            top_k
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入特征 [batch_size, input_size]
        
        Returns:
            output: 输出特征 [batch_size, output_size]
            gate_weights: 门控权重 [batch_size, total_experts]
            sparsity_loss: 稀疏度损失
        """
        batch_size = x.size(0)
        
        # 计算所有专家输出
        expert_outputs = []
        
        for expert in self.shared_experts:
            expert_outputs.append(expert(x))
        
        for expert in self.task_experts:
            expert_outputs.append(expert(x))
        
        # 堆叠专家输出 [batch_size, total_experts, output_size]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # 计算门控权重
        gate_weights, sparsity_loss = self.gate(x)
        
        # 加权组合 [batch_size, output_size]
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # [batch_size, total_experts, 1]
        output = (expert_outputs * gate_weights_expanded).sum(dim=1)
        
        return output, gate_weights, sparsity_loss


class ImageEncoder(nn.Module):
    """图像编码器"""
    
    def __init__(self, 
                 encoder_type: str = "resnet18",
                 pretrained: bool = True,
                 output_size: int = 256):
        super().__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type.startswith("resnet"):
            # 使用简化的 CNN 结构（避免依赖 torchvision models）
            self.encoder = nn.Sequential(
                # 初始卷积
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                # 残差块简化版
                self._make_residual_block(64, 64),
                self._make_residual_block(64, 128, stride=2),
                self._make_residual_block(128, 256, stride=2),
                self._make_residual_block(256, 512, stride=2),
                
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            feature_size = 512
        else:
            # 简单 CNN
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            feature_size = 128
        
        # 投影层
        self.projector = nn.Sequential(
            nn.Linear(feature_size, output_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
    
    def _make_residual_block(self, in_channels: int, out_channels: int, 
                             stride: int = 1) -> nn.Sequential:
        """创建残差块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像张量 [batch_size, 3, H, W]
        
        Returns:
            features: 图像特征 [batch_size, output_size]
        """
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        features = self.projector(features)
        return features


class TextEncoder(nn.Module):
    """文本编码器（简化版，不依赖预训练模型）"""
    
    def __init__(self,
                 vocab_size: int = 30522,
                 embed_size: int = 128,
                 hidden_size: int = 256,
                 output_size: int = 256,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 max_length: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        
        self.embed_size = embed_size
        self.max_length = max_length
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_length, embed_size)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.projector = nn.Sequential(
            nn.Linear(embed_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, text: List[str], tokenizer=None) -> torch.Tensor:
        """
        Args:
            text: 文本列表
            tokenizer: 分词器（可选）
        
        Returns:
            features: 文本特征 [batch_size, output_size]
        """
        batch_size = len(text)
        device = next(self.parameters()).device
        
        # 简单的字符级编码（实际应用中应使用真正的分词器）
        # 这里使用字符的 ASCII 码作为 token
        max_len = min(self.max_length, max(len(t) for t in text))
        
        tokens = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, t in enumerate(text):
            for j, char in enumerate(t[:max_len]):
                tokens[i, j] = ord(char) % 30522  # 映射到词汇表大小
        
        # 位置编码
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # 嵌入
        x = self.embedding(tokens) + self.pos_embedding(positions)
        
        # Transformer 编码
        x = self.transformer(x)
        
        # 取 [CLS] 位置或平均池化
        features = x.mean(dim=1)
        
        # 投影
        features = self.projector(features)
        
        return features


class MultimodalPLEModel(nn.Module):
    """
    多模态 PLE 模型
    实现图文梯度解耦
    """
    
    def __init__(self, config: ModelConfig, num_users: int, num_items: int):
        super().__init__()
        
        self.config = config
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(num_users, config.hidden_size)
        self.item_embedding = nn.Embedding(num_items, config.hidden_size)
        
        # 图像编码器
        self.image_encoder = ImageEncoder(
            encoder_type=config.image_encoder,
            pretrained=config.image_pretrained,
            output_size=config.hidden_size,
        )
        
        # 文本编码器
        self.text_encoder = TextEncoder(
            vocab_size=30522,
            embed_size=config.text_embedding_size,
            hidden_size=config.hidden_size,
            output_size=config.hidden_size,
        )
        
        # 图像任务 PLE 层
        self.image_ple = PLELayer(
            input_size=config.hidden_size * 2,  # user + item
            output_size=config.hidden_size,
            num_shared_experts=config.num_shared_experts,
            num_task_experts=config.num_task_specific_experts,
            expert_hidden_size=config.expert_hidden_size,
            gate_hidden_size=config.gate_hidden_size,
            dropout=config.dropout,
            temperature=config.gate_temperature,
            top_k=config.gate_top_k,
        )
        
        # 文本任务 PLE 层
        self.text_ple = PLELayer(
            input_size=config.hidden_size * 2,  # user + item
            output_size=config.hidden_size,
            num_shared_experts=config.num_shared_experts,
            num_task_experts=config.num_task_specific_experts,
            expert_hidden_size=config.expert_hidden_size,
            gate_hidden_size=config.gate_hidden_size,
            dropout=config.dropout,
            temperature=config.gate_temperature,
            top_k=config.gate_top_k,
        )
        
        # 图像特征融合层
        self.image_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        
        # 文本特征融合层
        self.text_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        
        # 任务塔
        self.image_tower = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
        )
        
        self.text_tower = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
        )
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
    
    def forward(self, 
                user_ids: torch.Tensor,
                item_ids: torch.Tensor,
                images: torch.Tensor,
                texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Args:
            user_ids: 用户ID [batch_size]
            item_ids: 物品ID [batch_size]
            images: 图像 [batch_size, 3, H, W]
            texts: 文本列表
        
        Returns:
            字典包含:
                - image_pred: 图像任务预测
                - text_pred: 文本任务预测
                - final_pred: 最终预测
                - image_gate_weights: 图像任务门控权重
                - text_gate_weights: 文本任务门控权重
                - image_sparsity_loss: 图像任务稀疏度损失
                - text_sparsity_loss: 文本任务稀疏度损失
        """
        # 用户和物品嵌入
        user_emb = self.user_embedding(user_ids)  # [batch_size, hidden_size]
        item_emb = self.item_embedding(item_ids)  # [batch_size, hidden_size]
        
        # 基础特征
        base_features = torch.cat([user_emb, item_emb], dim=-1)  # [batch_size, hidden_size * 2]
        
        # 图像编码
        image_features = self.image_encoder(images)  # [batch_size, hidden_size]
        
        # 文本编码
        text_features = self.text_encoder(texts)  # [batch_size, hidden_size]
        
        # 图像任务 PLE
        image_ple_out, image_gate_weights, image_sparsity_loss = self.image_ple(base_features)
        image_fused = self.image_fusion(torch.cat([image_ple_out, image_features], dim=-1))
        image_pred = self.image_tower(image_fused)
        
        # 文本任务 PLE
        text_ple_out, text_gate_weights, text_sparsity_loss = self.text_ple(base_features)
        text_fused = self.text_fusion(torch.cat([text_ple_out, text_features], dim=-1))
        text_pred = self.text_tower(text_fused)
        
        # 最终预测
        combined = torch.cat([image_pred, text_pred], dim=-1)
        final_pred = self.final_fusion(combined)
        
        return {
            "image_pred": image_pred.squeeze(-1),
            "text_pred": text_pred.squeeze(-1),
            "final_pred": final_pred.squeeze(-1),
            "image_gate_weights": image_gate_weights,
            "text_gate_weights": text_gate_weights,
            "image_sparsity_loss": image_sparsity_loss,
            "text_sparsity_loss": text_sparsity_loss,
            "image_features": image_features,
            "text_features": text_features,
        }
    
    def get_expert_utilization(self) -> Dict[str, float]:
        """获取专家利用率统计"""
        # 这里可以添加更详细的统计
        return {
            "image_shared_experts": self.config.num_shared_experts,
            "image_task_experts": self.config.num_task_specific_experts,
            "text_shared_experts": self.config.num_shared_experts,
            "text_task_experts": self.config.num_task_specific_experts,
        }
