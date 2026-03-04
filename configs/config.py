"""
多模态推荐系统配置文件
包含 PLE 解耦 + PCGrad 梯度隔离方案
"""
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "/home/z/my-project/multimodal_rec/data"
    beauty_reviews: str = "beauty_reviews.jsonl"
    beauty_meta: str = "beauty_metadata.jsonl"
    sports_reviews: str = "sports_reviews.jsonl"
    sports_meta: str = "sports_metadata.jsonl"
    
    # 数据划分
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # 数据采样
    max_samples: int = 10000
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    
    # 图像配置
    image_size: int = 224
    max_images: int = 1000
    
    # 文本配置
    max_text_length: int = 128
    vocab_size: int = 30522  # BERT vocab size


@dataclass
class ModelConfig:
    """模型配置"""
    # 共享参数
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.2
    
    # 图像编码器
    image_encoder: str = "resnet18"  # resnet18, resnet50, vit
    image_embedding_size: int = 512
    image_pretrained: bool = True
    
    # 文本编码器
    text_encoder: str = "bert-tiny"  # bert-tiny, bert-base, distilbert
    text_embedding_size: int = 128
    text_pretrained: bool = True
    
    # PLE (Progressive Layered Extraction) 配置
    num_experts: int = 8
    num_shared_experts: int = 2
    num_task_specific_experts: int = 2  # 每个任务的专家数
    expert_hidden_size: int = 256
    gate_hidden_size: int = 128
    
    # 任务配置
    tasks: List[str] = field(default_factory=lambda: ["image", "text"])
    
    # Gate 稀疏度约束
    gate_sparsity_lambda: float = 0.1
    gate_temperature: float = 1.0
    gate_top_k: int = 2  # Top-K 专家选择


@dataclass
class PCGradConfig:
    """PCGrad 梯度冲突投影配置"""
    enable_pcgrad: bool = True
    gradient_clip_norm: float = 1.0
    conflict_threshold: float = 0.0  # 余弦相似度阈值，小于此值视为冲突
    
    # 梯度分析
    log_gradient_conflict: bool = True
    conflict_log_interval: int = 100


@dataclass
class LossConfig:
    """损失函数配置"""
    # 主任务损失
    task_loss_type: str = "bce"  # bce, focal, cross_entropy
    
    # Focal Loss 参数
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # 逆频率降权
    use_inverse_freq_weight: bool = True
    freq_weight_smoothing: float = 1.0
    
    # 难例挖掘
    use_hard_negative_mining: bool = True
    hard_negative_ratio: float = 0.3
    
    # 文本增广
    use_text_augmentation: bool = True
    augmentation_ratio: float = 0.2


@dataclass
class RegularizationConfig:
    """正则化配置"""
    # 图像侧
    use_dropblock: bool = True
    dropblock_block_size: int = 7
    dropblock_prob: float = 0.1
    
    # 通用
    use_early_stop: bool = True
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.001
    
    label_smoothing: float = 0.1
    weight_decay: float = 1e-4
    
    # 文本侧
    use_rdrop: bool = True
    rdrop_alpha: float = 0.1
    
    # MLM 辅助任务
    use_mlm_auxiliary: bool = True
    mlm_probability: float = 0.15
    mlm_weight: float = 0.1


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    
    # 设备配置
    device: str = "cpu"  # cpu, cuda
    num_workers: int = 4
    
    # 日志配置
    log_interval: int = 100
    save_interval: int = 5
    output_dir: str = "/home/z/my-project/multimodal_rec/outputs"
    
    # 评估配置
    eval_metrics: List[str] = field(default_factory=lambda: ["auc", "logloss", "ndcg", "hit_rate"])
    eval_k: List[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pcgrad: PCGradConfig = field(default_factory=PCGradConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 实验名称
    experiment_name: str = "multimodal_rec_ple_pcgrad"
    seed: int = 42
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """从字典创建配置"""
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            pcgrad=PCGradConfig(**config_dict.get("pcgrad", {})),
            loss=LossConfig(**config_dict.get("loss", {})),
            regularization=RegularizationConfig(**config_dict.get("regularization", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            experiment_name=config_dict.get("experiment_name", "multimodal_rec_ple_pcgrad"),
            seed=config_dict.get("seed", 42),
        )
