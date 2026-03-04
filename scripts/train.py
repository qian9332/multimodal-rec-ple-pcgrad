"""
主训练脚本
多模态推荐系统 - PLE + PCGrad 方案
"""
import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import Config, DataConfig, ModelConfig, PCGradConfig, LossConfig, RegularizationConfig, TrainingConfig
from utils.data_processor import DataProcessor, create_dataloaders
from models.ple_model import MultimodalPLEModel
from models.pcgrad import PCGradOptimizer, GradientConflictLogger
from models.losses import MultimodalLoss
from utils.regularization import create_early_stopping, LabelSmoothingLoss
from utils.metrics import MetricsEvaluator, TrainingMonitor


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model: nn.Module,
                train_loader,
                optimizer: PCGradOptimizer,
                loss_fn: MultimodalLoss,
                device: str,
                config: Config,
                epoch: int,
                conflict_logger: GradientConflictLogger = None) -> Dict[str, float]:
    """
    训练一个 epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: PCGrad 优化器
        loss_fn: 损失函数
        device: 设备
        config: 配置
        epoch: 当前 epoch
        conflict_logger: 梯度冲突日志记录器
    
    Returns:
        训练指标字典
    """
    model.train()
    
    total_loss = 0.0
    total_image_loss = 0.0
    total_text_loss = 0.0
    total_sparsity_loss = 0.0
    all_predictions = []
    all_labels = []
    conflict_ratios = []
    
    for batch_idx, batch in enumerate(train_loader):
        # 获取数据
        user_ids = batch["user_id"].to(device)
        item_ids = batch["item_id"].to(device)
        images = batch["image"].to(device)
        texts = batch["text"]
        labels = batch["label"].to(device)
        
        # 前向传播
        outputs = model(user_ids, item_ids, images, texts)
        
        # 计算损失
        losses = loss_fn(outputs, labels)
        
        # PCGrad 优化步骤
        task_losses = {
            "image": losses["image_loss"],
            "text": losses["text_loss"],
        }
        
        conflict_analysis = optimizer.step(model, task_losses)
        
        # 记录梯度冲突
        if conflict_logger is not None:
            conflict_logger.log(epoch * len(train_loader) + batch_idx, conflict_analysis)
        
        # 记录指标
        total_loss += losses["total_loss"].item()
        total_image_loss += losses["image_loss"].item()
        total_text_loss += losses["text_loss"].item()
        total_sparsity_loss += losses["sparsity_loss"].item()
        
        all_predictions.extend(outputs["final_pred"].detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if conflict_analysis:
            conflict_ratios.append(conflict_analysis["conflict_ratio"])
    
    # 计算平均指标
    n_batches = len(train_loader)
    metrics = {
        "loss": total_loss / n_batches,
        "image_loss": total_image_loss / n_batches,
        "text_loss": total_text_loss / n_batches,
        "sparsity_loss": total_sparsity_loss / n_batches,
        "avg_conflict_ratio": np.mean(conflict_ratios) if conflict_ratios else 0.0,
    }
    
    # 计算 AUC
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    try:
        from utils.metrics import compute_auc
        metrics["auc"] = compute_auc(predictions, labels)
    except:
        metrics["auc"] = 0.5
    
    return metrics


def evaluate(model: nn.Module,
             val_loader,
             loss_fn: MultimodalLoss,
             device: str,
             evaluator: MetricsEvaluator) -> Dict[str, float]:
    """
    评估模型
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        loss_fn: 损失函数
        device: 设备
        evaluator: 评估器
    
    Returns:
        评估指标字典
    """
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_user_ids = []
    
    with torch.no_grad():
        for batch in val_loader:
            user_ids = batch["user_id"].to(device)
            item_ids = batch["item_id"].to(device)
            images = batch["image"].to(device)
            texts = batch["text"]
            labels = batch["label"].to(device)
            
            # 前向传播
            outputs = model(user_ids, item_ids, images, texts)
            
            # 计算损失
            losses = loss_fn(outputs, labels)
            total_loss += losses["total_loss"].item()
            
            all_predictions.extend(outputs["final_pred"].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_user_ids.extend(user_ids.cpu().numpy())
    
    # 计算指标
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    user_ids = np.array(all_user_ids)
    
    metrics = evaluator.evaluate(predictions, labels, user_ids)
    metrics["loss"] = total_loss / len(val_loader)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="多模态推荐系统训练")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default="/home/z/my-project/multimodal_rec/data")
    parser.add_argument("--output_dir", type=str, default="/home/z/my-project/multimodal_rec/outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=10000)
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建配置
    config = Config(
        data=DataConfig(data_dir=args.data_dir, max_samples=args.max_samples),
        model=ModelConfig(),
        pcgrad=PCGradConfig(),
        loss=LossConfig(),
        regularization=RegularizationConfig(),
        training=TrainingConfig(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            device=args.device,
            output_dir=str(output_dir),
        ),
        seed=args.seed,
    )
    
    print("="*60)
    print("多模态推荐系统 - PLE + PCGrad 方案")
    print("="*60)
    print(f"配置: {config}")
    
    # 数据处理
    print("\n加载数据...")
    processor = DataProcessor(config.data)
    
    # 加载 Beauty 数据
    data_path = Path(config.data.data_dir)
    reviews = processor.load_reviews(
        data_path / "beauty_reviews.jsonl", 
        max_samples=config.data.max_samples
    )
    items = processor.load_metadata(
        data_path / "beauty_metadata.jsonl",
        max_samples=config.data.max_samples // 2
    )
    
    print(f"加载了 {len(reviews)} 条评论, {len(items)} 个物品")
    
    # 过滤和构建映射
    reviews = processor.filter_interactions(
        reviews,
        min_user_interactions=config.data.min_user_interactions,
        min_item_interactions=config.data.min_item_interactions
    )
    processor.build_mappings(reviews)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        config.data, processor, reviews, items, config.training.batch_size
    )
    
    print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")
    
    # 创建模型
    print("\n创建模型...")
    model = MultimodalPLEModel(
        config.model,
        num_users=len(processor.user2id),
        num_items=len(processor.item2id),
    )
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    # 创建优化器
    base_optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.regularization.weight_decay,
    )
    
    optimizer = PCGradOptimizer(base_optimizer, config.pcgrad, num_tasks=2)
    
    # 创建学习率调度器
    warmup_scheduler = LinearLR(
        base_optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=int(len(train_loader) * config.training.warmup_ratio)
    )
    main_scheduler = CosineAnnealingLR(
        base_optimizer,
        T_max=len(train_loader) * config.training.num_epochs,
        eta_min=config.training.learning_rate * 0.01
    )
    scheduler = SequentialLR(
        base_optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[int(len(train_loader) * config.training.warmup_ratio)]
    )
    
    # 创建损失函数
    loss_fn = MultimodalLoss(config.loss)
    
    # 创建评估器
    evaluator = MetricsEvaluator(
        metrics=config.training.eval_metrics,
        k_values=config.training.eval_k
    )
    
    # 创建 Early Stopping
    early_stopping = create_early_stopping(config.regularization)
    
    # 创建监控器
    monitor = TrainingMonitor()
    conflict_logger = GradientConflictLogger(log_interval=config.pcgrad.conflict_log_interval)
    
    # 训练循环
    print("\n开始训练...")
    device = config.training.device
    
    for epoch in range(config.training.num_epochs):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, config, epoch, conflict_logger
        )
        
        # 验证
        val_metrics = evaluate(model, val_loader, loss_fn, device, evaluator)
        
        # 更新学习率
        scheduler.step()
        current_lr = base_optimizer.param_groups[0]['lr']
        
        # 更新监控器
        monitor.update(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            val_loss=val_metrics["loss"],
            train_auc=train_metrics["auc"],
            val_auc=val_metrics.get("auc", 0.5),
            lr=current_lr,
            conflict_ratio=train_metrics["avg_conflict_ratio"]
        )
        
        # Early Stopping
        if config.regularization.use_early_stop:
            if early_stopping(-val_metrics.get("auc", 0)):
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # 保存检查点
        if (epoch + 1) % config.training.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": base_optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }, checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")
    
    # 最终评估
    print("\n最终测试集评估...")
    test_metrics = evaluate(model, test_loader, loss_fn, device, evaluator)
    
    print("\n测试集结果:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 保存模型
    model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")
    
    # 保存训练历史
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            "history": {k: [float(v) for v in vals] for k, vals in monitor.history.items()},
            "summary": monitor.get_summary(),
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "config": {
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate,
                "num_epochs": config.training.num_epochs,
                "model_params": total_params,
            }
        }, f, indent=2)
    print(f"训练历史已保存到: {history_path}")
    
    # 打印梯度冲突统计
    conflict_summary = conflict_logger.get_summary()
    print("\n梯度冲突统计:")
    for metric, value in conflict_summary.items():
        print(f"  {metric}: {value}")
    
    print("\n训练完成!")
    return monitor.get_summary()


if __name__ == "__main__":
    main()
