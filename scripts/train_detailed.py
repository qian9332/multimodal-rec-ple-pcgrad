#!/usr/bin/env python3
"""
增强版训练脚本
多模态推荐系统 - PLE + PCGrad 方案
详细日志记录
"""
import os
import json
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

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
from utils.metrics import MetricsEvaluator, TrainingMonitor, compute_auc


def setup_logging(output_dir: Path) -> logging.Logger:
    """设置日志"""
    log_file = output_dir / "training.log"
    
    # 创建 logger
    logger = logging.getLogger("MultimodalRec")
    logger.setLevel(logging.DEBUG)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


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
                logger: logging.Logger,
                conflict_logger: GradientConflictLogger = None) -> Dict[str, float]:
    """
    训练一个 epoch
    """
    model.train()
    
    total_loss = 0.0
    total_image_loss = 0.0
    total_text_loss = 0.0
    total_final_loss = 0.0
    total_sparsity_loss = 0.0
    all_predictions = []
    all_labels = []
    conflict_ratios = []
    
    # 详细统计
    batch_losses = []
    gate_weights_image = []
    gate_weights_text = []
    
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        # 获取数据
        user_ids = batch["user_id"].to(device)
        item_ids = batch["item_id"].to(device)
        images = batch["image"].to(device)
        texts = batch["text"]
        labels = batch["label"].to(device)
        
        batch_size = user_ids.size(0)
        
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
            conflict_logger.log(epoch * num_batches + batch_idx, conflict_analysis)
        
        # 记录指标
        batch_loss = losses["total_loss"].item()
        total_loss += batch_loss
        total_image_loss += losses["image_loss"].item()
        total_text_loss += losses["text_loss"].item()
        total_final_loss += losses["final_loss"].item()
        total_sparsity_loss += losses["sparsity_loss"].item()
        
        all_predictions.extend(outputs["final_pred"].detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 记录门控权重
        gate_weights_image.append(outputs["image_gate_weights"].mean(dim=0).detach().cpu().numpy())
        gate_weights_text.append(outputs["text_gate_weights"].mean(dim=0).detach().cpu().numpy())
        
        if conflict_analysis:
            conflict_ratios.append(conflict_analysis["conflict_ratio"])
        
        batch_losses.append({
            "batch_idx": batch_idx,
            "loss": batch_loss,
            "image_loss": losses["image_loss"].item(),
            "text_loss": losses["text_loss"].item(),
        })
        
        # 每10个batch打印一次
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            conflict_str = ""
            if conflict_analysis:
                conflict_str = f", 冲突率: {conflict_analysis['conflict_ratio']:.2%}"
            
            logger.info(
                f"  Batch {batch_idx+1}/{num_batches}: "
                f"Loss={batch_loss:.4f} "
                f"(Image={losses['image_loss'].item():.4f}, "
                f"Text={losses['text_loss'].item():.4f})"
                f"{conflict_str}"
            )
    
    # 计算平均指标
    n_batches = len(train_loader)
    metrics = {
        "loss": total_loss / n_batches,
        "image_loss": total_image_loss / n_batches,
        "text_loss": total_text_loss / n_batches,
        "final_loss": total_final_loss / n_batches,
        "sparsity_loss": total_sparsity_loss / n_batches,
        "avg_conflict_ratio": np.mean(conflict_ratios) if conflict_ratios else 0.0,
    }
    
    # 计算 AUC
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    try:
        metrics["auc"] = compute_auc(predictions, labels)
    except:
        metrics["auc"] = 0.5
    
    # 计算平均门控权重
    avg_gate_image = np.mean(gate_weights_image, axis=0)
    avg_gate_text = np.mean(gate_weights_text, axis=0)
    
    metrics["gate_weights_image"] = avg_gate_image.tolist()
    metrics["gate_weights_text"] = avg_gate_text.tolist()
    metrics["batch_losses"] = batch_losses
    
    return metrics


def evaluate(model: nn.Module,
             val_loader,
             loss_fn: MultimodalLoss,
             device: str,
             evaluator: MetricsEvaluator,
             logger: logging.Logger) -> Dict[str, float]:
    """
    评估模型
    """
    model.eval()
    
    total_loss = 0.0
    total_image_loss = 0.0
    total_text_loss = 0.0
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
            total_image_loss += losses["image_loss"].item()
            total_text_loss += losses["text_loss"].item()
            
            all_predictions.extend(outputs["final_pred"].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_user_ids.extend(user_ids.cpu().numpy())
    
    # 计算指标
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    user_ids = np.array(all_user_ids)
    
    metrics = evaluator.evaluate(predictions, labels, user_ids)
    metrics["loss"] = total_loss / len(val_loader)
    metrics["image_loss"] = total_image_loss / len(val_loader)
    metrics["text_loss"] = total_text_loss / len(val_loader)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="多模态推荐系统训练")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default="/home/z/my-project/multimodal_rec/data")
    parser.add_argument("--output_dir", type=str, default="/home/z/my-project/multimodal_rec/outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
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
    
    # 设置日志
    logger = setup_logging(output_dir)
    
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
    
    # 记录配置
    logger.info("="*70)
    logger.info("多模态推荐系统 - PLE + PCGrad 梯度隔离方案")
    logger.info("="*70)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("")
    logger.info("训练配置:")
    logger.info(f"  - Batch Size: {config.training.batch_size}")
    logger.info(f"  - Epochs: {config.training.num_epochs}")
    logger.info(f"  - Learning Rate: {config.training.learning_rate}")
    logger.info(f"  - Device: {config.training.device}")
    logger.info(f"  - Seed: {config.seed}")
    logger.info("")
    logger.info("模型配置:")
    logger.info(f"  - Hidden Size: {config.model.hidden_size}")
    logger.info(f"  - Num Experts: {config.model.num_experts}")
    logger.info(f"  - Num Shared Experts: {config.model.num_shared_experts}")
    logger.info(f"  - Num Task Experts: {config.model.num_task_specific_experts}")
    logger.info(f"  - Gate Sparsity Lambda: {config.model.gate_sparsity_lambda}")
    logger.info("")
    logger.info("PCGrad配置:")
    logger.info(f"  - Enable PCGrad: {config.pcgrad.enable_pcgrad}")
    logger.info(f"  - Conflict Threshold: {config.pcgrad.conflict_threshold}")
    logger.info(f"  - Gradient Clip Norm: {config.pcgrad.gradient_clip_norm}")
    
    # 数据处理
    logger.info("")
    logger.info("="*70)
    logger.info("加载数据...")
    logger.info("="*70)
    
    processor = DataProcessor(config.data)
    
    # 加载 Sports 数据（数据量更大）
    data_path = Path(config.data.data_dir)
    reviews = processor.load_reviews(
        data_path / "sports_reviews.jsonl", 
        max_samples=config.data.max_samples
    )
    items = processor.load_metadata(
        data_path / "sports_metadata.jsonl",
        max_samples=config.data.max_samples // 2
    )
    
    logger.info(f"加载了 {len(reviews)} 条评论, {len(items)} 个物品")
    
    # 过滤和构建映射
    logger.info("过滤低频交互...")
    reviews = processor.filter_interactions(
        reviews,
        min_user_interactions=2,  # 降低阈值以保留更多数据
        min_item_interactions=2
    )
    processor.build_mappings(reviews)
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config.data, processor, reviews, items, config.training.batch_size
    )
    
    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"验证批次数: {len(val_loader)}")
    logger.info(f"测试批次数: {len(test_loader)}")
    
    # 创建模型
    logger.info("")
    logger.info("="*70)
    logger.info("创建模型...")
    logger.info("="*70)
    
    model = MultimodalPLEModel(
        config.model,
        num_users=len(processor.user2id),
        num_items=len(processor.item2id),
    )
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
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
        total_iters=max(1, int(len(train_loader) * config.training.warmup_ratio))
    )
    main_scheduler = CosineAnnealingLR(
        base_optimizer,
        T_max=max(1, len(train_loader) * config.training.num_epochs),
        eta_min=config.training.learning_rate * 0.01
    )
    scheduler = SequentialLR(
        base_optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[max(1, int(len(train_loader) * config.training.warmup_ratio))]
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
    conflict_logger = GradientConflictLogger(log_interval=50)
    
    # 训练历史
    training_history = {
        "config": {
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "num_epochs": config.training.num_epochs,
            "model_params": total_params,
            "num_users": len(processor.user2id),
            "num_items": len(processor.item2id),
        },
        "epochs": [],
        "gradient_conflicts": [],
    }
    
    # 训练循环
    logger.info("")
    logger.info("="*70)
    logger.info("开始训练...")
    logger.info("="*70)
    
    device = config.training.device
    best_val_auc = 0.0
    best_epoch = 0
    
    for epoch in range(config.training.num_epochs):
        logger.info("")
        logger.info(f"{'='*70}")
        logger.info(f"Epoch {epoch + 1}/{config.training.num_epochs}")
        logger.info(f"{'='*70}")
        
        # 训练
        logger.info("训练阶段:")
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, config, epoch, logger, conflict_logger
        )
        
        # 验证
        logger.info("")
        logger.info("验证阶段:")
        val_metrics = evaluate(model, val_loader, loss_fn, device, evaluator, logger)
        
        # 更新学习率
        scheduler.step()
        current_lr = base_optimizer.param_groups[0]['lr']
        
        # 记录 epoch 结果
        logger.info("")
        logger.info(f"Epoch {epoch + 1} 总结:")
        logger.info(f"  训练 Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}")
        logger.info(f"  验证 Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics.get('auc', 0):.4f}")
        logger.info(f"  图像 Loss: {train_metrics['image_loss']:.4f}")
        logger.info(f"  文本 Loss: {train_metrics['text_loss']:.4f}")
        logger.info(f"  稀疏度 Loss: {train_metrics['sparsity_loss']:.4f}")
        logger.info(f"  梯度冲突率: {train_metrics['avg_conflict_ratio']:.2%}")
        logger.info(f"  学习率: {current_lr:.6f}")
        
        # 门控权重分析
        if train_metrics.get("gate_weights_image"):
            gate_img = train_metrics["gate_weights_image"]
            gate_txt = train_metrics["gate_weights_text"]
            logger.info(f"  图像任务门控权重: {[f'{w:.3f}' for w in gate_img]}")
            logger.info(f"  文本任务门控权重: {[f'{w:.3f}' for w in gate_txt]}")
        
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
        
        # 记录历史
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics.get("auc", 0),
            "learning_rate": current_lr,
            "conflict_ratio": train_metrics["avg_conflict_ratio"],
            "image_loss": train_metrics["image_loss"],
            "text_loss": train_metrics["text_loss"],
            "sparsity_loss": train_metrics["sparsity_loss"],
            "gate_weights_image": train_metrics.get("gate_weights_image"),
            "gate_weights_text": train_metrics.get("gate_weights_text"),
        }
        training_history["epochs"].append(epoch_record)
        
        # 保存最佳模型
        if val_metrics.get("auc", 0) > best_val_auc:
            best_val_auc = val_metrics.get("auc", 0)
            best_epoch = epoch + 1
            model_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"  ★ 保存最佳模型 (AUC: {best_val_auc:.4f})")
        
        # Early Stopping
        if config.regularization.use_early_stop:
            if early_stopping(-val_metrics.get("auc", 0)):
                logger.info(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    # 最终评估
    logger.info("")
    logger.info("="*70)
    logger.info("最终测试集评估...")
    logger.info("="*70)
    
    # 加载最佳模型
    best_model_path = output_dir / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"加载最佳模型 (Epoch {best_epoch})")
    
    test_metrics = evaluate(model, test_loader, loss_fn, device, evaluator, logger)
    
    logger.info("")
    logger.info("测试集结果:")
    for metric, value in test_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
    
    # 梯度冲突统计
    conflict_summary = conflict_logger.get_summary()
    training_history["gradient_conflicts"] = conflict_summary
    
    logger.info("")
    logger.info("="*70)
    logger.info("梯度冲突统计:")
    logger.info("="*70)
    for metric, value in conflict_summary.items():
        logger.info(f"  {metric}: {value}")
    
    # 训练总结
    logger.info("")
    logger.info("="*70)
    logger.info("训练总结:")
    logger.info("="*70)
    logger.info(f"  最佳 Epoch: {best_epoch}")
    logger.info(f"  最佳验证 AUC: {best_val_auc:.4f}")
    logger.info(f"  测试 AUC: {test_metrics.get('auc', 0):.4f}")
    logger.info(f"  测试 LogLoss: {test_metrics.get('logloss', 0):.4f}")
    logger.info(f"  平均梯度冲突率: {conflict_summary.get('avg_conflict_ratio', 0):.2%}")
    logger.info(f"  结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 保存训练历史
    history_path = output_dir / "training_history.json"
    training_history["test_metrics"] = {k: float(v) if isinstance(v, (int, float)) else v 
                                         for k, v in test_metrics.items()}
    training_history["best_epoch"] = best_epoch
    training_history["best_val_auc"] = best_val_auc
    
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    logger.info(f"\n训练历史已保存到: {history_path}")
    
    logger.info("\n训练完成!")
    return training_history


if __name__ == "__main__":
    main()
