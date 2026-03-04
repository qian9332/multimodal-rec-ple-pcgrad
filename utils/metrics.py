"""
评估模块
包含 AUC、LogLoss、NDCG、Hit Rate 等指标
"""
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict


def compute_auc(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    计算 AUC
    
    Args:
        predictions: 预测值
        labels: 真实标签
    
    Returns:
        AUC 值
    """
    # 排序
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_labels = labels[sorted_indices]
    
    # 计算 TP 和 FP
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    
    # 计算 TPR 和 FPR
    tpr = tp / (tp[-1] + 1e-8)
    fpr = fp / (fp[-1] + 1e-8)
    
    # 计算 AUC (梯形法则)
    auc = np.trapz(tpr, fpr)
    
    return auc


def compute_logloss(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    计算 LogLoss
    
    Args:
        predictions: 预测概率
        labels: 真实标签
    
    Returns:
        LogLoss 值
    """
    # 裁剪预测值避免 log(0)
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
    
    # 计算 LogLoss
    logloss = -np.mean(
        labels * np.log(predictions) + 
        (1 - labels) * np.log(1 - predictions)
    )
    
    return logloss


def compute_ndcg_at_k(predictions: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """
    计算 NDCG@K
    
    Args:
        predictions: 预测值
        labels: 真实标签（可以是评分或二值）
        k: 截断位置
    
    Returns:
        NDCG@K 值
    """
    # 排序
    sorted_indices = np.argsort(predictions)[::-1][:k]
    sorted_labels = labels[sorted_indices]
    
    # 计算 DCG
    gains = 2 ** sorted_labels - 1
    discounts = np.log2(np.arange(2, len(sorted_labels) + 2))
    dcg = np.sum(gains / discounts)
    
    # 计算 IDCG
    ideal_labels = np.sort(labels)[::-1][:k]
    ideal_gains = 2 ** ideal_labels - 1
    ideal_discounts = np.log2(np.arange(2, len(ideal_labels) + 2))
    idcg = np.sum(ideal_gains / ideal_discounts)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_hit_rate_at_k(predictions: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """
    计算 Hit Rate@K
    
    Args:
        predictions: 预测值
        labels: 真实标签
        k: 截断位置
    
    Returns:
        Hit Rate@K 值
    """
    # 排序
    sorted_indices = np.argsort(predictions)[::-1][:k]
    sorted_labels = labels[sorted_indices]
    
    # 检查是否有正样本
    return 1.0 if np.sum(sorted_labels) > 0 else 0.0


def compute_precision_at_k(predictions: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """
    计算 Precision@K
    
    Args:
        predictions: 预测值
        labels: 真实标签
        k: 截断位置
    
    Returns:
        Precision@K 值
    """
    sorted_indices = np.argsort(predictions)[::-1][:k]
    sorted_labels = labels[sorted_indices]
    
    return np.mean(sorted_labels)


def compute_recall_at_k(predictions: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """
    计算 Recall@K
    
    Args:
        predictions: 预测值
        labels: 真实标签
        k: 截断位置
    
    Returns:
        Recall@K 值
    """
    sorted_indices = np.argsort(predictions)[::-1][:k]
    sorted_labels = labels[sorted_indices]
    
    total_positives = np.sum(labels)
    if total_positives == 0:
        return 0.0
    
    return np.sum(sorted_labels) / total_positives


class MetricsEvaluator:
    """
    评估器
    """
    
    def __init__(self, metrics: List[str] = None, k_values: List[int] = None):
        self.metrics = metrics or ["auc", "logloss", "ndcg", "hit_rate"]
        self.k_values = k_values or [5, 10, 20]
        
        # 指标函数映射
        self.metric_functions = {
            "auc": compute_auc,
            "logloss": compute_logloss,
        }
        
        # 排序指标函数映射
        self.ranking_metric_functions = {
            "ndcg": compute_ndcg_at_k,
            "hit_rate": compute_hit_rate_at_k,
            "precision": compute_precision_at_k,
            "recall": compute_recall_at_k,
        }
    
    def evaluate(self, 
                 predictions: np.ndarray, 
                 labels: np.ndarray,
                 user_ids: np.ndarray = None) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            predictions: 预测值
            labels: 真实标签
            user_ids: 用户ID（用于分组评估）
        
        Returns:
            指标字典
        """
        results = {}
        
        # 基础指标
        for metric in self.metrics:
            if metric in self.metric_functions:
                results[metric] = self.metric_functions[metric](predictions, labels)
        
        # 排序指标
        if user_ids is not None:
            # 按用户分组计算
            for metric in self.metrics:
                if metric in self.ranking_metric_functions:
                    for k in self.k_values:
                        metric_values = []
                        
                        unique_users = np.unique(user_ids)
                        for user_id in unique_users:
                            mask = user_ids == user_id
                            user_preds = predictions[mask]
                            user_labels = labels[mask]
                            
                            if len(user_preds) > 0:
                                value = self.ranking_metric_functions[metric](
                                    user_preds, user_labels, k
                                )
                                metric_values.append(value)
                        
                        if metric_values:
                            results[f"{metric}@{k}"] = np.mean(metric_values)
        else:
            # 全局计算
            for metric in self.metrics:
                if metric in self.ranking_metric_functions:
                    for k in self.k_values:
                        results[f"{metric}@{k}"] = self.ranking_metric_functions[metric](
                            predictions, labels, k
                        )
        
        return results
    
    def evaluate_per_user(self,
                          predictions: np.ndarray,
                          labels: np.ndarray,
                          user_ids: np.ndarray) -> Dict[str, Dict[int, float]]:
        """
        按用户评估
        
        Args:
            predictions: 预测值
            labels: 真实标签
            user_ids: 用户ID
        
        Returns:
            每个用户的指标字典
        """
        results = defaultdict(dict)
        
        unique_users = np.unique(user_ids)
        for user_id in unique_users:
            mask = user_ids == user_id
            user_preds = predictions[mask]
            user_labels = labels[mask]
            
            if len(user_preds) > 0:
                # 基础指标
                for metric in self.metrics:
                    if metric in self.metric_functions:
                        results[user_id][metric] = self.metric_functions[metric](
                            user_preds, user_labels
                        )
                
                # 排序指标
                for metric in self.metrics:
                    if metric in self.ranking_metric_functions:
                        for k in self.k_values:
                            results[user_id][f"{metric}@{k}"] = self.ranking_metric_functions[metric](
                                user_preds, user_labels, k
                            )
        
        return dict(results)


class TrainingMonitor:
    """
    训练监控器
    """
    
    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_auc": [],
            "val_auc": [],
            "learning_rate": [],
            "gradient_conflict_ratio": [],
        }
        self.best_val_auc = 0.0
        self.best_epoch = 0
    
    def update(self, 
               epoch: int,
               train_loss: float,
               val_loss: float,
               train_auc: float,
               val_auc: float,
               lr: float,
               conflict_ratio: float = None):
        """更新监控记录"""
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_auc"].append(train_auc)
        self.history["val_auc"].append(val_auc)
        self.history["learning_rate"].append(lr)
        
        if conflict_ratio is not None:
            self.history["gradient_conflict_ratio"].append(conflict_ratio)
        
        # 更新最佳记录
        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_epoch = epoch
        
        # 打印
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
        print(f"  LR: {lr:.6f}")
        if conflict_ratio is not None:
            print(f"  Gradient Conflict Ratio: {conflict_ratio:.2%}")
    
    def get_summary(self) -> Dict[str, any]:
        """获取训练摘要"""
        return {
            "best_val_auc": self.best_val_auc,
            "best_epoch": self.best_epoch,
            "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
            "final_val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else None,
            "avg_conflict_ratio": np.mean(self.history["gradient_conflict_ratio"]) 
                if self.history["gradient_conflict_ratio"] else None,
        }
