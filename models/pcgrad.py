"""
PCGrad (Projected Conflicting Gradient) 实现
用于解决多任务学习中的梯度冲突问题
"""
import math
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# 添加父目录到路径
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import PCGradConfig


class GradientConflictAnalyzer:
    """梯度冲突分析器"""
    
    def __init__(self, config: PCGradConfig):
        self.config = config
        self.conflict_history = defaultdict(list)
        self.step_count = 0
    
    def compute_cosine_similarity(self, grad1: torch.Tensor, grad2: torch.Tensor) -> float:
        """计算两个梯度向量的余弦相似度"""
        grad1_flat = grad1.flatten()
        grad2_flat = grad2.flatten()
        
        dot_product = torch.dot(grad1_flat, grad2_flat)
        norm1 = torch.norm(grad1_flat)
        norm2 = torch.norm(grad2_flat)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return (dot_product / (norm1 * norm2)).item()
    
    def analyze_conflicts(self, task_gradients: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """
        分析任务间的梯度冲突
        
        Args:
            task_gradients: 任务名称到梯度张量的字典
        
        Returns:
            冲突分析结果
        """
        task_names = list(task_gradients.keys())
        num_tasks = len(task_names)
        
        results = {
            "cosine_similarities": {},
            "conflict_pairs": [],
            "conflict_ratio": 0.0,
        }
        
        conflict_count = 0
        total_pairs = 0
        
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                task1, task2 = task_names[i], task_names[j]
                
                cos_sim = self.compute_cosine_similarity(
                    task_gradients[task1], 
                    task_gradients[task2]
                )
                
                results["cosine_similarities"][(task1, task2)] = cos_sim
                
                # 判断是否冲突（余弦相似度小于阈值）
                is_conflict = cos_sim < self.config.conflict_threshold
                if is_conflict:
                    conflict_count += 1
                    results["conflict_pairs"].append((task1, task2, cos_sim))
                
                total_pairs += 1
        
        results["conflict_ratio"] = conflict_count / total_pairs if total_pairs > 0 else 0.0
        
        # 记录历史
        self.conflict_history[self.step_count] = results
        self.step_count += 1
        
        return results
    
    def get_statistics(self) -> Dict[str, any]:
        """获取统计信息"""
        if not self.conflict_history:
            return {}
        
        all_conflict_ratios = [
            h["conflict_ratio"] for h in self.conflict_history.values()
        ]
        
        return {
            "avg_conflict_ratio": sum(all_conflict_ratios) / len(all_conflict_ratios),
            "max_conflict_ratio": max(all_conflict_ratios),
            "min_conflict_ratio": min(all_conflict_ratios),
            "total_steps": len(self.conflict_history),
        }


class PCGradOptimizer:
    """
    PCGrad 优化器
    通过投影解决梯度冲突
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 config: PCGradConfig,
                 num_tasks: int = 2):
        self.optimizer = optimizer
        self.config = config
        self.num_tasks = num_tasks
        
        # 梯度冲突分析器
        self.analyzer = GradientConflictAnalyzer(config)
    
    def project_conflicting_gradients(self, 
                                      task_gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        投影冲突梯度
        
        当两个任务的梯度冲突时（余弦相似度 < 0），
        将一个任务的梯度投影到另一个任务梯度的法平面
        
        Args:
            task_gradients: 任务名称到梯度张量的字典
        
        Returns:
            投影后的梯度字典
        """
        task_names = list(task_gradients.keys())
        projected_gradients = {name: grad.clone() for name, grad in task_gradients.items()}
        
        for i, task_i in enumerate(task_names):
            for j, task_j in enumerate(task_names):
                if i >= j:
                    continue
                
                grad_i = projected_gradients[task_i]
                grad_j = projected_gradients[task_j]
                
                # 计算余弦相似度
                cos_sim = self.analyzer.compute_cosine_similarity(grad_i, grad_j)
                
                # 如果冲突（余弦相似度 < 阈值）
                if cos_sim < self.config.conflict_threshold:
                    # 投影 grad_i 到 grad_j 的法平面
                    # grad_i_proj = grad_i - (grad_i · grad_j) / ||grad_j||^2 * grad_j
                    grad_j_norm_sq = torch.dot(grad_j.flatten(), grad_j.flatten())
                    
                    if grad_j_norm_sq > 0:
                        dot_product = torch.dot(grad_i.flatten(), grad_j.flatten())
                        projected_gradients[task_i] = grad_i - (dot_product / grad_j_norm_sq) * grad_j
        
        return projected_gradients
    
    def compute_gradients(self, 
                          model: nn.Module,
                          losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算各任务的梯度
        
        Args:
            model: 模型
            losses: 任务名称到损失张量的字典
        
        Returns:
            任务名称到梯度张量的字典
        """
        task_gradients = {}
        
        for task_name, loss in losses.items():
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 计算梯度
            loss.backward(retain_graph=True)
            
            # 收集梯度
            gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.clone().flatten())
            
            if gradients:
                task_gradients[task_name] = torch.cat(gradients)
        
        return task_gradients
    
    def apply_gradients(self, 
                        model: nn.Module,
                        projected_gradients: Dict[str, torch.Tensor]):
        """
        应用投影后的梯度
        
        Args:
            model: 模型
            projected_gradients: 投影后的梯度字典
        """
        # 平均所有任务的梯度
        avg_gradient = None
        for grad in projected_gradients.values():
            if avg_gradient is None:
                avg_gradient = grad.clone()
            else:
                avg_gradient += grad
        
        if avg_gradient is not None:
            avg_gradient /= len(projected_gradients)
            
            # 将平均梯度分配回模型参数
            offset = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_size = param.numel()
                    param.grad = avg_gradient[offset:offset + param_size].view(param.shape)
                    offset += param_size
    
    def step(self, 
             model: nn.Module,
             losses: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """
        执行一步优化
        
        Args:
            model: 模型
            losses: 任务名称到损失张量的字典
        
        Returns:
            冲突分析结果
        """
        # 计算各任务梯度
        task_gradients = self.compute_gradients(model, losses)
        
        # 分析梯度冲突
        conflict_analysis = self.analyzer.analyze_conflicts(task_gradients)
        
        # 投影冲突梯度
        if self.config.enable_pcgrad:
            projected_gradients = self.project_conflicting_gradients(task_gradients)
        else:
            projected_gradients = task_gradients
        
        # 应用梯度
        self.apply_gradients(model, projected_gradients)
        
        # 梯度裁剪
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.config.gradient_clip_norm
            )
        
        # 更新参数
        self.optimizer.step()
        
        return conflict_analysis
    
    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()


class GradientConflictLogger:
    """梯度冲突日志记录器"""
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.history = []
    
    def log(self, step: int, conflict_analysis: Dict[str, any]):
        """记录冲突分析结果"""
        self.history.append({
            "step": step,
            "conflict_ratio": conflict_analysis["conflict_ratio"],
            "conflict_pairs": conflict_analysis["conflict_pairs"],
        })
        
        if step % self.log_interval == 0:
            print(f"Step {step}:")
            print(f"  冲突比例: {conflict_analysis['conflict_ratio']:.2%}")
            
            if conflict_analysis["conflict_pairs"]:
                print("  冲突任务对:")
                for task1, task2, cos_sim in conflict_analysis["conflict_pairs"]:
                    print(f"    {task1} vs {task2}: cos_sim={cos_sim:.4f}")
    
    def get_summary(self) -> Dict[str, any]:
        """获取摘要统计"""
        if not self.history:
            return {}
        
        conflict_ratios = [h["conflict_ratio"] for h in self.history]
        
        return {
            "total_steps": len(self.history),
            "avg_conflict_ratio": sum(conflict_ratios) / len(conflict_ratios),
            "max_conflict_ratio": max(conflict_ratios),
            "min_conflict_ratio": min(conflict_ratios),
        }


def visualize_gradient_conflicts(conflict_history: List[Dict], save_path: str = None):
    """
    可视化梯度冲突历史
    
    Args:
        conflict_history: 冲突历史列表
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        
        steps = [h["step"] for h in conflict_history]
        ratios = [h["conflict_ratio"] for h in conflict_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, ratios, 'b-', linewidth=2)
        plt.axhline(y=0.4, color='r', linestyle='--', label='40% threshold')
        plt.axhline(y=0.6, color='r', linestyle='--', label='60% threshold')
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Conflict Ratio')
        plt.title('Gradient Conflict Ratio During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("matplotlib 未安装，跳过可视化")
