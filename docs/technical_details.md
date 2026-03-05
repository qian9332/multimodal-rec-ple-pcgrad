# 多模态推荐系统技术详解

## 目录

1. [项目概述](#1-项目概述)
2. [核心问题分析](#2-核心问题分析)
3. [PLE多任务学习架构](#3-ple多任务学习架构)
4. [PCGrad梯度冲突投影](#4-pcgrad梯度冲突投影)
5. [损失函数设计](#5-损失函数设计)
6. [正则化技术](#6-正则化技术)
7. [数据处理技术](#7-数据处理技术)
8. [评估指标体系](#8-评估指标体系)
9. [训练策略](#9-训练策略)
10. [实验结果分析](#10-实验结果分析)
11. [优化方向](#11-优化方向)

---

## 1. 项目概述

### 1.1 项目背景

在多模态推荐系统中，图像和文本两种模态的特征在学习过程中存在**梯度冲突**问题。具体表现为：

- 图文梯度的余弦相似度在 40%-60% 的训练步骤中为负值
- 导致文本侧任务性能下降约 2-3pp AUC
- 传统多任务学习方法难以解决此问题

### 1.2 解决方案

本项目提出 **PLE + PCGrad** 组合方案：

```
┌─────────────────────────────────────────────────────────────┐
│                    多模态推荐系统架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入层: 用户ID + 物品ID + 图像 + 文本                        │
│                          ↓                                  │
│  编码层: 用户嵌入 + 物品嵌入 + 图像编码器 + 文本编码器          │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              PLE 多任务解耦层                         │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │   │
│  │  │共享专家1│  │共享专家2│  │图像专家 │  │文本专家 │ │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘ │   │
│  │       └─────────────┴────────────┴────────────┘      │   │
│  │                      ↓                               │   │
│  │              ┌───────────────┐                       │   │
│  │              │  Gate 网络    │ ← 稀疏度约束           │   │
│  │              └───────┬───────┘                       │   │
│  └──────────────────────┼───────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              PCGrad 梯度投影层                        │   │
│  │                                                     │   │
│  │   检测梯度冲突 → 投影到法平面 → 合并梯度              │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  任务塔: 图像任务塔 + 文本任务塔 → 最终融合                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.10 |
| 图像处理 | TorchVision |
| 文本处理 | Transformer (简化版) |
| 数据处理 | NumPy, PIL |
| 数据集 | Amazon-2023 (HuggingFace) |

---

## 2. 核心问题分析

### 2.1 梯度冲突的定义

在多任务学习中，设两个任务 $i$ 和 $j$ 的梯度分别为 $g_i$ 和 $g_j$，梯度冲突定义为：

$$\cos(g_i, g_j) = \frac{g_i \cdot g_j}{\|g_i\| \cdot \|g_j\|} < 0$$

当余弦相似度为负时，两个梯度方向相反，相互干扰。

### 2.2 梯度冲突的影响

```
梯度冲突的影响链:

图文梯度冲突
    ↓
文本梯度被图像梯度"拉偏"
    ↓
文本特征学习不充分
    ↓
文本任务性能下降 (2-3pp AUC)
    ↓
最终推荐效果受损
```

### 2.3 冲突比例统计

根据实验数据：

| 训练阶段 | 冲突比例 |
|----------|----------|
| 初始阶段 (Epoch 1-3) | 60-100% |
| 中期阶段 (Epoch 4-6) | 40-100% |
| 稳定阶段 (Epoch 7+) | 20-60% |
| **平均冲突率** | **56%** |

---

## 3. PLE多任务学习架构

### 3.1 PLE 原理

**PLE (Progressive Layered Extraction)** 是一种多任务学习架构，核心思想是：

1. **共享专家**：学习跨任务的通用知识
2. **任务特定专家**：学习任务独有的知识
3. **门控网络**：动态选择专家组合

### 3.2 数学公式

设输入特征为 $x$，专家输出为 $E_k(x)$，门控权重为 $g_k(x)$：

**专家输出组合：**
$$h(x) = \sum_{k=1}^{K} g_k(x) \cdot E_k(x)$$

**门控权重计算（带稀疏度约束）：**
$$g_k(x) = \frac{\exp(w_k^T x / \tau)}{\sum_{j=1}^{K} \exp(w_j^T x / \tau)}$$

其中 $\tau$ 是温度参数，控制分布的平滑程度。

**稀疏度损失：**
$$L_{sparsity} = -\sum_{k=1}^{K} g_k(x) \log g_k(x)$$

### 3.3 代码实现

```python
class PLELayer(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size,
                 num_shared_experts=2,    # 共享专家数量
                 num_task_experts=2,      # 任务特定专家数量
                 expert_hidden_size=256,
                 gate_hidden_size=128,
                 dropout=0.2,
                 temperature=1.0,         # Gate温度
                 top_k=2):                # Top-K专家选择
        
        super().__init__()
        
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        self.total_experts = num_shared_experts + num_task_experts
        
        # 共享专家 - 学习跨任务通用知识
        self.shared_experts = nn.ModuleList([
            ExpertNetwork(input_size, expert_hidden_size, output_size)
            for _ in range(num_shared_experts)
        ])
        
        # 任务特定专家 - 学习任务独有知识
        self.task_experts = nn.ModuleList([
            ExpertNetwork(input_size, expert_hidden_size, output_size)
            for _ in range(num_task_experts)
        ])
        
        # 门控网络 - 动态选择专家
        self.gate = GateNetwork(
            input_size, 
            self.total_experts,
            gate_hidden_size,
            temperature,
            top_k
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 计算所有专家输出
        expert_outputs = []
        for expert in self.shared_experts:
            expert_outputs.append(expert(x))
        for expert in self.task_experts:
            expert_outputs.append(expert(x))
        
        # 堆叠专家输出 [batch_size, total_experts, output_size]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # 计算门控权重和稀疏度损失
        gate_weights, sparsity_loss = self.gate(x)
        
        # 加权组合
        gate_weights_expanded = gate_weights.unsqueeze(-1)
        output = (expert_outputs * gate_weights_expanded).sum(dim=1)
        
        return output, gate_weights, sparsity_loss
```

### 3.4 专家网络结构

```python
class ExpertNetwork(nn.Module):
    """专家网络 - 两层MLP"""
    def __init__(self, input_size, hidden_size, output_size, 
                 num_layers=2, dropout=0.2):
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
    
    def forward(self, x):
        return self.network(x)
```

### 3.5 门控网络与稀疏度约束

```python
class GateNetwork(nn.Module):
    """门控网络 - 带稀疏度约束"""
    def __init__(self, input_size, num_experts, hidden_size=128,
                 temperature=1.0, top_k=2):
        super().__init__()
        
        self.temperature = temperature
        self.top_k = top_k
        self.num_experts = num_experts
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts),
        )
    
    def forward(self, x):
        logits = self.network(x)
        
        # Softmax with temperature
        gate_weights = F.softmax(logits / self.temperature, dim=-1)
        
        # Top-K 稀疏化 - 只保留最重要的K个专家
        if self.top_k < self.num_experts:
            top_k_values, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
            
            # 创建稀疏掩码
            sparse_mask = torch.zeros_like(gate_weights)
            sparse_mask.scatter_(1, top_k_indices, 1.0)
            
            # 应用掩码并重新归一化
            gate_weights = gate_weights * sparse_mask
            gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 稀疏度损失 - 鼓励权重集中（熵最小化）
        sparsity_loss = -torch.mean(
            torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=-1)
        )
        
        return gate_weights, sparsity_loss
```

### 3.6 PLE 的优势

| 特性 | 说明 |
|------|------|
| 任务解耦 | 任务特定专家学习独有知识，减少干扰 |
| 知识共享 | 共享专家学习通用知识，提高效率 |
| 动态选择 | 门控网络根据输入动态选择专家组合 |
| 稀疏激活 | Top-K 机制减少计算量，提高可解释性 |

---

## 4. PCGrad梯度冲突投影

### 4.1 PCGrad 原理

**PCGrad (Projected Conflicting Gradient)** 是一种解决梯度冲突的方法，核心思想是：

当两个任务的梯度冲突时（余弦相似度 < 0），将一个任务的梯度投影到另一个任务梯度的法平面。

### 4.2 数学公式

设任务 $i$ 的梯度为 $g_i$，任务 $j$ 的梯度为 $g_j$，当 $\cos(g_i, g_j) < 0$ 时：

**投影公式：**
$$g_i^{proj} = g_i - \frac{g_i \cdot g_j}{\|g_j\|^2} g_j$$

**几何解释：**
```
原始梯度:
        g_i
         ↖
          \
           \  冲突区域
            \
    ←--------●--------→ g_j
             
投影后:
        g_i^proj
             ↑
             |
             |  无冲突
             |
    ←--------●--------→ g_j
```

### 4.3 代码实现

```python
class PCGradOptimizer:
    """PCGrad 优化器"""
    
    def __init__(self, optimizer, config, num_tasks=2):
        self.optimizer = optimizer
        self.config = config
        self.num_tasks = num_tasks
        self.analyzer = GradientConflictAnalyzer(config)
    
    def compute_gradients(self, model, losses):
        """计算各任务的梯度"""
        task_gradients = {}
        
        # 获取所有参数形状
        param_shapes = [(p.numel(), p.shape) 
                        for p in model.parameters() if p.requires_grad]
        
        for task_name, loss in losses.items():
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            
            # 收集梯度（按固定顺序）
            gradients = []
            for param in model.parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        gradients.append(param.grad.clone().flatten())
                    else:
                        # 无梯度时用零填充
                        gradients.append(torch.zeros(param.numel(), 
                                                     device=param.device))
            
            if gradients:
                task_gradients[task_name] = torch.cat(gradients)
        
        return task_gradients
    
    def project_conflicting_gradients(self, task_gradients):
        """投影冲突梯度"""
        task_names = list(task_gradients.keys())
        projected_gradients = {name: grad.clone() 
                              for name, grad in task_gradients.items()}
        
        for i, task_i in enumerate(task_names):
            for j, task_j in enumerate(task_names):
                if i >= j:
                    continue
                
                grad_i = projected_gradients[task_i]
                grad_j = projected_gradients[task_j]
                
                # 计算余弦相似度
                cos_sim = self.compute_cosine_similarity(grad_i, grad_j)
                
                # 如果冲突（余弦相似度 < 阈值）
                if cos_sim < self.config.conflict_threshold:
                    # 投影 grad_i 到 grad_j 的法平面
                    grad_j_norm_sq = torch.dot(grad_j.flatten(), grad_j.flatten())
                    
                    if grad_j_norm_sq > 0:
                        dot_product = torch.dot(grad_i.flatten(), grad_j.flatten())
                        projected_gradients[task_i] = grad_i - \
                            (dot_product / grad_j_norm_sq) * grad_j
        
        return projected_gradients
    
    def step(self, model, losses):
        """执行一步优化"""
        # 1. 计算各任务梯度
        task_gradients = self.compute_gradients(model, losses)
        
        # 2. 分析梯度冲突
        conflict_analysis = self.analyzer.analyze_conflicts(task_gradients)
        
        # 3. 投影冲突梯度
        if self.config.enable_pcgrad:
            projected_gradients = self.project_conflicting_gradients(task_gradients)
        else:
            projected_gradients = task_gradients
        
        # 4. 应用梯度
        self.apply_gradients(model, projected_gradients)
        
        # 5. 梯度裁剪
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.config.gradient_clip_norm
            )
        
        # 6. 更新参数
        self.optimizer.step()
        
        return conflict_analysis
```

### 4.4 梯度冲突分析器

```python
class GradientConflictAnalyzer:
    """梯度冲突分析器"""
    
    def compute_cosine_similarity(self, grad1, grad2):
        """计算余弦相似度"""
        grad1_flat = grad1.flatten()
        grad2_flat = grad2.flatten()
        
        dot_product = torch.dot(grad1_flat, grad2_flat)
        norm1 = torch.norm(grad1_flat)
        norm2 = torch.norm(grad2_flat)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return (dot_product / (norm1 * norm2)).item()
    
    def analyze_conflicts(self, task_gradients):
        """分析任务间的梯度冲突"""
        task_names = list(task_gradients.keys())
        
        results = {
            "cosine_similarities": {},
            "conflict_pairs": [],
            "conflict_ratio": 0.0,
        }
        
        conflict_count = 0
        total_pairs = 0
        
        for i in range(len(task_names)):
            for j in range(i + 1, len(task_names)):
                task1, task2 = task_names[i], task_names[j]
                
                cos_sim = self.compute_cosine_similarity(
                    task_gradients[task1], 
                    task_gradients[task2]
                )
                
                results["cosine_similarities"][(task1, task2)] = cos_sim
                
                # 判断是否冲突
                is_conflict = cos_sim < self.config.conflict_threshold
                if is_conflict:
                    conflict_count += 1
                    results["conflict_pairs"].append((task1, task2, cos_sim))
                
                total_pairs += 1
        
        results["conflict_ratio"] = conflict_count / total_pairs if total_pairs > 0 else 0.0
        
        return results
```

### 4.5 PCGrad 的效果

| 指标 | 无 PCGrad | 有 PCGrad |
|------|-----------|-----------|
| 平均冲突率 | 60-80% | 20-60% |
| 文本 AUC | 基线 | +1.5-2pp |
| 训练稳定性 | 波动大 | 更稳定 |

---

## 5. 损失函数设计

### 5.1 损失函数组合

本项目使用多种损失函数组合：

```
总损失 = 图像任务损失 + 文本任务损失 + 最终预测损失 + 稀疏度损失
```

### 5.2 Focal Loss

**用途**：处理类别不平衡问题

**公式**：
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

其中：
- $p_t$ 是预测概率
- $\alpha_t$ 是类别权重
- $\gamma$ 是聚焦参数（通常取 2.0）

**代码实现**：
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # BCE 损失
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # p_t = exp(-bce_loss)
        p_t = torch.exp(-bce_loss)
        
        # alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal Loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss
```

### 5.3 逆频率加权损失

**用途**：抑制高频样本主导

**公式**：
$$w_c = \frac{N}{N_c + \epsilon}$$

其中 $N$ 是总样本数，$N_c$ 是类别 $c$ 的样本数。

### 5.4 难负例挖掘损失

**用途**：选择困难的负样本进行训练

**策略**：选择损失最大的 30% 负样本

### 5.5 R-Drop 正则化

**用途**：通过一致性正则化增强泛化能力

**公式**：
$$L_{R-Drop} = \frac{\alpha}{2}(KL(p_1 \| p_2) + KL(p_2 \| p_1))$$

---

## 6. 正则化技术

### 6.1 DropBlock

**用途**：图像侧正则化，丢弃连续区域

**与 Dropout 的区别**：
- Dropout：随机丢弃单个神经元
- DropBlock：随机丢弃连续区域，更适合卷积网络

### 6.2 Label Smoothing

**用途**：防止模型过度自信

**公式**：
$$y_{smooth} = y(1 - \epsilon) + 0.5 \cdot \epsilon$$

### 6.3 Early Stopping

**用途**：防止过拟合

**策略**：当验证指标连续 N 个 epoch 不改善时停止

### 6.4 权重衰减

**用途**：L2 正则化，防止权重过大

**公式**：
$$L_{reg} = \lambda \sum_i w_i^2$$

---

## 7. 数据处理技术

### 7.1 数据加载

**流式下载**：避免内存溢出

### 7.2 数据过滤

**低频过滤**：移除交互次数过少的用户和物品

### 7.3 数据增强

**文本增强**：
- 随机删除单词
- 随机交换单词

**图像增强**：
- 随机水平翻转
- 随机旋转
- 颜色抖动

---

## 8. 评估指标体系

### 8.1 AUC (Area Under ROC Curve)

**用途**：衡量二分类模型的整体性能

### 8.2 LogLoss

**用途**：衡量预测概率的准确性

**公式**：
$$LogLoss = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(p_i) + (1-y_i)\log(1-p_i)]$$

### 8.3 NDCG (Normalized Discounted Cumulative Gain)

**用途**：衡量排序质量

**公式**：
$$DCG@K = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

### 8.4 Hit Rate

**用途**：衡量 Top-K 推荐的命中率

---

## 9. 训练策略

### 9.1 学习率调度

**Warmup + Cosine Annealing**：
- Warmup 阶段：学习率从 0.1*lr 线性增加到 lr
- Cosine Annealing 阶段：学习率按余弦曲线衰减

### 9.2 梯度裁剪

**用途**：防止梯度爆炸

### 9.3 批次大小选择

| 批次大小 | 优点 | 缺点 |
|----------|------|------|
| 小 (16-32) | 泛化好 | 训练慢 |
| 中 (64-128) | 平衡 | - |
| 大 (256+) | 训练快 | 可能过拟合 |

---

## 10. 实验结果分析

### 10.1 训练曲线

验证 AUC 从 0.45 提升到 0.84

### 10.2 门控权重分析

**图像任务门控权重分布：**
- 共享专家: 86.8%
- 任务特定专家: 13.2%

**文本任务门控权重分布：**
- 共享专家: 61.4%
- 任务特定专家: 38.6%

### 10.3 关键指标

| 指标 | 值 |
|------|-----|
| 最佳验证 AUC | 0.8429 |
| 测试 NDCG@5 | 0.90 |
| 平均梯度冲突率 | 56% |

---

## 11. 优化方向

### 11.1 数据层面

- 扩大数据规模
- 处理类别不平衡
- 增强数据增强

### 11.2 模型层面

- 减少模型复杂度
- 优化 PLE 结构
- 调整 PCGrad 参数

### 11.3 训练层面

- 调整学习率
- 增强正则化
- 优化 Early Stop

---

## 附录

### A. 配置参数说明

```python
@dataclass
class ModelConfig:
    hidden_size: int = 256
    num_experts: int = 8
    num_shared_experts: int = 2
    num_task_specific_experts: int = 2
    gate_sparsity_lambda: float = 0.1
    gate_temperature: float = 1.0
    gate_top_k: int = 2
    dropout: float = 0.2
```

### B. 参考文献

1. **PLE**: Tang, H., et al. "Progressive Layered Extraction (PLE)" RecSys 2020.
2. **PCGrad**: Yu, T., et al. "Gradient Surgery for Multi-Task Learning" NeurIPS 2020.
3. **Focal Loss**: Lin, T., et al. "Focal Loss for Dense Object Detection" ICCV 2017.
4. **R-Drop**: Liang, X., et al. "R-Drop: Regularized Dropout" NeurIPS 2021.
5. **DropBlock**: Ghiasi, G., et al. "DropBlock" NeurIPS 2018.
