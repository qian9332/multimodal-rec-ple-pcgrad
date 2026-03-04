"""
数据处理模块
包含数据加载、预处理、增强等功能
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 添加父目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import DataConfig


@dataclass
class Interaction:
    """用户-物品交互数据"""
    user_id: str
    item_id: str
    rating: float
    timestamp: int
    text: str = ""
    image_path: str = ""


@dataclass
class Item:
    """物品数据"""
    item_id: str
    title: str
    description: str
    categories: List[str]
    images: List[str]
    avg_rating: float
    num_ratings: int


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.user2id: Dict[str, int] = {}
        self.item2id: Dict[str, int] = {}
        self.id2user: Dict[int, str] = {}
        self.id2item: Dict[int, str] = {}
        
    def load_reviews(self, filepath: Path, max_samples: int = None) -> List[Interaction]:
        """加载评论数据"""
        interactions = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    interaction = Interaction(
                        user_id=data.get("user_id", ""),
                        item_id=data.get("asin", ""),
                        rating=float(data.get("rating", 0)),
                        timestamp=int(data.get("timestamp", 0)),
                        text=data.get("text", ""),
                    )
                    interactions.append(interaction)
                except Exception as e:
                    continue
        
        return interactions
    
    def load_metadata(self, filepath: Path, max_samples: int = None) -> Dict[str, Item]:
        """加载物品元数据"""
        items = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    item_id = data.get("asin", "")
                    
                    # 处理描述
                    description = data.get("description", [])
                    if isinstance(description, list):
                        description = " ".join(description)
                    
                    # 处理图像
                    images = data.get("images", [])
                    image_urls = []
                    for img in images:
                        if isinstance(img, dict):
                            url = img.get("large") or img.get("medium") or img.get("small", "")
                            if url:
                                image_urls.append(url)
                        elif isinstance(img, str):
                            image_urls.append(img)
                    
                    item = Item(
                        item_id=item_id,
                        title=data.get("title", ""),
                        description=description,
                        categories=data.get("categories", []),
                        images=image_urls,
                        avg_rating=float(data.get("average_rating", 0)),
                        num_ratings=int(data.get("rating_number", 0)),
                    )
                    items[item_id] = item
                except Exception as e:
                    continue
        
        return items
    
    def build_mappings(self, interactions: List[Interaction]):
        """构建用户和物品的ID映射"""
        users = set()
        items = set()
        
        for interaction in interactions:
            users.add(interaction.user_id)
            items.add(interaction.item_id)
        
        self.user2id = {u: i for i, u in enumerate(sorted(users))}
        self.item2id = {i: j for j, i in enumerate(sorted(items))}
        self.id2user = {i: u for u, i in self.user2id.items()}
        self.id2item = {j: i for i, j in self.item2id.items()}
        
        print(f"用户数: {len(self.user2id)}, 物品数: {len(self.item2id)}")
    
    def filter_interactions(self, interactions: List[Interaction], 
                           min_user_interactions: int = 5,
                           min_item_interactions: int = 5) -> List[Interaction]:
        """过滤低频交互"""
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)
        
        for interaction in interactions:
            user_counts[interaction.user_id] += 1
            item_counts[interaction.item_id] += 1
        
        filtered = [
            interaction for interaction in interactions
            if user_counts[interaction.user_id] >= min_user_interactions
            and item_counts[interaction.item_id] >= min_item_interactions
        ]
        
        print(f"过滤前: {len(interactions)}, 过滤后: {len(filtered)}")
        return filtered
    
    def split_data(self, interactions: List[Interaction], 
                   train_ratio: float = 0.8, 
                   val_ratio: float = 0.1) -> Tuple[List[Interaction], List[Interaction], List[Interaction]]:
        """划分数据集"""
        # 按时间排序
        interactions = sorted(interactions, key=lambda x: x.timestamp)
        
        n = len(interactions)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = interactions[:train_end]
        val_data = interactions[train_end:val_end]
        test_data = interactions[val_end:]
        
        print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")
        return train_data, val_data, test_data


class MultimodalDataset(Dataset):
    """多模态数据集"""
    
    def __init__(self, 
                 interactions: List[Interaction],
                 items: Dict[str, Item],
                 user2id: Dict[str, int],
                 item2id: Dict[str, int],
                 image_dir: Path = None,
                 image_size: int = 224,
                 max_text_length: int = 128,
                 is_train: bool = True,
                 transform=None):
        
        self.interactions = interactions
        self.items = items
        self.user2id = user2id
        self.item2id = item2id
        self.image_dir = image_dir
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.is_train = is_train
        
        # 图像变换
        if transform:
            self.transform = transform
        else:
            if is_train:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225]),
                ])
        
        # 默认图像（当图像不存在时使用）
        self.default_image = torch.zeros(3, image_size, image_size)
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        interaction = self.interactions[idx]
        
        # 用户ID和物品ID
        user_id = self.user2id.get(interaction.user_id, 0)
        item_id = self.item2id.get(interaction.item_id, 0)
        
        # 标签（评分 >= 4 视为正样本）
        label = 1.0 if interaction.rating >= 4.0 else 0.0
        
        # 加载图像
        image = self._load_image(interaction.item_id)
        
        # 获取文本
        item = self.items.get(interaction.item_id)
        if item:
            text = f"{item.title} {item.description} {interaction.text}"
        else:
            text = interaction.text
        
        # 截断文本
        text = text[:self.max_text_length]
        
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "item_id": torch.tensor(item_id, dtype=torch.long),
            "image": image,
            "text": text,
            "label": torch.tensor(label, dtype=torch.float),
            "rating": torch.tensor(interaction.rating, dtype=torch.float),
        }
    
    def _load_image(self, item_id: str) -> torch.Tensor:
        """加载图像"""
        if self.image_dir:
            image_path = self.image_dir / f"{item_id}.jpg"
            if image_path.exists():
                try:
                    img = Image.open(image_path).convert('RGB')
                    return self.transform(img)
                except Exception:
                    pass
        
        return self.default_image


class TextAugmenter:
    """文本增强器"""
    
    def __init__(self, augmentation_ratio: float = 0.2):
        self.augmentation_ratio = augmentation_ratio
        
        # 停用词（用于随机删除）
        self.stopwords = set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 
                              'be', 'been', 'being', 'have', 'has', 'had',
                              'do', 'does', 'did', 'will', 'would', 'could'])
    
    def random_delete(self, text: str, p: float = 0.1) -> str:
        """随机删除单词"""
        words = text.split()
        if len(words) == 0:
            return text
        
        new_words = []
        for word in words:
            if word.lower() not in self.stopwords and random.random() > p:
                new_words.append(word)
        
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """随机交换单词"""
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def augment(self, text: str) -> str:
        """执行数据增强"""
        if random.random() > self.augmentation_ratio:
            return text
        
        aug_type = random.choice(['delete', 'swap'])
        
        if aug_type == 'delete':
            return self.random_delete(text)
        else:
            return self.random_swap(text)


class HardNegativeMiner:
    """难例挖掘器"""
    
    def __init__(self, hard_negative_ratio: float = 0.3):
        self.hard_negative_ratio = hard_negative_ratio
    
    def mine(self, 
             user_embeddings: np.ndarray,
             item_embeddings: np.ndarray,
             positive_items: Dict[int, List[int]],
             num_negatives: int = 4) -> Dict[int, List[int]]:
        """
        挖掘难负例
        
        Args:
            user_embeddings: 用户嵌入 [num_users, embed_dim]
            item_embeddings: 物品嵌入 [num_items, embed_dim]
            positive_items: 用户正样本物品字典
            num_negatives: 每个用户需要的负例数
        
        Returns:
            每个用户的难负例字典
        """
        hard_negatives = {}
        
        # 计算相似度
        similarities = np.dot(user_embeddings, item_embeddings.T)
        
        for user_idx in range(len(user_embeddings)):
            user_sims = similarities[user_idx]
            pos_items = set(positive_items.get(user_idx, []))
            
            # 按相似度排序（高相似度的负例更难）
            sorted_items = np.argsort(-user_sims)
            
            # 选择难负例
            hard_negs = []
            num_hard = int(num_negatives * self.hard_negative_ratio)
            num_random = num_negatives - num_hard
            
            for item_idx in sorted_items:
                if item_idx not in pos_items:
                    hard_negs.append(item_idx)
                    if len(hard_negs) >= num_hard:
                        break
            
            # 补充随机负例
            all_negatives = [i for i in range(len(item_embeddings)) if i not in pos_items]
            random_negs = random.sample(all_negatives, min(num_random, len(all_negatives)))
            
            hard_negatives[user_idx] = hard_negs + random_negs
        
        return hard_negatives


def create_dataloaders(config: DataConfig, 
                       processor: DataProcessor,
                       interactions: List[Interaction],
                       items: Dict[str, Item],
                       batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    
    # 划分数据
    train_data, val_data, test_data = processor.split_data(
        interactions, 
        config.train_ratio, 
        config.val_ratio
    )
    
    # 图像目录
    image_dir = Path(config.data_dir) / "beauty_images"
    
    # 创建数据集
    train_dataset = MultimodalDataset(
        train_data, items, processor.user2id, processor.item2id,
        image_dir=image_dir, image_size=config.image_size,
        max_text_length=config.max_text_length, is_train=True
    )
    
    val_dataset = MultimodalDataset(
        val_data, items, processor.user2id, processor.item2id,
        image_dir=image_dir, image_size=config.image_size,
        max_text_length=config.max_text_length, is_train=False
    )
    
    test_dataset = MultimodalDataset(
        test_data, items, processor.user2id, processor.item2id,
        image_dir=image_dir, image_size=config.image_size,
        max_text_length=config.max_text_length, is_train=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader, test_loader
