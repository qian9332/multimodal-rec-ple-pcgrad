#!/usr/bin/env python3
"""
数据分析脚本
分析 Amazon-2023 数据集的分布和统计信息
"""
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_jsonl(filepath: Path, max_samples: int = None) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                data.append(json.loads(line.strip()))
            except:
                continue
    return data


def analyze_reviews(reviews: List[Dict]) -> Dict:
    """分析评论数据"""
    print("\n" + "="*60)
    print("评论数据分析")
    print("="*60)
    
    # 基本统计
    print(f"\n总评论数: {len(reviews):,}")
    
    # 用户统计
    user_counts = Counter(r.get("user_id") for r in reviews)
    print(f"唯一用户数: {len(user_counts):,}")
    print(f"平均每用户评论数: {np.mean(list(user_counts.values())):.2f}")
    print(f"评论数中位数: {np.median(list(user_counts.values())):.2f}")
    print(f"最多评论用户: {max(user_counts.values())} 条")
    print(f"最少评论用户: {min(user_counts.values())} 条")
    
    # 物品统计
    item_counts = Counter(r.get("asin") for r in reviews)
    print(f"\n唯一物品数: {len(item_counts):,}")
    print(f"平均每物品评论数: {np.mean(list(item_counts.values())):.2f}")
    print(f"评论数中位数: {np.median(list(item_counts.values())):.2f}")
    
    # 评分分布
    ratings = [r.get("rating", 0) for r in reviews]
    rating_counts = Counter(ratings)
    print(f"\n评分分布:")
    for rating in sorted(rating_counts.keys()):
        count = rating_counts[rating]
        pct = count / len(reviews) * 100
        bar = "█" * int(pct / 2)
        print(f"  {rating}星: {count:,} ({pct:.1f}%) {bar}")
    
    # 正负样本比例 (>=4 为正样本)
    positive = sum(1 for r in ratings if r >= 4)
    negative = len(ratings) - positive
    print(f"\n正负样本比例:")
    print(f"  正样本 (>=4星): {positive:,} ({positive/len(reviews)*100:.1f}%)")
    print(f"  负样本 (<4星): {negative:,} ({negative/len(reviews)*100:.1f}%)")
    
    # 文本长度统计
    text_lengths = [len(r.get("text", "")) for r in reviews]
    print(f"\n评论文本长度统计:")
    print(f"  平均长度: {np.mean(text_lengths):.1f} 字符")
    print(f"  中位数: {np.median(text_lengths):.1f} 字符")
    print(f"  最大长度: {max(text_lengths):,} 字符")
    print(f"  最小长度: {min(text_lengths):,} 字符")
    
    # 验证购买
    verified = sum(1 for r in reviews if r.get("verified_purchase", False))
    print(f"\n验证购买: {verified:,} ({verified/len(reviews)*100:.1f}%)")
    
    # 时间分布
    timestamps = [r.get("timestamp", 0) for r in reviews if r.get("timestamp")]
    if timestamps:
        min_time = min(timestamps)
        max_time = max(timestamps)
        from datetime import datetime
        print(f"\n时间范围:")
        print(f"  最早: {datetime.fromtimestamp(min_time/1000).strftime('%Y-%m-%d')}")
        print(f"  最晚: {datetime.fromtimestamp(max_time/1000).strftime('%Y-%m-%d')}")
    
    return {
        "total_reviews": len(reviews),
        "unique_users": len(user_counts),
        "unique_items": len(item_counts),
        "avg_user_reviews": np.mean(list(user_counts.values())),
        "avg_item_reviews": np.mean(list(item_counts.values())),
        "positive_ratio": positive / len(reviews),
        "avg_text_length": np.mean(text_lengths),
    }


def analyze_metadata(items: List[Dict]) -> Dict:
    """分析元数据"""
    print("\n" + "="*60)
    print("元数据分析")
    print("="*60)
    
    print(f"\n总物品数: {len(items):,}")
    
    # 有图像的物品
    items_with_images = sum(1 for i in items if i.get("images"))
    print(f"有图像的物品: {items_with_images:,} ({items_with_images/len(items)*100:.1f}%)")
    
    # 图像数量统计
    image_counts = [len(i.get("images", [])) for i in items]
    print(f"\n图像数量统计:")
    print(f"  平均: {np.mean(image_counts):.2f}")
    print(f"  中位数: {np.median(image_counts):.0f}")
    print(f"  最多: {max(image_counts)} 张")
    
    # 评分分布
    avg_ratings = [i.get("average_rating", 0) for i in items]
    print(f"\n物品平均评分:")
    print(f"  均值: {np.mean(avg_ratings):.2f}")
    print(f"  中位数: {np.median(avg_ratings):.2f}")
    
    # 评论数分布
    rating_nums = [i.get("rating_number", 0) for i in items]
    print(f"\n物品评论数:")
    print(f"  均值: {np.mean(rating_nums):.1f}")
    print(f"  中位数: {np.median(rating_nums):.0f}")
    print(f"  最多: {max(rating_nums):,}")
    
    # 价格分布
    prices = [i.get("price") for i in items if i.get("price") is not None]
    if prices:
        print(f"\n价格统计:")
        print(f"  有价格物品: {len(prices):,}")
        print(f"  均值: ${np.mean(prices):.2f}")
        print(f"  中位数: ${np.median(prices):.2f}")
    
    # 类别分布
    categories = []
    for i in items:
        cats = i.get("categories", [])
        if cats:
            categories.extend(cats if isinstance(cats, list) else [cats])
    
    if categories:
        cat_counter = Counter(categories)
        print(f"\nTop 10 类别:")
        for cat, count in cat_counter.most_common(10):
            print(f"  {cat}: {count:,}")
    
    # 文本描述统计
    desc_lengths = []
    for i in items:
        desc = i.get("description", [])
        if isinstance(desc, list):
            desc_lengths.append(len(" ".join(desc)))
        else:
            desc_lengths.append(len(str(desc)))
    
    print(f"\n描述文本长度:")
    print(f"  均值: {np.mean(desc_lengths):.1f} 字符")
    print(f"  中位数: {np.median(desc_lengths):.0f} 字符")
    
    return {
        "total_items": len(items),
        "items_with_images": items_with_images,
        "avg_images": np.mean(image_counts),
        "avg_rating": np.mean(avg_ratings),
        "avg_rating_number": np.mean(rating_nums),
    }


def analyze_long_tail(user_counts: Counter, item_counts: Counter) -> Dict:
    """分析长尾分布"""
    print("\n" + "="*60)
    print("长尾分布分析")
    print("="*60)
    
    # 用户长尾
    user_values = sorted(user_counts.values(), reverse=True)
    total = sum(user_values)
    cumsum = np.cumsum(user_values)
    
    # 头部用户占比
    head_10pct = int(len(user_values) * 0.1)
    head_interactions = sum(user_values[:head_10pct]) if head_10pct > 0 else 0
    
    print(f"\n用户长尾分析:")
    print(f"  头部10%用户贡献: {head_interactions/total*100:.1f}% 交互")
    print(f"  单次交互用户: {sum(1 for v in user_values if v == 1):,} ({sum(1 for v in user_values if v == 1)/len(user_values)*100:.1f}%)")
    print(f"  5次以下交互用户: {sum(1 for v in user_values if v < 5):,} ({sum(1 for v in user_values if v < 5)/len(user_values)*100:.1f}%)")
    
    # 物品长尾
    item_values = sorted(item_counts.values(), reverse=True)
    total = sum(item_values)
    
    head_10pct = int(len(item_values) * 0.1)
    head_interactions = sum(item_values[:head_10pct]) if head_10pct > 0 else 0
    
    print(f"\n物品长尾分析:")
    print(f"  头部10%物品贡献: {head_interactions/total*100:.1f}% 交互")
    print(f"  单次交互物品: {sum(1 for v in item_values if v == 1):,} ({sum(1 for v in item_values if v == 1)/len(item_values)*100:.1f}%)")
    print(f"  5次以下交互物品: {sum(1 for v in item_values if v < 5):,} ({sum(1 for v in item_values if v < 5)/len(item_values)*100:.1f}%)")
    
    # Gini 系数
    def gini_coefficient(values):
        sorted_values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n+1)) * sorted_values) - (n + 1) * sum(sorted_values)) / (n * sum(sorted_values))
    
    user_gini = gini_coefficient(user_values)
    item_gini = gini_coefficient(item_values)
    
    print(f"\nGini系数 (越接近1越不均匀):")
    print(f"  用户Gini: {user_gini:.3f}")
    print(f"  物品Gini: {item_gini:.3f}")
    
    return {
        "user_gini": user_gini,
        "item_gini": item_gini,
        "single_interaction_users": sum(1 for v in user_values if v == 1),
        "single_interaction_items": sum(1 for v in item_values if v == 1),
    }


def analyze_sparsity(user_counts: Counter, item_counts: Counter, total_interactions: int) -> Dict:
    """分析稀疏度"""
    print("\n" + "="*60)
    print("稀疏度分析")
    print("="*60)
    
    num_users = len(user_counts)
    num_items = len(item_counts)
    
    # 理论最大交互数
    max_interactions = num_users * num_items
    
    # 实际交互数
    actual_interactions = total_interactions
    
    # 稀疏度
    sparsity = 1 - (actual_interactions / max_interactions)
    
    print(f"\n用户数: {num_users:,}")
    print(f"物品数: {num_items:,}")
    print(f"理论最大交互: {max_interactions:,}")
    print(f"实际交互: {actual_interactions:,}")
    print(f"稀疏度: {sparsity:.6f} ({sparsity*100:.4f}%)")
    print(f"密度: {(1-sparsity)*100:.6f}%")
    
    return {
        "num_users": num_users,
        "num_items": num_items,
        "sparsity": sparsity,
        "density": 1 - sparsity,
    }


def main():
    data_dir = Path("/home/z/my-project/multimodal_rec/data")
    
    print("="*60)
    print("Amazon-2023 数据集分析报告")
    print("="*60)
    
    all_stats = {}
    
    # 分析 Beauty 数据
    beauty_reviews_path = data_dir / "beauty_reviews.jsonl"
    beauty_meta_path = data_dir / "beauty_metadata.jsonl"
    
    if beauty_reviews_path.exists():
        print("\n" + "#"*60)
        print("# Beauty 数据集")
        print("#"*60)
        
        reviews = load_jsonl(beauty_reviews_path)
        items = load_jsonl(beauty_meta_path)
        
        review_stats = analyze_reviews(reviews)
        item_stats = analyze_metadata(items)
        
        user_counts = Counter(r.get("user_id") for r in reviews)
        item_counts = Counter(r.get("asin") for r in reviews)
        
        long_tail_stats = analyze_long_tail(user_counts, item_counts)
        sparsity_stats = analyze_sparsity(user_counts, item_counts, len(reviews))
        
        all_stats["beauty"] = {
            "reviews": review_stats,
            "items": item_stats,
            "long_tail": long_tail_stats,
            "sparsity": sparsity_stats,
        }
    
    # 分析 Sports 数据
    sports_reviews_path = data_dir / "sports_reviews.jsonl"
    sports_meta_path = data_dir / "sports_metadata.jsonl"
    
    if sports_reviews_path.exists():
        print("\n" + "#"*60)
        print("# Sports 数据集")
        print("#"*60)
        
        reviews = load_jsonl(sports_reviews_path)
        items = load_jsonl(sports_meta_path)
        
        review_stats = analyze_reviews(reviews)
        item_stats = analyze_metadata(items)
        
        user_counts = Counter(r.get("user_id") for r in reviews)
        item_counts = Counter(r.get("asin") for r in reviews)
        
        long_tail_stats = analyze_long_tail(user_counts, item_counts)
        sparsity_stats = analyze_sparsity(user_counts, item_counts, len(reviews))
        
        all_stats["sports"] = {
            "reviews": review_stats,
            "items": item_stats,
            "long_tail": long_tail_stats,
            "sparsity": sparsity_stats,
        }
    
    # 保存分析结果
    output_path = data_dir / "data_analysis_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print(f"分析报告已保存到: {output_path}")
    print("="*60)
    
    return all_stats


if __name__ == "__main__":
    main()
