#!/usr/bin/env python3
"""
下载 Amazon-2023 Beauty 和 Sports 数据集
直接从 HuggingFace Hub 下载原始 JSONL 文件
"""
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import requests
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from io import BytesIO
import gzip

# 数据保存目录
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace 数据集信息
HF_DATASET = "McAuley-Lab/Amazon-Reviews-2023"

# 数据文件路径
DATA_FILES = {
    "Beauty": {
        "reviews": "raw/review_categories/Beauty_and_Personal_Care.jsonl",
        "meta": "raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl",
    },
    "Sports": {
        "reviews": "raw/review_categories/Sports_and_Outdoors.jsonl",
        "meta": "raw/meta_categories/meta_Sports_and_Outdoors.jsonl",
    },
    "All_Beauty": {
        "reviews": "raw/review_categories/All_Beauty.jsonl",
        "meta": "raw/meta_categories/meta_All_Beauty.jsonl",
    },
}


def download_file_from_hf(filename: str, save_dir: Path) -> Path:
    """
    从 HuggingFace Hub 下载文件
    
    Args:
        filename: HuggingFace 上的文件路径
        save_dir: 本地保存目录
    
    Returns:
        本地文件路径
    """
    print(f"下载: {filename}")
    
    try:
        local_path = hf_hub_download(
            repo_id=HF_DATASET,
            filename=filename,
            repo_type="dataset",
            local_dir=save_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✅ 已保存到: {local_path}")
        return Path(local_path)
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None


def process_reviews(input_file: Path, output_file: Path, max_samples: int = None):
    """
    处理评论数据，提取关键字段
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        max_samples: 最大样本数
    """
    print(f"\n处理评论数据: {input_file}")
    
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="处理评论"):
            try:
                item = json.loads(line.strip())
                
                # 提取关键字段
                review = {
                    "user_id": item.get("user_id", ""),
                    "asin": item.get("asin", ""),  # 产品ID
                    "rating": item.get("rating", 0),
                    "title": item.get("title", ""),
                    "text": item.get("text", ""),
                    "timestamp": item.get("timestamp", 0),
                    "helpful_vote": item.get("helpful_vote", 0),
                    "verified_purchase": item.get("verified_purchase", False),
                }
                f_out.write(json.dumps(review, ensure_ascii=False) + '\n')
                count += 1
                
                if max_samples and count >= max_samples:
                    break
                    
            except json.JSONDecodeError:
                continue
    
    print(f"✅ 处理完成: {count} 条评论")
    return count


def process_metadata(input_file: Path, output_file: Path, max_samples: int = None):
    """
    处理产品元数据，提取关键字段（包含图像URL）
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        max_samples: 最大样本数
    """
    print(f"\n处理元数据: {input_file}")
    
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="处理元数据"):
            try:
                item = json.loads(line.strip())
                
                # 提取关键字段
                meta = {
                    "asin": item.get("asin", ""),
                    "title": item.get("title", ""),
                    "main_category": item.get("main_category", ""),
                    "categories": item.get("categories", []),
                    "description": item.get("description", []),
                    "price": item.get("price", None),
                    "average_rating": item.get("average_rating", 0),
                    "rating_number": item.get("rating_number", 0),
                    "images": item.get("images", []),  # 图像URL列表
                    "features": item.get("features", []),
                    "details": item.get("details", {}),
                    "parent_asin": item.get("parent_asin", ""),
                }
                f_out.write(json.dumps(meta, ensure_ascii=False) + '\n')
                count += 1
                
                if max_samples and count >= max_samples:
                    break
                    
            except json.JSONDecodeError:
                continue
    
    print(f"✅ 处理完成: {count} 条元数据")
    return count


def download_sample_images(meta_file: Path, img_dir: Path, num_images: int = 100):
    """
    下载样本图像
    
    Args:
        meta_file: 元数据文件路径
        img_dir: 图像保存目录
        num_images: 下载图像数量
    """
    print(f"\n下载样本图像...")
    
    img_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="下载图像"):
            if count >= num_images:
                break
            
            try:
                item = json.loads(line.strip())
                images = item.get("images", [])
                asin = item.get("asin", "unknown")
                
                if images and len(images) > 0:
                    # 获取图像URL (优先大图)
                    img_info = images[0]
                    img_url = img_info.get("large") or img_info.get("medium") or img_info.get("small", "")
                    
                    if img_url and isinstance(img_url, str):
                        # 下载图像
                        response = requests.get(img_url, timeout=10)
                        if response.status_code == 200:
                            img = Image.open(BytesIO(response.content))
                            img_path = img_dir / f"{asin}.jpg"
                            img.save(img_path, "JPEG")
                            count += 1
                            
            except Exception as e:
                continue
    
    print(f"✅ 已下载 {count} 张图像到 {img_dir}")
    return count


def get_dataset_stats(data_dir: Path):
    """
    获取数据集统计信息
    """
    print(f"\n{'='*60}")
    print("数据集统计信息")
    print(f"{'='*60}")
    
    for category in ["beauty", "sports", "all_beauty"]:
        review_file = data_dir / f"{category}_reviews.jsonl"
        meta_file = data_dir / f"{category}_metadata.jsonl"
        img_dir = data_dir / f"{category}_images"
        
        review_count = 0
        meta_count = 0
        img_count = 0
        
        if review_file.exists():
            with open(review_file, 'r') as f:
                review_count = sum(1 for _ in f)
        
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta_count = sum(1 for _ in f)
        
        if img_dir.exists():
            img_count = len(list(img_dir.glob("*.jpg")))
        
        if review_count > 0 or meta_count > 0:
            print(f"\n{category.upper()}:")
            print(f"  - 评论数: {review_count:,}")
            print(f"  - 产品数: {meta_count:,}")
            print(f"  - 图像数: {img_count:,}")


def main():
    parser = argparse.ArgumentParser(description="下载 Amazon-2023 数据集")
    parser.add_argument("--category", type=str, 
                        choices=["Beauty", "Sports", "All_Beauty", "all"], 
                        default="all", help="数据集类别")
    parser.add_argument("--max_reviews", type=int, default=None, 
                        help="最大评论数 (用于测试)")
    parser.add_argument("--max_meta", type=int, default=None, 
                        help="最大元数据数 (用于测试)")
    parser.add_argument("--download_images", action="store_true", 
                        help="是否下载图像")
    parser.add_argument("--num_images", type=int, default=100, 
                        help="下载图像数量")
    parser.add_argument("--skip_download", action="store_true",
                        help="跳过下载，只处理已有数据")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Amazon-2023 数据集下载工具")
    print("="*60)
    print(f"数据保存目录: {DATA_DIR}")
    
    categories = list(DATA_FILES.keys()) if args.category == "all" else [args.category]
    
    for category in categories:
        print(f"\n{'='*60}")
        print(f"处理 {category} 数据集")
        print(f"{'='*60}")
        
        files = DATA_FILES[category]
        
        # 下载评论数据
        if not args.skip_download:
            raw_review_path = download_file_from_hf(files["reviews"], DATA_DIR)
            raw_meta_path = download_file_from_hf(files["meta"], DATA_DIR)
        else:
            raw_review_path = DATA_DIR / files["reviews"]
            raw_meta_path = DATA_DIR / files["meta"]
        
        # 处理评论数据
        if raw_review_path and raw_review_path.exists():
            processed_review_path = DATA_DIR / f"{category.lower()}_reviews.jsonl"
            process_reviews(raw_review_path, processed_review_path, args.max_reviews)
        
        # 处理元数据
        if raw_meta_path and raw_meta_path.exists():
            processed_meta_path = DATA_DIR / f"{category.lower()}_metadata.jsonl"
            process_metadata(raw_meta_path, processed_meta_path, args.max_meta)
            
            # 下载图像
            if args.download_images and processed_meta_path.exists():
                img_dir = DATA_DIR / f"{category.lower()}_images"
                download_sample_images(processed_meta_path, img_dir, args.num_images)
    
    # 显示统计信息
    get_dataset_stats(DATA_DIR)
    
    print("\n✅ 数据下载完成!")


if __name__ == "__main__":
    main()
