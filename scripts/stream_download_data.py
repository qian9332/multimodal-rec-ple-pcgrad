#!/usr/bin/env python3
"""
流式下载 Amazon-2023 数据集
只下载需要的样本数量，节省磁盘空间
"""
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import requests
from huggingface_hub import hf_hub_url
from PIL import Image
from io import BytesIO
import urllib.request

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


def stream_download_jsonl(filename: str, output_file: Path, max_samples: int = None, 
                          process_func=None):
    """
    流式下载 JSONL 文件
    
    Args:
        filename: HuggingFace 上的文件路径
        output_file: 本地输出文件
        max_samples: 最大样本数
        process_func: 处理函数
    """
    # 构建 URL
    url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{filename}"
    print(f"流式下载: {url}")
    
    count = 0
    
    try:
        # 使用 urllib 流式下载
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req, timeout=300) as response:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                buffer = ""
                
                with tqdm(desc="下载中", unit="行") as pbar:
                    while True:
                        chunk = response.read(65536)  # 64KB chunks
                        if not chunk:
                            break
                        
                        buffer += chunk.decode('utf-8', errors='ignore')
                        
                        # 处理完整的行
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            if line.strip():
                                try:
                                    item = json.loads(line.strip())
                                    
                                    if process_func:
                                        processed = process_func(item)
                                    else:
                                        processed = item
                                    
                                    if processed:
                                        f_out.write(json.dumps(processed, ensure_ascii=False) + '\n')
                                        count += 1
                                        pbar.update(1)
                                        
                                        if max_samples and count >= max_samples:
                                            print(f"\n已达到最大样本数: {max_samples}")
                                            return count
                                            
                                except json.JSONDecodeError:
                                    continue
        
        print(f"✅ 已保存 {count} 条记录到 {output_file}")
        return count
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return count


def process_review_item(item):
    """处理评论数据"""
    return {
        "user_id": item.get("user_id", ""),
        "asin": item.get("asin", ""),
        "rating": item.get("rating", 0),
        "title": item.get("title", ""),
        "text": item.get("text", ""),
        "timestamp": item.get("timestamp", 0),
        "helpful_vote": item.get("helpful_vote", 0),
        "verified_purchase": item.get("verified_purchase", False),
    }


def process_meta_item(item):
    """处理元数据"""
    return {
        "asin": item.get("asin", ""),
        "title": item.get("title", ""),
        "main_category": item.get("main_category", ""),
        "categories": item.get("categories", []),
        "description": item.get("description", []),
        "price": item.get("price", None),
        "average_rating": item.get("average_rating", 0),
        "rating_number": item.get("rating_number", 0),
        "images": item.get("images", []),
        "features": item.get("features", []),
        "details": item.get("details", {}),
        "parent_asin": item.get("parent_asin", ""),
    }


def download_sample_images(meta_file: Path, img_dir: Path, num_images: int = 100):
    """下载样本图像"""
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
                    img_info = images[0]
                    img_url = img_info.get("large") or img_info.get("medium") or img_info.get("small", "")
                    
                    if img_url and isinstance(img_url, str):
                        response = requests.get(img_url, timeout=10)
                        if response.status_code == 200:
                            img = Image.open(BytesIO(response.content))
                            img_path = img_dir / f"{asin}.jpg"
                            img.save(img_path, "JPEG")
                            count += 1
                            
            except Exception:
                continue
    
    print(f"✅ 已下载 {count} 张图像到 {img_dir}")
    return count


def get_dataset_stats(data_dir: Path):
    """获取数据集统计信息"""
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
    parser = argparse.ArgumentParser(description="流式下载 Amazon-2023 数据集")
    parser.add_argument("--category", type=str, 
                        choices=["Beauty", "Sports", "All_Beauty", "all"], 
                        default="all", help="数据集类别")
    parser.add_argument("--max_reviews", type=int, default=10000, 
                        help="最大评论数")
    parser.add_argument("--max_meta", type=int, default=5000, 
                        help="最大元数据数")
    parser.add_argument("--download_images", action="store_true", 
                        help="是否下载图像")
    parser.add_argument("--num_images", type=int, default=100, 
                        help="下载图像数量")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Amazon-2023 数据集流式下载工具")
    print("="*60)
    print(f"数据保存目录: {DATA_DIR}")
    
    categories = list(DATA_FILES.keys()) if args.category == "all" else [args.category]
    
    for category in categories:
        print(f"\n{'='*60}")
        print(f"处理 {category} 数据集")
        print(f"{'='*60}")
        
        files = DATA_FILES[category]
        
        # 流式下载评论数据
        review_output = DATA_DIR / f"{category.lower()}_reviews.jsonl"
        stream_download_jsonl(
            files["reviews"], 
            review_output, 
            args.max_reviews,
            process_review_item
        )
        
        # 流式下载元数据
        meta_output = DATA_DIR / f"{category.lower()}_metadata.jsonl"
        stream_download_jsonl(
            files["meta"], 
            meta_output, 
            args.max_meta,
            process_meta_item
        )
        
        # 下载图像
        if args.download_images and meta_output.exists():
            img_dir = DATA_DIR / f"{category.lower()}_images"
            download_sample_images(meta_output, img_dir, args.num_images)
    
    # 显示统计信息
    get_dataset_stats(DATA_DIR)
    
    print("\n✅ 数据下载完成!")


if __name__ == "__main__":
    main()
