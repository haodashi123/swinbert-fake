import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from faker import Faker
import random
from datetime import datetime, timedelta
import torch
import os
from pathlib import Path

def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(2026)

# Initialize Faker with Chinese locale
fake = Faker('zh_CN')

class MockDataLoader:
    @staticmethod
    @st.cache_data
    def load_fakeddit_sample(n=500):
        """
        加载 Fakeddit 测试集样本
        """
        repo_root = Path(__file__).resolve().parents[2]
        csv_path = repo_root / "data" / "dataset_test.csv"
        tsv_path = repo_root / "fakeddit_multimodal_only_samples" / "multimodal_test_public.tsv"
        try:
            if csv_path.exists():
                df = pd.read_csv(str(csv_path), on_bad_lines="skip")
            else:
                df = pd.read_csv(str(tsv_path), sep="\t", on_bad_lines="skip")
            
            # 随机采样 n 条
            if len(df) > n:
                df = df.sample(n=n, random_state=2026)
            
            samples = []
            for _, row in df.iterrows():
                # 提取基础字段
                title = str(row.get('clean_title', '无标题'))
                img_url = str(row.get('image_url', ''))
                
                # Check if image_url is a local ID (e.g. "1a0k74") or a full URL
                if not img_url.startswith("http"):
                    # Assuming it's an ID, construct local path
                    # Check if ID has extension
                    if not img_url.endswith(".jpg"):
                         img_url += ".jpg"
                    
                    # Construct absolute local path
                    local_path = f"data/images/{img_url}"
                    if os.path.exists(local_path):
                        img_url = os.path.abspath(local_path)
                    else:
                        # Fallback placeholder if local image missing
                        img_url = "https://via.placeholder.com/300?text=Image+Not+Found"
                
                # Label: 1 (Real) or 0 (Fake). Handle potential parsing errors.
                try:
                    label = int(row.get("2_way_label", 0))
                except:
                    label = 0
                
                sample = {
                    "clean_title": title,
                    "image_url": img_url,
                    "label": label, # Ground Truth
                    "simulated_result": {
                        "generated_caption": f"BLIP 模型对图片内容 ({title[:10]}...) 的生成描述..."
                    }
                }
                samples.append(sample)
                
            return samples
            
        except Exception as e:
            # 回退逻辑：生成纯随机数据
            st.warning(f"无法读取本地 Fakeddit 数据集 ({e})，已切换至随机模拟模式。")
            fallback_samples = []
            for i in range(n):
                fallback_samples.append({
                    "clean_title": f"模拟样本新闻标题 {i+1}",
                    "image_url": "https://via.placeholder.com/300",
                    "label": random.randint(0, 1),
                    "simulated_result": {
                        "generated_caption": "模拟生成的图片描述文本"
                    }
                })
            return fallback_samples
