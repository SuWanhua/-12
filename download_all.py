import pandas as pd
import requests
import os
from tqdm import tqdm

# 配置
CSV_PATH = "data.csv"
SAVE_DIR = "./gallery_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. 读取数据
df = pd.read_csv(CSV_PATH)
# 注意：这里必须使用截图中的列名 'image_url'
urls = df['image_url'].tolist()[:10000] 

# 模拟浏览器请求头
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

print(f"开始下载 10,000 张图片至 {SAVE_DIR}...")

for i, url in enumerate(tqdm(urls)):
    img_name = f"{i}.jpg"
    img_path = os.path.join(SAVE_DIR, img_name)
    
    # 断点续传：如果文件已存在则跳过
    if os.path.exists(img_path):
        continue
        
    try:
        # 设置超时：3秒连接，10秒读取
        response = requests.get(url, headers=headers, timeout=(3, 10))
        if response.status_code == 200:
            with open(img_path, 'wb') as f:
                f.write(response.content)
    except Exception:
        # 遇到死链或下载失败自动跳过
        continue

print(f"下载结束。当前成功下载图片数：{len(os.listdir(SAVE_DIR))}")