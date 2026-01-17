import os
import numpy as np
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side
from tqdm import tqdm

def main():
    # 1. 获取绝对路径，防止找不到文件夹
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_DIR = os.path.join(ROOT_DIR, "gallery_images")
    WEIGHTS_PATH = os.path.join(ROOT_DIR, "vit-dinov2-base.npz")
    
    # 2. 检查环境
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ 错误：在根目录找不到权重文件 {WEIGHTS_PATH}")
        return
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ 错误：找不到图片目录 {IMAGE_DIR}")
        return

    # 3. 加载模型
    print("正在加载模型并初始化权重映射...")
    try:
        weights = np.load(WEIGHTS_PATH)
        model = Dinov2Numpy(weights)
    except Exception as e:
        print(f"❌ 加载模型失败，可能是 dinov2_numpy.py 里的键名没对上: {e}")
        return

    # 4. 扫描图片
    # 支持多种图片格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)])
    
    if not files:
        print(f"⚠️ 警告：{IMAGE_DIR} 文件夹内没有发现图片！")
        return

    print(f"🚀 准备提取 {len(files)} 张图片的特征...")
    
    all_features = []
    image_names = []

    # 5. 循环提取特征
    for filename in tqdm(files):
        img_path = os.path.join(IMAGE_DIR, filename)
        try:
            # 预处理
            img_tensor = resize_short_side(img_path, target_size=224)
            
            # 模型推理
            # 注意：最新版 Dinov2Numpy 返回 (1, 768)，我们需要取 [0] 变成 (768,)
            feat = model(img_tensor)
            if isinstance(feat, np.ndarray) and feat.ndim > 1:
                feat = feat[0]
            
            # 归一化 (特征工程的关键，确保搜图准确)
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                feat = feat / norm
            
            all_features.append(feat)
            image_names.append(filename)
            
        except Exception as e:
            # 打印具体哪张图报错，方便排查
            print(f"\n❌ 处理 {filename} 时出错: {e}")
            continue

    # 6. 保存结果
    if all_features:
        feat_arr = np.array(all_features)
        name_arr = np.array(image_names)
        
        # 保存到当前目录
        np.save(os.path.join(ROOT_DIR, "gallery_features.npy"), feat_arr)
        np.save(os.path.join(ROOT_DIR, "gallery_names.npy"), name_arr)
        
        print(f"\n✅ 特征提取成功！")
        print(f"📊 最终特征库维度: {feat_arr.shape}")
        print(f"💾 已保存 gallery_features.npy 和 gallery_names.npy")
    else:
        print("❌ 未提取到任何有效特征，请检查图片是否损坏。")

if __name__ == "__main__":
    main()