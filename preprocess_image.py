import numpy as np
from PIL import Image

def center_crop(img_path, crop_size=224):
    # Step 1: load image
    image = Image.open(img_path).convert("RGB")

    # Step 2: center crop
    w, h = image.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))  # PIL Image, size (224, 224)

    # Step 3: to_numpy
    image = np.array(image).astype(np.float32) / 255.0  # (H, W, C)

    # Step 4: norm
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std  # (H, W, C)
    image = image.transpose(2, 0, 1) # (C, H, W)
    return image[None] # (1, C, H, W)

# ************* ToDo, resize short side *************p

def resize_short_side(img_path, target_size=224, patch_size=14):
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    print(f"Original size: {w}x{h}")  # 打印原始尺寸
    # 计算新的宽高
    if w < h:
        new_w = target_size
        new_h = int(h * (target_size / w + 0.5) // 1) * 1  # 确保是整数
    else:
        new_h = target_size
        new_w = int(w * (target_size / h + 0.5) // 1) * 1  # 确保是整数
    # 确保宽高是 patch_size 的倍数
    new_w = (new_w + patch_size - 1) // patch_size * patch_size
    new_h = (new_h + patch_size - 1) // patch_size * patch_size
    print(f"Resized size: {new_w}x{new_h}")  # 打印调整后的尺寸
    image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    # 转换为 numpy 数组并归一化
    image = np.array(image).astype(np.float32) / 255.0  # (H, W, C)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std  # (H, W, C)
    image = image.transpose(2, 0, 1)  # (C, H, W)
    return image[None]  # (1, C, H, W)