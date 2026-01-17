import streamlit as st
import numpy as np
import os
from PIL import Image
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

# 模型权重文件路径
MODEL_PATH = "vit-dinov2-base.npz"
# 图片在子目录下
IMAGE_DIR = "gallery_images"

# 设置页面配置
st.set_page_config(page_title="🔍 图像检索系统", layout="wide")

# 检查模型权重文件是否存在
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ 找不到权重文件！请确保 vit-dinov2-base.npz 放在: {MODEL_PATH}")
else:
    # 加载模型
    model = Dinov2Numpy(np.load(MODEL_PATH))

    # 加载图库特征和名称
    gallery_feats_path = "gallery_features.npy"
    gallery_names_path = "gallery_names.npy"
    if os.path.exists(gallery_feats_path) and os.path.exists(gallery_names_path):
        gallery_feats = np.load(gallery_feats_path)
        gallery_names = np.load(gallery_names_path)
    else:
        st.warning("⚠️ 检测到特征库为空。请先运行 python image_retrieval.py 提取特征。")
        gallery_feats, gallery_names = None, None

    # 上传图片并搜索相似图片
    st.title("🔍 基于 Dinov2 的图像检索")
    uploaded_file = st.file_uploader("上传图片开始搜图", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        query_img = Image.open(uploaded_file)
        st.image(query_img, width=200, caption="查询图")

        with st.spinner("正在搜索..."):
            # 临时保存上传的图片
            t_path = os.path.join(os.getcwd(), "temp_q.jpg")
            query_img.convert("RGB").save(t_path)

            # 提取查询图片的特征
            q_tensor = resize_short_side(t_path)
            q_feat = model(q_tensor)[0]
            q_feat /= np.linalg.norm(q_feat)

            # 比对特征并获取最相似的十张图片
            if gallery_feats is not None and gallery_names is not None:
                scores = np.dot(gallery_feats, q_feat)
                top_indices = np.argsort(scores)[::-1][:10]  # 获取分数最高的前10个索引

                # 显示最相似的十张图片
                st.subheader("最相似的十张图片：")
                cols = st.columns(5)  # 每行显示5张图片
                for i, match_idx in enumerate(top_indices):
                    name = gallery_names[match_idx]
                    img_path = os.path.join(IMAGE_DIR, name)
                    if os.path.exists(img_path):
                        with cols[i % 5]:
                            st.image(img_path, caption=f"分值: {scores[match_idx]:.2f}")
            else:
                st.error("❌ 特征库为空，无法进行搜索！请先运行 image_retrieval.py 提取图库特征。")