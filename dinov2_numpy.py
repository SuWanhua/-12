import numpy as np
from scipy.ndimage import zoom

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

class Dinov2Numpy:
    def __init__(self, weights):
        self.w = weights
        self.keys = list(weights.files)
        self.config = {"hidden_size": 768, "num_heads": 12, "num_layers": 12}
        
        # 1. 模糊抓取基础权重
        self.cls_token = self._get("cls_token")
        self.pos_embed = self._get("position_embeddings")
        self.patch_w = self._get("patch_embeddings.projection.weight")
        self.patch_b = self._get("patch_embeddings.projection.bias")
        self.norm_w = self._get("layernorm.weight")
        self.norm_b = self._get("layernorm.bias")

        # 2. 转换 Patch 权重形状 (从 Conv2d 卷积核转为全连接矩阵)
        self.patch_w = self.patch_w.reshape(768, -1).T
        self.patch_b = self.patch_b.reshape(768, 1).T

    def _get(self, keyword):
        """模糊匹配函数：只要键名包含关键字就返回"""
        for k in self.keys:
            if keyword in k:
                return self.w[k]
        raise KeyError(f"在权重文件中找不到包含 '{keyword}' 的参数")

    def _get_layer(self, idx, keyword):
        """抓取特定层的权重，例如 layer.0 里的 q_proj"""
        # 同时兼容 encoder.layer.0 和 blocks.0 两种写法
        targets = [f"layer.{idx}.", f"blocks.{idx}."]
        for k in self.keys:
            if any(t in k for t in targets) and keyword in k:
                return self.w[k]
        raise KeyError(f"在第 {idx} 层找不到 '{keyword}'")

    def interpolate_pos_encoding(self, height, width):
        num_patches = (height // 14) * (width // 14)
        orig_pos_embed = self.pos_embed[:, 1:, :]
        orig_dim = int(np.sqrt(orig_pos_embed.shape[1]))
        orig_pos_embed = orig_pos_embed.reshape(1, orig_dim, orig_dim, 768).transpose(0, 3, 1, 2)
        new_h, new_w = height // 14, width // 14
        new_pos_embed = zoom(orig_pos_embed, (1, 1, new_h / orig_dim, new_w / orig_dim), order=1)
        new_pos_embed = new_pos_embed.transpose(0, 2, 3, 1).reshape(1, num_patches, 768)
        return np.concatenate([self.pos_embed[:, :1, :], new_pos_embed], axis=1)

    def __call__(self, pixel_values):
        B, _, H, W = pixel_values.shape
        # Patch Embedding
        x = []
        for i in range(0, H, 14):
            for j in range(0, W, 14):
                patch = pixel_values[:, :, i:i+14, j:j+14].reshape(B, -1)
                x.append(patch)
        x = np.stack(x, axis=1) @ self.patch_w + self.patch_b
        # Add CLS and Pos
        cls_t = np.tile(self.cls_token, (B, 1, 1))
        x = np.concatenate([cls_t, x], axis=1) + self.interpolate_pos_encoding(H, W)
        # Transformer Layers
        for i in range(12):
            # 获取当前层权重
            n1_w = self._get_layer(i, "norm1.weight")
            n1_b = self._get_layer(i, "norm1.bias")
            qw, qb = self._get_layer(i, "query.weight"), self._get_layer(i, "query.bias")
            kw, kb = self._get_layer(i, "key.weight"), self._get_layer(i, "key.bias")
            vw, vb = self._get_layer(i, "value.weight"), self._get_layer(i, "value.bias")
            ow, ob = self._get_layer(i, "output.dense.weight"), self._get_layer(i, "output.dense.bias")
            ls1 = self._get_layer(i, "layer_scale1.lambda1")
            
            n2_w = self._get_layer(i, "norm2.weight")
            n2_b = self._get_layer(i, "norm2.bias")
            f1w, f1b = self._get_layer(i, "mlp.fc1.weight"), self._get_layer(i, "mlp.fc1.bias")
            f2w, f2b = self._get_layer(i, "mlp.fc2.weight"), self._get_layer(i, "mlp.fc2.bias")
            ls2 = self._get_layer(i, "layer_scale2.lambda1")

            # Block Forward
            # Attention
            res = x
            x = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-6) * n1_w + n1_b
            q = (x @ qw.T + qb).reshape(B, -1, 12, 64).transpose(0, 2, 1, 3)
            k = (x @ kw.T + kb).reshape(B, -1, 12, 64).transpose(0, 2, 1, 3)
            v = (x @ vw.T + vb).reshape(B, -1, 12, 64).transpose(0, 2, 1, 3)
            attn = softmax(np.matmul(q / 8, k.transpose(0, 1, 3, 2)))
            x = res + ls1 * (np.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, -1, 768) @ ow.T + ob)
            
            # MLP
            res = x
            x = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-6) * n2_w + n2_b
            x = res + ls2 * (gelu(x @ f1w.T + f1b) @ f2w.T + f2b)

        # Final Norm
        x = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-6) * self.norm_w + self.norm_b
        return x[:, 0]