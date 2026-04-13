from RMSNorm import RMSNorm
from ModelConfig import ModelConfig
import torch

# --- 测试代码 ---

if __name__ == '__main__':
    args = ModelConfig()

    # 实例化模块
    norm = RMSNorm(args.dim, args.norm_eps)

    # 创建一个随机输入张量 (Batch=1, Seq_Len=50, Dim=768)
    x = torch.randn(1, 50, args.dim)

    # 前向传播
    output = norm(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")