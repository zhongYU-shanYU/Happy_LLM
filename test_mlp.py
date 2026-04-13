import torch
import unittest
from MLP import MLP
from ModelConfig import ModelConfig

class TestMLP(unittest.TestCase):
    """
    测试MLP组件
    """
    def test_mlp_forward(self):
        # 假设 ModelConfig 是一个 dataclass 或者普通类，以下是常见参数的模拟：
        args = ModelConfig()
        # 创建MLP实例
        mlp = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
        # 随机生成数据
        x = torch.randn(1, 50, args.dim)
        # 运行MLP模型
        output = mlp(x)
        print(output.shape)

if __name__ == '__main__':
    unittest.main()

