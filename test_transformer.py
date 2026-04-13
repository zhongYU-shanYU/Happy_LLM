import torch
import unittest
from Transformer import Transformer
from ModelConfig import ModelConfig


class TestMLP(unittest.TestCase):
    def test_mlp_forward(self):
        # LLaMA2Model.forward 接受两个参数，tokens和targets，其中tokens是输入的张量, 应为int类型
        x = torch.randint(0, 6144, (1, 50))  # [bs, seq_len]
        # 假设 ModelConfig 是一个 dataclass 或者普通类，以下是常见参数的模拟：
        args = ModelConfig()
        # 实例化LLaMA2Model
        model = Transformer(args=args)
        # 计算model的全部参数
        num_params = sum(p.numel() for p in model.parameters())
        print('Number of parameters:', num_params)

        out = model(x)
        print(out.logits.shape)  # [batch_size, 1, vocab_size]


if __name__ == '__main__':
    unittest.main()
