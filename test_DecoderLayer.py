import torch
import unittest
from Attention import apply_rotary_emb, precompute_freqs_cis, repeat_kv, Attention
from DecoderLayer import DecoderLayer
from ModelConfig import ModelConfig


class TestDecoderLayer(unittest.TestCase):
    def test_de_forward(self):
        # 假设 ModelConfig 是一个 dataclass 或者普通类，以下是常见参数的模拟：
        args = ModelConfig()
        # 创建LLaMADecoderLayer实例
        decoderlayer = DecoderLayer(0, args)

        # 模拟输入数据
        dim = args.dim
        seq_len = 50

        x = torch.randn(1, seq_len, dim)  # [bs, seq_len, dim]

        freqs_cos, freqs_sin = precompute_freqs_cis(dim // args.n_heads, seq_len)

        out = decoderlayer(x, freqs_cos, freqs_sin)

        print(out.shape)  # 形状和输入的x一样 [batch_size, seq_len, dim]

if __name__ == '__main__':
    unittest.main()
