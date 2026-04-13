import torch
import unittest
from Attention import apply_rotary_emb, precompute_freqs_cis, repeat_kv, Attention
from ModelConfig import ModelConfig

class TestAttentionComponents(unittest.TestCase):
    """注意力机制相关组件的单元测试类"""
    def test_apply_rotary_emb(self):
        """
        测试apply_rotary_emb函数
        :return:
        """
        xq = torch.randn(1, 50, 6, 48)  # bs, seq_len, dim//n_head, n_head_dim
        xk = torch.randn(1, 50, 6, 48)  # bs, seq_len, dim//n_head, n_head_dim

        # 使用 precompute_freqs_cis 函数获取 sin和cos
        cos, sin = precompute_freqs_cis(288 // 6, 50)
        print(cos.shape, sin.shape)
        xq_out, xk_out = apply_rotary_emb(xq, xk, cos, sin)

        print((xq_out.shape, xk_out.shape))

    def test_attention(self):
        # 创建Attention实例
        # 假设 ModelConfig 是一个 dataclass 或者普通类，以下是常见参数的模拟：
        args = ModelConfig()
        attention_model = Attention(args)

        # 模拟输入数据
        batch_size = 1
        seq_len = 50  # 假设实际使用的序列长度为50
        dim = args.dim
        x = torch.rand(batch_size, seq_len, dim)  # 随机生成输入张量
        # freqs_cos = torch.rand(seq_len, dim // 2)  # 模拟cos频率，用于RoPE
        # freqs_sin = torch.rand(seq_len, dim // 2)  # 模拟sin频率，用于RoPE

        freqs_cos, freqs_sin = precompute_freqs_cis(dim // args.n_heads, seq_len)

        # 运行Attention模型
        output = attention_model(x, freqs_cos, freqs_sin)

        # attention出来之后的形状 依然是[batch_size, seq_len, dim]
        print("Output shape:", output.shape)

if __name__ == '__main__':
    unittest.main()