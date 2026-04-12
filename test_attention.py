import torch
from Attention import apply_rotary_emb, precompute_freqs_cis, repeat_kv


if __name__ == '__main__':
    xq = torch.randn(1, 50, 6, 48)  # bs, seq_len, dim//n_head, n_head_dim
    xk = torch.randn(1, 50, 6, 48)  # bs, seq_len, dim//n_head, n_head_dim

    # 使用 precompute_freqs_cis 函数获取 sin和cos
    cos, sin = precompute_freqs_cis(288 // 6, 50)
    print(cos.shape, sin.shape)
    xq_out, xk_out = apply_rotary_emb(xq, xk, cos, sin)

    print((xq_out.shape, xk_out.shape))
