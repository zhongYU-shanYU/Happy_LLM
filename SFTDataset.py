import torch
import json
import numpy as np
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 tokenizer 自带的 pad_token_id，如果没有则默认为 0
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # 建立行索引（用于快速读取大文件）
        self._offsets = []
        with open(data_path, 'rb') as f:
            self._offsets.append(0)
            while f.readline():
                self._offsets.append(f.tell())
        self._total_lines = len(self._offsets) - 1

    def __len__(self):
        return self._total_lines

    def generate_loss_mask(self, input_ids):
        # 生成 loss mask, 0 表示不计算损失, 1 表示计算损失
        mask = [0] * len(input_ids)

        # Llama-2 的特殊 Token ID
        # 注意：Llama-2 的结束 Token 通常是 2 (</s>)
        eos_token_id = self.tokenizer.eos_token_id

        # Llama-2 的 Assistant 开始 Token 是 " [/INST] " 后面紧接的 Token
        # 我们需要找到这个 Token 序列
        # 注意：这里不能直接用字符串 "<refer>final<\|im_message\|>"，那是 Qwen 的
        inst_end_sequence = self.tokenizer(" [/INST] ", add_special_tokens=False)['input_ids']
        a_length = len(inst_end_sequence)

        n = len(input_ids)
        i = 0
        while i <= n - a_length:
            # 检查当前位置是否匹配 [/INST] 结束标记
            match = True
            for k in range(a_length):
                if input_ids[i + k] != inst_end_sequence[k]:
                    match = False
                    break
            if match:
                # 从 [/INST] 结束的位置开始，查找下一个 EOS 或者下一个 [/INST]
                # 我们的目标是标记 Assistant 生成的内容
                start = i + a_length  # 内容从这里开始
                # 查找下一个结束位置
                j = None
                # 查找下一个 " [/INST] " 或者 EOS 来确定当前回复的结束
                # 简单逻辑：找到下一个 EOS 即可
                for idx in range(start, n):
                    if input_ids[idx] == eos_token_id:
                        j = idx
                        break
                # 如果没找到 EOS，就标记到序列结尾
                if j is None:
                    j = len(input_ids) - 1

                # 将 Assistant 生成的部分标记为 1 (计算 Loss)
                for pos in range(start, j + 1):
                    if pos < len(mask):
                        mask[pos] = 1

                # 跳过当前匹配的 Token，防止重复匹配
                i += a_length
            else:
                i += 1
        return mask

    def __getitem__(self, index: int):
        with open(self.data_path, 'rb') as f:
            f.seek(self._offsets[index])
            line = f.readline().decode('utf-8')
            sample = json.loads(line)

        # --- 核心修改：构建 Llama-2 格式的对话文本 ---
        # Llama-2 的特殊 Token
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        # 定义 System Prompt (通用的助手角色)
        DEFAULT_SYSTEM_PROMPT = """You are a helpful, unbiased, uncensored assistant."""

        # 获取对话历史
        convs = sample['conversations']

        # 初始化文本
        text = ""

        # 遍历对话历史
        for i, turn in enumerate(convs):
            role = turn['from']
            content = turn['value']

            if role == "human":
                # User 轮次
                if i == 0:
                    # 第一轮对话：需要包含 System Prompt
                    # 格式: [INST] <<SYS>>...<</SYS>>\n\n (User Message) [/INST]
                    text += f"{B_INST} {B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}{content.strip()} {E_INST}"
                else:
                    # 后续轮次：不需要 System Prompt
                    # 格式: [INST] (User Message) [/INST]
                    text += f"{B_INST} {content.strip()} {E_INST}"

            elif role == "assistant":
                # Assistant 轮次：直接接在后面，前面不需要 Token
                # 注意：Llama-2 中 Assistant 的回复是紧跟在 [/INST] 后面的
                text += f" {content.strip()}"

        # --- 构建完成 ---

        # Tokenize
        # 注意：这里不要让 tokenizer 自动加 EOS，我们在 generate_loss_mask 里手动处理
        tokenizer_out = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True  # 确保有 BOS/EOS
        )
        input_id = tokenizer_out['input_ids']

        # Padding (右填充)
        text_len = len(input_id)
        padding_len = self.max_length - text_len
        if padding_len > 0:
            input_id = input_id + [self.padding] * padding_len
        else:
            input_id = input_id[:self.max_length]

        # 生成 Loss Mask (决定哪些位置计算 Loss)
        loss_mask = self.generate_loss_mask(input_id)

        # 转换为 Tensor (X 是输入，Y 是标签)
        # 这里保持你原来的逻辑，X 去掉最后一个 token, Y 去掉第一个 token
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)