import random
import json
import os
# 强制使用多线程，避免 Windows 下多进程启动失败导致的单核运行
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "8"  # 根据你的CPU核心数修改，比如4或8
import glob
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from tokenizers.normalizers import NFKC
from typing import Generator

random.seed(42)


def read_texts_from_jsonl(data_path: str) -> Generator[str, None, None]:
    """读取JSONL文件或目录下的所有JSONL文件并安全提取文本数据"""
    files = []
    if os.path.isfile(data_path):
        files = [data_path]
    elif os.path.isdir(data_path):
        # 获取目录下所有 .jsonl 文件
        files = glob.glob(os.path.join(data_path, "*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files found in directory: {data_path}")
    else:
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    for file_path in files:
        print(f"Reading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    if 'text' not in data:
                        raise KeyError(f"Missing 'text' field in line {line_num}")
                    yield data['text']
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in line {line_num}")
                    continue
                except KeyError as e:
                    print(e)
                    continue


def create_tokenizer_config(save_dir: str) -> None:
    """创建完整的tokenizer配置文件"""
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # 保存主配置文件
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # 创建special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)


# def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8192) -> None:
#     """训练并保存自定义tokenizer"""
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 初始化tokenizer
#     tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
#     tokenizer.normalizer = NFKC()  # 添加文本规范化
#     tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
#     tokenizer.decoder = decoders.ByteLevel()
#
#     # 配置特殊token
#     special_tokens = [
#         "<unk>",
#         "<s>",
#         "</s>",
#         "<|im_start|>",
#         "<|im_end|>"
#     ]
#
#     # 配置训练器
#     trainer = trainers.BpeTrainer(
#         vocab_size=vocab_size,
#         special_tokens=special_tokens,
#         min_frequency=2,  # 提高低频词过滤
#         show_progress=True,
#         initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
#     )
#
#     # 训练tokenizer
#     print(f"Training tokenizer with data from {data_path}")
#     texts = read_texts_from_jsonl(data_path)
#     tokenizer.train_from_iterator(texts, trainer=trainer, length=os.path.getsize(data_path))
#
#     # 验证特殊token映射
#     try:
#         assert tokenizer.token_to_id("<unk>") == 0
#         assert tokenizer.token_to_id("<s>") == 1
#         assert tokenizer.token_to_id("</s>") == 2
#         assert tokenizer.token_to_id("<|im_start|>") == 3
#         assert tokenizer.token_to_id("<|im_end|>") == 4
#     except AssertionError as e:
#         print("Special tokens mapping error:", e)
#         raise
#
#     # 保存tokenizer文件
#     tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
#
#     # 创建配置文件
#     create_tokenizer_config(save_dir)
#     print(f"Tokenizer saved to {save_dir}")


def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8192) -> None:
    """训练并保存自定义tokenizer (优化版)"""
    os.makedirs(save_dir, exist_ok=True)

    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 配置特殊token (使用变量避免显示问题)
    special_tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "<|im_start|>",
        "<|im_end|>"
    ]

    # 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        limit_alphabet=50000  # 限制字符集数量，大幅提速
    )

    # 准备文件列表
    files = []
    if os.path.isfile(data_path):
        files = [data_path]
    elif os.path.isdir(data_path):
        files = glob.glob(os.path.join(data_path, "*.jsonl"))

    if not files:
        raise FileNotFoundError(f"No files found in {data_path}")

    print(f"Found {len(files)} files. Starting training...")

    # 批量迭代器：一次读取一批数据，减少 Python 与底层的交互次数
    def batch_iterator(file_paths, batch_size=1000):
        # 1. 定义一个临时列表，用来装数据
        current_batch = []

        for file_path in file_paths:
            print(f"👉 正在读取文件: {file_path}")  # 确认开始读文件

            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    # 2. 解析 JSON (假设每行是一个 JSON)
                    try:
                        # 这里根据你的数据格式提取文本，假设文本在 "text" 字段
                        data = json.loads(line)
                        text = data.get("text", "")

                        if text:
                            current_batch.append(text)
                    except:
                        continue

                    # 3. 核心逻辑：如果凑够了一批，就 yield 出去，并打印日志
                    if len(current_batch) >= batch_size:
                        print(f"📦 已生成第 {i // batch_size} 批数据 (共 {len(current_batch)} 条)")  # <--- 关键日志
                        yield current_batch  # <--- 把数据吐给训练器
                        current_batch = []  # <--- 吐完后清空列表，准备装下一批

                    if i>=10000:
                        break

            # 4. 文件读完后，如果还有剩余没满一批的数据，也要吐出来
            if current_batch:
                print(f"📦 文件结束，吐出最后 {len(current_batch)} 条数据")
                yield current_batch

    # 开始训练
    tokenizer.train_from_iterator(batch_iterator(files), trainer=trainer)

    # 保存
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    # 如果有配置保存函数，请确保其存在
    # create_tokenizer_config(save_dir)
    print(f"Tokenizer saved to {save_dir}")


def eval_tokenizer(tokenizer_path: str) -> None:
    """评估tokenizer功能"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 测试基本属性
    print("\n=== Tokenizer基本信息 ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    # 测试聊天模板
    messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]

    print("\n=== 聊天模板测试 ===")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        # add_generation_prompt=True
    )
    print("Generated prompt:\n", prompt, sep="")

    # 测试编码解码
    print("\n=== 编码解码测试 ===")
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    print("Decoded text matches original:", decoded == prompt)

    # 测试特殊token处理
    print("\n=== 特殊token处理 ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    print("Special tokens preserved:", decoded == test_text)


def main():
    # 配置路径
    data_path = "test_data"
    save_dir = "tokenizer_k"

    # 训练tokenizer
    train_tokenizer(
        data_path=data_path,
        save_dir=save_dir,
        vocab_size=6144
    )

    # 评估tokenizer
    eval_tokenizer(save_dir)


if __name__ == '__main__':
    main()
