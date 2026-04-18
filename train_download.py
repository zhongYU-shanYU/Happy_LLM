import json
import os
from tqdm import tqdm

# 寄了，python还是用不弄明白，手动下载数据了
# ==========================================
# 1. 处理预训练数据 (Seq-Monkey) - 修改版
# ==========================================
def process_pretrain_data(input_file, output_file, chunk_size=512):
    print(f"开始处理预训练数据: {input_file}")

    # 使用 'w' 模式清空旧文件，如果不想清空用 'a'
    with open(output_file, 'w', encoding='utf-8') as pretrain:
        with open(input_file, 'r', encoding='utf-8') as f:
            # 不要 readlines()，直接迭代文件对象
            # 这样可以一行一行读，不占内存
            for line in tqdm(f, desc="Processing Pretrain"):
                try:
                    line_data = json.loads(line)
                    text = line_data.get('text', '')

                    # 切分文本
                    for i in range(0, len(text), chunk_size):
                        chunk = text[i:i + chunk_size]
                        if len(chunk) > 10:  # 过滤掉太短的
                            pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')
                except Exception as e:
                    continue  # 跳过坏数据


# ==========================================
# 2. 处理SFT数据 (BelleGroup) - 修改版
# ==========================================
def process_sft_data(input_file, output_file):
    print(f"开始处理SFT数据: {input_file}")

    def convert_message(data):
        message = [{"role": "system", "content": "你是一个AI助手"}]
        for item in data:
            if item['from'] == 'human':
                message.append({'role': 'user', 'content': item['value']})
            elif item['from'] == 'assistant':
                message.append({'role': 'assistant', 'content': item['value']})
        return message

    with open(output_file, 'w', encoding='utf-8') as sft:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing SFT"):
                try:
                    item = json.loads(line)
                    # 确保有 conversations 字段
                    if 'conversations' in item:
                        message = convert_message(item['conversations'])
                        sft.write(json.dumps(message, ensure_ascii=False) + '\n')
                except Exception as e:
                    continue


# ==========================================
# 3. 运行入口
# ==========================================
if __name__ == "__main__":
    # 注意：你需要先确保下载并解压了文件，路径要对！

    # 1. 处理预训练数据 (如果你下载了的话)
    process_pretrain_data('mobvoi_seq_monkey_general_open_corpus.jsonl', 'seq_monkey_processed.jsonl')

    # 2. 处理 SFT 数据 (BelleGroup)
    # 假设你下载解压后的文件路径是这个，请根据实际情况修改
    if os.path.exists('BelleGroup/train_3.5M_CN.json'):
        process_sft_data('BelleGroup/train_3.5M_CN.json', 'BelleGroup_processed.jsonl')
    else:
        print("⚠️ 没找到 BelleGroup 数据文件，请先下载！")