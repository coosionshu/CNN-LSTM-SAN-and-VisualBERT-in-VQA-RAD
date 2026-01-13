import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class VQARADDataset(Dataset):
    def __init__(self, data_dir, image_dir, transform=None, tokenizer=None, mode='train', answer_to_idx=None):
        """
        Args:
            data_dir: (已弃用) 保留是为了兼容 train.py 的调用
            image_dir: (已弃用) 保留是为了兼容 train.py 的调用
            transform: 图片预处理
            tokenizer: BERT tokenizer
            mode: 'train' 或 'test'
            answer_to_idx: 答案到索引的映射字典
        """
        self.transform = transform
        self.tokenizer = tokenizer
        self.mode = mode

        # 🚀 修改点 1: 从 Hugging Face 加载数据集
        # 即使本地没有数据，这行代码也会自动下载并缓存
        dataset = load_dataset("flaviagiammarino/vqa-rad")

        # 🚀 修改点 2: 使用官方的 train/test 划分
        # HF 数据集本身就有 'train' (1793条) 和 'test' (451条)
        if mode == 'train':
            self.data = dataset['train']
        else:
            self.data = dataset['test']

        # 构建或使用传入的答案词表
        if answer_to_idx is None:
            self.answer_to_idx = self._build_answer_vocab()
        else:
            self.answer_to_idx = answer_to_idx

    def _build_answer_vocab(self):
        # 从当前数据集中提取所有答案构建词表
        answers = [str(item['answer']).lower().strip() for item in self.data]
        vocab = {ans: idx for idx, ans in enumerate(set(answers))}
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # --- 1. Image 处理 ---
        # 🚀 修改点 3: HF 数据集直接返回 PIL Image 对象，无需路径读取
        image = item['image'].convert('RGB')

        if self.transform:
            image = self.transform(image)

        # --- 2. Question & Answer ---
        question = item['question']
        # 统一转小写并去空格
        answer = str(item['answer']).lower().strip()

        # 获取标签，如果不在词表中则归为 <unk> (0)
        label = self.answer_to_idx.get(answer, 0)

        # --- 3. Answer Type 推断 ---
        # 🚀 修改点 4: HF 数据集没有 answer_type 字段，我们需要手动推断
        # 逻辑：如果是 yes/no 问题，则为 Closed (0)，否则为 Open (1)
        if answer in ['yes', 'no']:
            type_id = 0  # Closed
        else:
            type_id = 1  # Open

        # --- Return ---
        if self.tokenizer:
            encoded_q = self.tokenizer(
                question,
                padding='max_length',
                truncation=True,
                max_length=32,
                return_tensors='pt'
            )
            return {
                'image': image,
                'input_ids': encoded_q['input_ids'].squeeze(),
                'attention_mask': encoded_q['attention_mask'].squeeze(),
                'label': torch.tensor(label, dtype=torch.long),
                'type_id': torch.tensor(type_id, dtype=torch.long)
            }
        else:
            return image, question, torch.tensor(label, dtype=torch.long), torch.tensor(type_id, dtype=torch.long)