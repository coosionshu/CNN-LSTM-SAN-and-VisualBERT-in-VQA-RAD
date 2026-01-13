import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from datasets import load_dataset

# 引用模块
from model import CNN_LSTM_VQA
# 去掉 SimpleVocab 的引用，我们在本文件定义它，以防 data_loader 中丢失
from data_loader import VQARADDataset

# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
IMG_DIR = os.path.join(DATA_DIR, "VQA_RAD Image Folder")

# --- 参数优化 ---
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 12  # <--- [新增] 早停耐心值


# --- 辅助类：简单的词表构建 (替代原 data_loader 中的 SimpleVocab) ---
class LocalSimpleVocab:
    def __init__(self, data):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.build_vocab(data)

    def build_vocab(self, data):
        # 统计所有问题中的单词
        for item in data:
            q = item['question']  # HF dataset 中直接是 'question'
            tokens = self.tokenize(q)
            for token in tokens:
                if token not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[token] = idx
                    self.idx2word[idx] = token

    def tokenize(self, text):
        return str(text).lower().replace('?', '').split()

    def encode(self, question, max_len=20):
        tokens = self.tokenize(question)
        indices = [self.word2idx.get(t, self.word2idx["<unk>"]) for t in tokens]
        # Padding
        if len(indices) < max_len:
            indices += [self.word2idx["<pad>"]] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return torch.tensor(indices, dtype=torch.long)


# --- 绘图函数 ---
def plot_curves(history, model_name="Baseline"):
    epochs = range(1, len(history['loss']) + 1)

    # 1. Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_acc_all'], 'b-', label='Overall', linewidth=2)
    plt.plot(epochs, history['val_acc_closed'], 'g--', label='Closed')
    plt.plot(epochs, history['val_acc_open'], 'r:', label='Open')
    plt.title(f'{model_name} Accuracy Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_acc.png')
    plt.close()

    # 2. F1 Score
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_f1_all'], 'b-', label='Overall F1', linewidth=2)
    plt.plot(epochs, history['val_f1_closed'], 'g--', label='Closed F1')
    plt.plot(epochs, history['val_f1_open'], 'r:', label='Open F1')
    plt.title(f'{model_name} F1 Score Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_f1.png')
    plt.close()

    # 3. Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['loss'], 'r-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'b--', label='Val Loss')
    plt.title(f'{model_name} Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_loss.png')
    plt.close()

    print(f"✅ Curves saved: {model_name}_acc.png, _f1.png, _loss.png")


# --- 增强策略 ---
def get_transforms(split):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])


# --- 指标计算 ---
def calculate_metrics(y_true, y_pred, types):
    y_true, y_pred, types = np.array(y_true), np.array(y_pred), np.array(types)

    # Overall
    acc_all = accuracy_score(y_true, y_pred)
    f1_all = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Closed
    mask_closed = (types == 0)
    if mask_closed.sum() > 0:
        acc_closed = accuracy_score(y_true[mask_closed], y_pred[mask_closed])
        f1_closed = f1_score(y_true[mask_closed], y_pred[mask_closed], average='weighted', zero_division=0)
    else:
        acc_closed, f1_closed = 0.0, 0.0

    # Open
    mask_open = (types == 1)
    if mask_open.sum() > 0:
        acc_open = accuracy_score(y_true[mask_open], y_pred[mask_open])
        f1_open = f1_score(y_true[mask_open], y_pred[mask_open], average='weighted', zero_division=0)
    else:
        acc_open, f1_open = 0.0, 0.0

    return {
        'acc_all': acc_all, 'acc_closed': acc_closed, 'acc_open': acc_open,
        'f1_all': f1_all, 'f1_closed': f1_closed, 'f1_open': f1_open
    }


# --- 训练循环 ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Baseline (CNN-LSTM) Training on {device}...")

    # 1. 准备数据 (从 HF 加载)
    print("⏳ Loading dataset from Hugging Face...")
    dataset = load_dataset("flaviagiammarino/vqa-rad")
    full_data = list(dataset['train']) + list(dataset['test'])

    # 构建问题词表 (LSTM 需要)
    vocab_builder = LocalSimpleVocab(full_data)

    # --- 应用答案过滤 ---
    all_raw_answers = [str(item['answer']).lower().strip() for item in full_data]
    counts = Counter(all_raw_answers)

    # 只保留出现次数 >= 2 的答案
    min_freq = 2
    filtered_answers = [ans for ans, freq in counts.items() if freq >= min_freq]

    # 强制包含 Yes/No
    for w in ['yes', 'no']:
        if w not in filtered_answers: filtered_answers.append(w)

    filtered_answers = sorted(list(set(filtered_answers)))

    # 插入 <unk> 到索引 0
    filtered_answers.insert(0, "<unk>")

    answer_map = {ans: idx for idx, ans in enumerate(filtered_answers)}

    print(f"📉 Optimization: Reduced answers from {len(set(all_raw_answers))} to {len(answer_map)} classes.")
    print(f"   (Removed {len(set(all_raw_answers)) - len(answer_map)} rare answers)")

    with open('answer_classes.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_answers, f)

    train_dataset = VQARADDataset(DATA_DIR, IMG_DIR, transform=get_transforms('train'), mode='train',
                                  answer_to_idx=answer_map)
    val_dataset = VQARADDataset(DATA_DIR, IMG_DIR, transform=get_transforms('test'), mode='test',
                                answer_to_idx=answer_map)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. 模型
    model = CNN_LSTM_VQA(vocab_size=len(vocab_builder.word2idx), num_answers=len(answer_map)).to(device)

    print("🔒 Freezing ResNet backbone...")
    for param in model.resnet_features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 初始化 history
    history = {
        'loss': [], 'val_loss': [],
        'val_acc_all': [], 'val_acc_closed': [], 'val_acc_open': [],
        'val_f1_all': [], 'val_f1_closed': [], 'val_f1_open': []
    }

    best_acc = 0.0
    patience_counter = 0 # <--- [新增] 早停计数器

    # 3. 循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, questions_text, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            # 编码问题
            input_ids = torch.stack([vocab_builder.encode(q) for q in questions_text]).to(device)

            optimizer.zero_grad()
            outputs = model(images, input_ids)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Val
        model.eval()
        val_total_loss = 0
        all_preds, all_labels, all_types = [], [], []
        with torch.no_grad():
            for images, questions_text, labels, types in val_loader:
                images, labels = images.to(device), labels.to(device)
                input_ids = torch.stack([vocab_builder.encode(q) for q in questions_text]).to(device)

                outputs = model(images, input_ids)
                loss = criterion(outputs, labels)
                val_total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_types.extend(types.numpy())

        val_loss = val_total_loss / len(val_loader)
        metrics = calculate_metrics(all_labels, all_preds, all_types)

        scheduler.step(val_loss)

        # 记录所有指标
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        history['val_acc_all'].append(metrics['acc_all'])
        history['val_acc_closed'].append(metrics['acc_closed'])
        history['val_acc_open'].append(metrics['acc_open'])

        history['val_f1_all'].append(metrics['f1_all'])
        history['val_f1_closed'].append(metrics['f1_closed'])
        history['val_f1_open'].append(metrics['f1_open'])

        print(f"Ep {epoch + 1} | Loss: {train_loss:.3f}/{val_loss:.3f} | "
              f"Acc: {metrics['acc_all']:.3f} (C:{metrics['acc_closed']:.3f}/O:{metrics['acc_open']:.3f}) | "
              f"F1: {metrics['f1_all']:.3f}")

        # --- [新增] 早停逻辑 ---
        if metrics['acc_all'] > best_acc:
            best_acc = metrics['acc_all']
            patience_counter = 0 # 重置计数器
            torch.save(model.state_dict(), 'best_baseline_model.pth')
            print(f"  🌟 New Best Model Saved! ({best_acc:.3f})")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n⏹ Early Stopping Triggered at Epoch {epoch + 1}!")
                break

    plot_curves(history, "Baseline")


if __name__ == "__main__":
    train()