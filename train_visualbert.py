import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset  # <--- 新增引用

# 引用模块
from model import VisualBERT_VQA
from data_loader import VQARADDataset

# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
IMG_DIR = os.path.join(DATA_DIR, "VQA_RAD Image Folder")

# --- 训练超参数 ---
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 2e-5
RESNET_LR = 1e-5
PATIENCE = 12


# --- 1. 绘图函数 (已完善 F1, Acc, Loss) ---
def plot_curves(history, model_name="VisualBERT_Final"):
    epochs = range(1, len(history['train_loss']) + 1)

    # 1. Accuracy Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_acc_all'], 'b-', label='Overall', linewidth=2)
    plt.plot(epochs, history['val_acc_closed'], 'g--', label='Closed')
    plt.plot(epochs, history['val_acc_open'], 'r:', label='Open')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_acc.png')
    plt.close()

    # 2. F1 Score Curve (新增)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_f1_all'], 'b-', label='Overall F1', linewidth=2)
    plt.plot(epochs, history['val_f1_closed'], 'g--', label='Closed F1')
    plt.plot(epochs, history['val_f1_open'], 'r:', label='Open F1')
    plt.title('F1 Score Curves')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_f1.png')
    plt.close()

    # 3. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'r-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'b--', label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_loss.png')
    plt.close()

    print(f"📊 Curves saved: {model_name}_acc.png, {model_name}_f1.png, {model_name}_loss.png")


# --- 2. 增强策略 ---
def get_transforms(split):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomRotation(30),  # 保持你之前的强增强
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])


# --- 3. 指标计算 (已完善 F1 计算) ---
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


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Full Training on {device}...")

    # --- A. 智能数据准备 ---
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("⏳ Loading dataset from Hugging Face for vocabulary building...")
    dataset = load_dataset("flaviagiammarino/vqa-rad")

    # 合并 train 和 test 数据来统计全局答案频率
    full_data = list(dataset['train']) + list(dataset['test'])

    # 1. 统计频率
    all_raw_answers = [str(item['answer']).lower().strip() for item in full_data]
    counts = Counter(all_raw_answers)

    # 2. 过滤：只保留出现次数 >= 2 的答案
    min_freq = 2
    filtered_answers = [ans for ans, freq in counts.items() if freq >= min_freq]

    # 3. 强制包含核心词 (Yes/No)
    for w in ['yes', 'no']:
        if w not in filtered_answers: filtered_answers.append(w)

    filtered_answers = sorted(list(set(filtered_answers)))

    # 4. 插入 <unk> (未知/低频词) 到索引 0
    filtered_answers.insert(0, "<unk>")

    answer_map = {ans: idx for idx, ans in enumerate(filtered_answers)}

    print(f"📉 Optimization: Reduced answers from {len(set(all_raw_answers))} to {len(answer_map)} classes.")
    print(f"   (Removed {len(set(all_raw_answers)) - len(answer_map)} rare answers)")

    # 保存映射
    with open('answer_classes.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_answers, f)

    # --- B. 加载数据 ---
    train_dataset = VQARADDataset(DATA_DIR, IMG_DIR, transform=get_transforms('train'), tokenizer=tokenizer,
                                  mode='train', answer_to_idx=answer_map)
    val_dataset = VQARADDataset(DATA_DIR, IMG_DIR, transform=get_transforms('test'), tokenizer=tokenizer, mode='test',
                                answer_to_idx=answer_map)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- C. 模型设置 ---
    model = VisualBERT_VQA(num_answers=len(answer_map)).to(device)

    print("🔓 Unfreezing last block of ResNet for fine-tuning...")
    for param in model.vis_encoder[7].parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {'params': model.bert.parameters(), 'lr': LEARNING_RATE},
        {'params': model.cls_head.parameters(), 'lr': LEARNING_RATE},
        {'params': model.vis_encoder[7].parameters(), 'lr': RESNET_LR},
    ], weight_decay=0.01)

    criterion = nn.CrossEntropyLoss()

    # 初始化 History (包含 F1)
    history = {
        'train_loss': [], 'val_loss': [],
        'val_acc_all': [], 'val_acc_closed': [], 'val_acc_open': [],
        'val_f1_all': [], 'val_f1_closed': [], 'val_f1_open': []
    }

    best_acc = 0.0
    patience_counter = 0

    # --- D. 训练 ---
    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0
        for batch in train_loader:
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            images, labels = batch['image'].to(device), batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # Val
        model.eval()
        val_loss_sum = 0
        all_preds, all_labels, all_types = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                images, labels, type_ids = batch['image'].to(device), batch['label'].to(device), batch['type_id']

                outputs = model(input_ids, attention_mask, images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_types.extend(type_ids.numpy())

        avg_val_loss = val_loss_sum / len(val_loader)
        metrics = calculate_metrics(all_labels, all_preds, all_types)

        # 记录所有指标
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc_all'].append(metrics['acc_all'])
        history['val_acc_closed'].append(metrics['acc_closed'])
        history['val_acc_open'].append(metrics['acc_open'])
        history['val_f1_all'].append(metrics['f1_all'])
        history['val_f1_closed'].append(metrics['f1_closed'])
        history['val_f1_open'].append(metrics['f1_open'])

        # 打印信息 (包含 F1)
        print(f"Ep {epoch + 1}/{EPOCHS} | Loss: T={avg_train_loss:.3f}/V={avg_val_loss:.3f} | "
              f"Acc: {metrics['acc_all']:.3f} (C:{metrics['acc_closed']:.3f}/O:{metrics['acc_open']:.3f}) | "
              f"F1: {metrics['f1_all']:.3f} (C:{metrics['f1_closed']:.3f}/O:{metrics['f1_open']:.3f})")

        if metrics['acc_all'] > best_acc:
            best_acc = metrics['acc_all']
            patience_counter = 0
            torch.save(model.state_dict(), 'best_visualbert_model.pth')
            print(f"  🌟 Best Model Saved! ({best_acc:.3f})")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("⏹ Early Stopping.")
                break

    plot_curves(history)


if __name__ == "__main__":
    train()