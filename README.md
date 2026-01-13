# Medical VQA on VQA-RAD: CNN-LSTM+SAN vs. VisualBERT

**基于 VQA-RAD 数据集的医学视觉问答：CNN-LSTM+SAN 与 VisualBERT 对比研究**

## 📌 Overview / 项目概述

This repository implements and compares two distinct architectures for the **VQA-RAD** (Medical Visual Question Answering) dataset. The project aims to bridge the gap between clinical imaging and natural language understanding using both traditional attention-based models and modern Transformer-based pre-trained models.

本项目针对 **VQA-RAD**（医学影像视觉问答）数据集实现并对比了两种不同的架构。项目旨在利用传统注意力机制模型和现代 Transformer 预训练模型，缩小临床影像与自然语言理解之间的鸿沟。

------

## 🏗️ Model Architectures / 模型架构

### 1. CNN-LSTM + SAN (Stacked Attention Networks)

A classical dual-stream approach for VQA tasks:

- **Image Encoder**: Pre-trained ResNet152 to extract spatial visual features.
- **Text Encoder**: LSTM to process the sequence of the medical question.
- **Attention**: **Stacked Attention Networks (SAN)** perform multi-layer query-image reasoning to locate lesion areas related to the question.

一种经典的 VQA 双流处理方法：

- **图像编码器**：使用预训练的 ResNet152 提取空间视觉特征。
- **文本编码器**：使用 LSTM 处理医学问题的序列信息。
- **注意力机制**：**堆叠注意力网络 (SAN)** 通过多层“问题-图像”推理，定位与问题相关的病灶区域。

### 2. VisualBERT

A single-stream Transformer-based model:

- **Fusion Strategy**: Concatenates visual tokens (extracted via Faster R-CNN or Grid Features) with text tokens.
- **Self-Attention**: Automatically learns the implicit alignment between medical terms and radiological image regions through the Transformer layers.

一种基于 Transformer 的单流模型：

- **融合策略**：将视觉 Token（通过 Faster R-CNN 或网格特征提取）与文本 Token 直接拼接。
- **自注意力机制**：通过 Transformer 层自动学习医学术语与放射影像区域之间的隐式对齐关系。

------

## 📊 Dataset: VQA-RAD / 数据集介绍

[flaviagiammarino/vqa-rad · Datasets at Hugging Face](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)

VQA-RAD is a high-quality, manually labeled dataset by clinicians:

- **Modality**: CT, MRI, X-ray.
- **Anatomy**: Head, Chest, Abdomen.
- **Question Types**: Categorized into Closed-ended (Yes/No) and Open-ended (What, Where, How, etc.).

VQA-RAD 是由临床医生手动标注的高质量数据集：

- **模态**：包含 CT、MRI、X-ray。
- **部位**：头部、胸部、腹部。
- **问题类型**：分为封闭式（是/否）和开放式（什么、在哪里、如何等）。

------

## 🚀 Quick Start / 快速开始

### 1. Requirements / 环境要求

Bash

**Conda**

```
conda create -n vqa_env python=3.9
conda activate vqa_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

Python

```
pip install -r requirements.txt
```

### 2. Training / 训练示例

**Train CNN-LSTM+SAN:**

Bash

```
python train_baseline.py
```

**Train VisualBERT:**

Bash

```
python train_visualbert.py
```

------

## 📈 Experimental Results / 实验结果

| Comparative VisualBERT  vs. Baseline |                                      |                               |                 |
| :----------------------------------: | ------------------------------------ | ----------------------------- | --------------- |
|              **Metric**              | **CNN-LSTM+SAN (Baseline)**          | **VisualBERT (Advanced)**     | **Improvement** |
|         **Overall Accuracy**         | 41.20%                               | 49.20%                        | 8.00%           |
|         **Closed Accuracy**          | 54.20%                               | 64.10%                        | 9.90%           |
|          **Open Accuracy**           | 25.00%                               | 30.50%                        | 5.50%           |
|        **Model Architecture**        | ResNet152 + LSTM +SAN (From Scratch) | ResNet50 + BERT (Pre-trained) |                 |

------

## 📂 Project Structure / 目录结构

- `./data_loader.py`: Source code for data_loader. (数据加载源代码)
- `./model.py`: Source code for model. (模型源代码)
- `./train_baseline.py`: Source code for train CNN-LSTM +SAN. (模型训练源代码)
- `./train_visualbert.py`: Source code for train VisualBERT. (模型训练源代码)
- `./requirements.txt`: env_txt. (所需环境)
- `weights/`: Saved model checkpoints. (模型权重保存路径)
- `eval/`: Evaluation scripts for VQA accuracy and BLEU scores. (评估指标计算脚本)

------
