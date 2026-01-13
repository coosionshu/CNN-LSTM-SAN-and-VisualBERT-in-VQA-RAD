import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import VisualBertModel


# --- Attention Layer (SAN的核心组件) ---
class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = nn.Tanh()
        # 1. 将图像特征映射到公共空间
        self.xw = nn.Linear(v_dim, num_hid)
        # 2. 将问题特征映射到公共空间
        self.yw = nn.Linear(q_dim, num_hid)
        # 3. 计算注意力分数的权重
        self.hwa = nn.Linear(num_hid, 1)

    def forward(self, v, q):
        """
        v: [Batch, Num_Regions, v_dim]
        q: [Batch, q_dim]
        """
        # q: [B, 1024] -> [B, 1, 1024] -> [B, 49, 1024]
        q_expanded = q.unsqueeze(1).expand(v.size(0), v.size(1), -1)

        # h_a = Tanh( W_v*V + W_q*Q )
        h_a = self.nonlinear(self.xw(v) + self.yw(q_expanded))

        # 得到每个区域的分数
        probs = self.hwa(h_a)  # [B, 49, 1]
        probs = F.softmax(probs, dim=1)

        # 加权求和
        v_weighted = (probs * v).sum(dim=1)  # [B, v_dim]

        return v_weighted, probs


# --- Baseline Model: CNN-LSTM + SAN (已修复维度匹配) ---
class CNN_LSTM_VQA(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=1024, num_answers=458):
        super(CNN_LSTM_VQA, self).__init__()

        # 1. Image Encoder: ResNet152
        resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])

        # === 修复点 1: 增加降维层 ===
        # 把 ResNet 的 2048 维降到 LSTM 的 1024 维
        self.img_proj = nn.Linear(2048, hidden_dim)

        # 2. Question Encoder: LSTM
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.5)

        # 3. SAN (注意: v_dim 现在是 hidden_dim=1024 了)
        self.san1 = Attention(v_dim=hidden_dim, q_dim=hidden_dim, num_hid=512)
        self.san2 = Attention(v_dim=hidden_dim, q_dim=hidden_dim, num_hid=512)

        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_answers)
        )

    def forward(self, images, questions):
        # --- Image ---
        img_feats = self.resnet_features(images)  # [B, 2048, 7, 7]
        v = img_feats.view(img_feats.size(0), 2048, -1).permute(0, 2, 1)  # [B, 49, 2048]

        # === 修复点 2: 执行降维 ===
        v = self.img_proj(v)  # [B, 49, 1024]

        # --- Question ---
        emb = self.embedding(questions)
        _, (h_n, _) = self.lstm(emb)
        q = h_n[-1]  # [B, 1024]

        # --- SAN Stack 1 ---
        # 现在 v 和 q 都是 1024 维，可以相加了
        v1, _ = self.san1(v, q)
        u1 = q + v1  # [B, 1024]

        # --- SAN Stack 2 ---
        v2, _ = self.san2(v, u1)
        u2 = u1 + v2  # [B, 1024]

        # --- Classify ---
        logits = self.classifier(u2)
        return logits


# --- Advanced Model: VisualBERT ---
class VisualBERT_VQA(nn.Module):
    def __init__(self, num_answers=458):
        super(VisualBERT_VQA, self).__init__()
        self.bert = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')

        # 【新增】定义 Dropout 层 (防止过拟合)
        self.dropout = nn.Dropout(0.5)

        self.cls_head = nn.Linear(768, num_answers)

        # VisualBERT expects visual embeddings, not raw images.
        # We add a small feature extractor here to make it runnable with raw images.
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.vis_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.vis_proj = nn.Linear(2048, 2048)  # VisualBERT expects 2048 dim visual inputs

        # 【新增】冻结 ResNet 的参数 (Freeze Backbone)
        # 这会阻止训练过程中更新图像提取器的权重
        for param in self.vis_encoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, images):
        # 1. Extract Visual Embeddings
        vis_feats = self.vis_encoder(images)  # [B, 2048, 7, 7]
        vis_feats = vis_feats.view(vis_feats.size(0), 2048, -1).permute(0, 2, 1)  # [B, 49, 2048]
        vis_embeds = self.vis_proj(vis_feats)

        # 2. VisualBERT Forward
        # token_type_ids is managed automatically if not provided, but we pass visual_embeds
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=vis_embeds,
            visual_token_type_ids=torch.ones(vis_embeds.shape[:-1], dtype=torch.long, device=images.device)
        )

        pooled_output = outputs.pooler_output

        # 【新增】应用 Dropout
        pooled_output = self.dropout(pooled_output)

        logits = self.cls_head(pooled_output)
        return logits