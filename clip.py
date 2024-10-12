import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel

class TextEncoder(nn.Module):
    def __init__(self, proj_dim):
        super().__init__()
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.proj = nn.Linear(768, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim, eps=1e-6)

    def forward(self, tokens, attention_mask):
        text_emb = self.model(input_ids=tokens, attention_mask=attention_mask).last_hidden_state
        text_emb = text_emb[:, 0, :]
        text_proj = self.layer_norm(self.proj(text_emb))
        return text_proj

class ImageEncoder(nn.Module):
    def __init__(self, proj_dim):
        super().__init__()
        vit_config = config()
        self.model = ViT(vit_config)
        self.proj = nn.Linear(vit_config.d_model, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, img):
        image_emb = self.model(img)
        image_emb = image_emb[:, 0, :]
        img_proj = self.layer_norm(self.proj(image_emb))
        return img_proj

class CLIP(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.image_encoder = ImageEncoder(256)
        self.text_encoder = TextEncoder(256)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, img, tokens, attention_mask):
        img_logits = self.image_encoder(img)
        text_logits = self.text_encoder(tokens, attention_mask)

        img_logits = F.normalize(img_logits, dim=-1)
        text_logits = F.normalize(text_logits, dim=-1)

        logits = torch.matmul(img_logits, text_logits.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(logits.device)

        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return logits, loss
