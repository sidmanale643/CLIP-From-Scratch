import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from datasets import load_dataset

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        caption = item['text']
        image = item['image']

        image = self.transform(image)

        inputs = self.tokenizer(caption, padding='max_length', truncation=True, max_length=64, return_tensors="pt")

        return {
            "image": image,
            "input_ids": inputs['input_ids'].squeeze(),
            "attention_mask": inputs['attention_mask'].squeeze()
        }

def train(model, dataloader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            images = batch['image'].to(device)
            tokens = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            _, loss = model(images, tokens, attention_mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("lambdalabs/naruto-blip-captions")
train_dataset = CustomDataset(ds['train'])

dataloader = DataLoader(train_dataset, batch_size = 64 , shuffle=True)

model = CLIP().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

train(model, dataloader, optimizer, device)
