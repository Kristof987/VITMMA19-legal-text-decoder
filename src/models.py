import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel

from config import (
    MODEL_NAME,
    NUM_CLASSES,
    DROPOUT_RATE,
    CLASSIFIER_HIDDEN_SIZE,
    HIDDEN_SIZE,
    MAX_LENGTH
)


class LegalTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label - 1, dtype=torch.long)
        }


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text
        }


class HuBERTClassifier(nn.Module):   
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, 
                 dropout_rate=DROPOUT_RATE, hidden_size=CLASSIFIER_HIDDEN_SIZE):
        super(HuBERTClassifier, self).__init__()
        
        self.hubert = AutoModel.from_pretrained(model_name)

        for param in self.hubert.embeddings.parameters():
            param.requires_grad = False
        
        for layer in self.hubert.encoder.layer[:-4]:
            for param in layer.parameters():
                param.requires_grad = False
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(HIDDEN_SIZE, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.hubert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
        
        return logits
