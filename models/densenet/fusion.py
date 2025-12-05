import torchvision.transforms as T
import torch.nn as nn
from transformers import CamembertTokenizer, AutoModel
from torchvision import models

TEXT_MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
# mean/std ของ train set
DATASET_MEAN = [0.6089681386947632, 0.411862850189209, 0.4016309380531311]
DATASET_STD = [0.19572971761226654, 0.17690013349056244, 0.170320525765419]

# Preprocessing WangchanBERTa
def get_tokenizer():
    return CamembertTokenizer.from_pretrained(TEXT_MODEL_NAME, use_fast=False)

def get_transforms(img_size=(224,224)):
    return T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])

class DenseNet121Classifier(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        base = models.densenet121(weights="IMAGENET1K_V1")
        in_feat = base.classifier.in_features
        base.classifier = nn.Identity()
        self.encoder = base
        self.classifier = nn.Sequential(
            nn.Linear(in_feat, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat)

class TextClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        cls = out.last_hidden_state[:,0,:]
        return self.classifier(cls)
