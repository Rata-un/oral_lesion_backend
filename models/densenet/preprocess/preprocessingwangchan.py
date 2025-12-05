# -*- coding: utf-8 -*-
import os
import json
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
# Import โมดูลที่จำเป็น
from transformers import CamembertTokenizer

# ===== CONFIG =====
TEXT_MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"

# ใส่ค่า mean/std ของ "train set" คุณ (ที่เพิ่งคำนวณได้)
DATASET_MEAN = [0.6089681386947632, 0.411862850189209, 0.4016309380531311]
DATASET_STD = [0.19572971761226654, 0.17690013349056244, 0.170320525765419]

# คีย์ใน CSV
COL_IMG = "img_path"
COL_LBL = "disease"
COL_TXT = "caption"

# ข้อความกลางเมื่อไม่มีอาการ
NEUTRAL_PROMPT = "ไม่มีอาการ รายละเอียดไม่ระบุ ตำแหน่งไม่ระบุ"
MAX_LEN = 128
# ==================

def get_tokenizer():
    # โหลด tokenizer โดยใช้ CamembertTokenizerFast
    # ซึ่งใช้ SentencePiece และเข้ากันได้กับโมเดลนี้

    return CamembertTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased", use_fast=False)


def get_transforms(img_size=(224,224)):
    return T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])

class CustomDataset(Dataset):
    """
    อ่านจาก CSV: ต้องมี img_path, label, caption (caption ว่างได้)
    label_map: dict เช่น {"Leukoplakia":0, "Lichen":1, ...}
    """
    def __init__(self, csv_path, tokenizer, transform, label_map=None):
        self.df = pd.read_csv(csv_path)
        assert COL_IMG in self.df.columns and COL_LBL in self.df.columns, \
            f"CSV ต้องมีคอลัมน์ {COL_IMG}, {COL_LBL}"
        if COL_TXT not in self.df.columns:
            self.df[COL_TXT] = ""

        self.tok = tokenizer
        self.tf = transform
        self.label_map = label_map or self._build_label_map()
        self._check_paths()

    def _build_label_map(self):
        classes = sorted(self.df[COL_LBL].astype(str).unique().tolist())
        return {c:i for i,c in enumerate(classes)}

    def _check_paths(self):
        # แจ้งเตือนเฉย ๆ ถ้ารูปหาย
        missing = (~self.df[COL_IMG].astype(str).apply(os.path.exists)).sum()
        if missing > 0:
            print(f"⚠️ พบรูปหาย {missing} ไฟล์ ใน {len(self.df)} แถว (จะ error ตอน __getitem__ ถ้าถึงแถวนั้น)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        img_path = str(row[COL_IMG])
        image = Image.open(img_path).convert("RGB")
        image = self.tf(image)

        # Label
        label_name = str(row[COL_LBL])
        label = torch.tensor(self.label_map[label_name], dtype=torch.long)

        # Text
        text_raw = str(row.get(COL_TXT, "") or "").strip()
        text = text_raw if text_raw else NEUTRAL_PROMPT

        enc = self.tok(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)      # (L,)
        attention_mask = enc["attention_mask"].squeeze(0)

        return {
            "image": image,              # (3,H,W)
            "input_ids": input_ids,      # (L,)
            "attention_mask": attention_mask,
            "label": label,
            "text_is_neutral": torch.tensor(int(text_raw == ""), dtype=torch.long)
        }