# -*- coding: utf-8 -*-
import os, argparse, json, random, csv,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from .preprocess.preprocessingwangchan import CustomDataset, get_tokenizer, get_transforms

TEXT_MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); y_true, y_pred, loss_sum, n_sum = [], [], 0.0, 0
    for b in loader:
        ids, mask, y = b["input_ids"].to(device), b["attention_mask"].to(device), b["label"].to(device)
        logits = model(ids, mask)
        loss = F.cross_entropy(logits, y)
        loss_sum += loss.item()*y.size(0)
        pred = logits.argmax(dim=1)
        y_true.extend(y.cpu().tolist()); y_pred.extend(pred.cpu().tolist())
        n_sum+=y.size(0)
    return loss_sum/n_sum, accuracy_score(y_true,y_pred)

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = get_tokenizer(); tfm = get_transforms((args.img_size,args.img_size))
    ds_tmp = CustomDataset(args.train_csv, tok, tfm, label_map=None)
    label_map = ds_tmp.label_map
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir,"label_map_txt.json"),"w",encoding="utf-8") as f:
        json.dump(label_map,f,ensure_ascii=False,indent=2)

    ds_tr = CustomDataset(args.train_csv, tok, tfm, label_map=label_map)
    ds_va = CustomDataset(args.val_csv, tok, tfm, label_map=label_map)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size)

    model = TextClassifier(num_classes=len(label_map)).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(
        optim, int(0.1*len(dl_tr)*args.epochs), len(dl_tr)*args.epochs
    )

    best_acc, patience= -1,0
    best_path=os.path.join(args.out_dir,"best_text.pth")
    last_path=os.path.join(args.out_dir,"last_text.pth")

    # ===== prepare CSV logger =====
    csv_path = os.path.join(args.out_dir, "metrics_log_text.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","train_acc","val_loss","val_acc"])

    for ep in range(1,args.epochs+1):
        model.train(); y_true,y_pred,loss_sum,n_sum=[],[],0.0,0
        for b in dl_tr:
            ids, mask, y = b["input_ids"].to(device), b["attention_mask"].to(device), b["label"].to(device)
            optim.zero_grad()
            logits=model(ids,mask)
            loss=F.cross_entropy(logits,y)
            loss.backward(); optim.step(); sched.step()
            pred=logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist()); y_pred.extend(pred.cpu().tolist())
            loss_sum+=loss.item()*y.size(0); n_sum+=y.size(0)
        tr_loss=loss_sum/n_sum; tr_acc=accuracy_score(y_true,y_pred)
        va_loss,va_acc=evaluate(model,dl_va,device)

        # ==== Print & Log ====
        print(
            f"Epoch {ep}/{args.epochs} | "
            f"TrainLoss={tr_loss:.4f} | TrainAcc={tr_acc:.4f} | "
            f"ValLoss={va_loss:.4f} | ValAcc={va_acc:.4f}"
        )
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([ep, f"{tr_loss:.4f}", f"{tr_acc:.4f}", f"{va_loss:.4f}", f"{va_acc:.4f}"])

        torch.save(model.state_dict(),last_path)
        if va_acc>best_acc: 
            best_acc=va_acc
            torch.save(model.state_dict(),best_path)
            patience=0
            print("💾 saved best_text.pth")
        else:
            patience+=1
            if patience>=args.patience: 
                print("⏹️ Early stopping"); break

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_csv",default="C://pattyarea//project1_weighted//data//train.csv")
    ap.add_argument("--val_csv",default="C://pattyarea//project1_weighted//data//val.csv")
    ap.add_argument("--out_dir",default="C://pattyarea//project1_weighted//weights")
    ap.add_argument("--img_size",type=int,default=224)
    ap.add_argument("--batch_size",type=int,default=8)
    ap.add_argument("--epochs",type=int,default=50)
    ap.add_argument("--patience",type=int,default=50)
    ap.add_argument("--lr",type=float,default=2e-5)
    ap.add_argument("--weight_decay",type=float,default=0.01)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args();main(args)
