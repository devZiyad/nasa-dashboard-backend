# ml/evidence_classifier.py
import os, numpy as np, pandas as pd
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"   # keep vision deps out

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

BASE = "distilbert-base-uncased"
OUT  = "artifacts/evidence_clf"
DATA = "data/evidence_labeled.csv"

def _to_ds(csv_path):
    df = pd.read_csv(csv_path)  # columns: text, label (0/1)
    return Dataset.from_pandas(df[["text","label"]])

def _tok_fn(tok):
    def _inner(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=256)
    return _inner

def train():
    os.makedirs(OUT, exist_ok=True)
    ds = _to_ds(DATA).train_test_split(test_size=0.1, seed=42)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)

    train_ds = ds["train"].map(_tok_fn(tok), batched=True)
    eval_ds  = ds["test"].map(_tok_fn(tok),  batched=True)

    # Minimal args for broad transformers compatibility (no evaluation_strategy!)
    args = TrainingArguments(
        output_dir=OUT,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        logging_steps=50
    )

    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        acc = (preds == p.label_ids).mean()
        return {"accuracy": float(acc)}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        compute_metrics=metrics
    )
    trainer.train()
    try:
        print(trainer.evaluate())
    except Exception:
        pass

    trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    print("[ok] model saved ->", OUT)

def predict(texts):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tok = AutoTokenizer.from_pretrained(OUT)
    mdl = AutoModelForSequenceClassification.from_pretrained(OUT)
    enc = tok(list(texts), truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = mdl(**enc).logits
        probs = logits.softmax(-1).cpu().numpy()[:,1]
    return probs

if __name__ == "__main__":
    train()
