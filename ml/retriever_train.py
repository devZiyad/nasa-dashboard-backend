# Train a domain retriever (sentence-transformers)
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
import json, os, random

DATA = "data/retriever_pairs.jsonl"   # one JSON per line: {"query": "...", "positive": ["..."], "negatives": ["..."]}
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTDIR = "artifacts/retriever"
BATCH = 32
EPOCHS = 2

def load_pairs(path):
    ex = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            q = row["query"]
            for p in row["positive"]:
                ex.append(InputExample(texts=[q, p]))
    return ex

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    model = SentenceTransformer(MODEL)
    train_ex = load_pairs(DATA)
    random.shuffle(train_ex)
    loader = DataLoader(train_ex, shuffle=True, batch_size=BATCH)
    loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(train_objectives=[(loader, loss)],
              epochs=EPOCHS, warmup_steps=100,
              output_path=OUTDIR)

if __name__ == "__main__":
    main()
