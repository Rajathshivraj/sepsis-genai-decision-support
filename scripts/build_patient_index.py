import os
import pickle
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATA_DIR = "data/raw/training_setA"
INDEX_PATH = "models/patient_index.faiss"
CASES_PATH = "models/patient_cases.pkl"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

cases = []
texts = []

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".psv")])

print(f"Processing {len(files)} patient files...")

for f in tqdm(files):

    path = os.path.join(DATA_DIR, f)

    df = pd.read_csv(path, sep="|")

    hr = df["HR"].mean()
    temp = df["Temp"].mean()
    mapv = df["MAP"].mean()
    resp = df["Resp"].mean()
    lact = df["Lactate"].max()

    sepsis = int(df["SepsisLabel"].max())

    summary = (
        f"Patient HR {hr:.1f}, Temp {temp:.1f}, MAP {mapv:.1f}, "
        f"Resp {resp:.1f}, Lactate {lact:.1f}. "
        f"{'Sepsis diagnosed' if sepsis else 'No sepsis diagnosed'}."
    )

    texts.append(summary)

    cases.append({
        "patient_id": f,
        "summary": summary,
        "sepsis": sepsis
    })

print("Computing embeddings...")

embeddings = model.encode(texts, show_progress_bar=True)

dim = embeddings.shape[1]

index = faiss.IndexFlatL2(dim)

index.add(np.array(embeddings).astype("float32"))

print("Saving index...")

faiss.write_index(index, INDEX_PATH)

with open(CASES_PATH, "wb") as f:
    pickle.dump(cases, f)

print("Done.")

print(f"Index size: {index.ntotal}")
