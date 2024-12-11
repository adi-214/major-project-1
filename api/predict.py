from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import torch
import pandas as pd
import os

from model import load_model, smiles_to_input
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cpu')
MODEL_PATH = "best_model_tox21.pth"

# Assume we have the same dataset structure available or we know the tasks from a file
df = pd.read_csv('tox21.csv', nrows=1)
tasks = df.columns.tolist()[1:]  # all columns except 'smiles' are tasks
n_tasks = len(tasks)

# Load the model architecture (without loading the state_dict inside load_model)
# Modify load_model in model.py so it only returns the model architecture.
model = load_model(MODEL_PATH, n_tasks=n_tasks, device=DEVICE, load_state=False)

# Manually load the state dict with strict=False
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model_state = model.state_dict()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
model.load_state_dict(filtered_state_dict, strict=False)

model.eval()


@app.get("/api/predict")
async def predict(smiles: str = Query(..., description="SMILES string")):
    g, fp = smiles_to_input(smiles)
    if g is None or fp is None:
        return JSONResponse(content={"error": "Invalid SMILES string."}, status_code=400)
    g = g.to(DEVICE)
    fp = torch.tensor(fp, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    node_feats = g.ndata.pop('h')
    edge_feats = g.edata.pop('e')

    with torch.no_grad():
        logits = model(g, node_feats, edge_feats, fp)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    predictions = []
    for task_name, p in zip(tasks, probs):
        predictions.append({"task": task_name, "prob": float(p)})

    return {"predictions": predictions}
