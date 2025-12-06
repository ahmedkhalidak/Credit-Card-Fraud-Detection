import os
import pickle
import torch

# ============================================
# 1) SAVE SKLEARN MODELS (RF, XGB, LR, Voting)
# ============================================

def save_sklearn_model(model, path="models/sklearn_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[Saved] Sklearn model → {path}")


def load_sklearn_model(path="models/sklearn_model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[Loaded] Sklearn model → {path}")
    return model


# ============================================
# 2) SAVE TORCH MODELS (Neural Network)
# ============================================

def save_torch_model(model, path="models/nn_model.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Saved] Torch model → {path}")


def load_torch_model(model_class, path="models/nn_model.pt", device="cpu"):
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"[Loaded] Torch model → {path}")
    return model
