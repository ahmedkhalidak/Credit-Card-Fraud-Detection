import os
import pickle
import torch


# =============================================================
#   Unified Save Function  (Sklearn + PyTorch)
# =============================================================
def save_model(model, name):
    """
    Save any ML model (Sklearn or PyTorch) inside:
    models/saved_models/<name>.pkl or .pt
    """
    save_dir = "models/saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # Torch model
    if hasattr(model, "state_dict"):
        path = os.path.join(save_dir, f"{name}.pt")
        torch.save(model.state_dict(), path)
    
    # Sklearn model
    else:
        path = os.path.join(save_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)

    print(f"✔ Saved model → {path}")
    return path



# =============================================================
#   Unified Load Function  (Sklearn + PyTorch)
# =============================================================
def load_model(model_class=None, path=None):
    """
    Load any ML model (Sklearn or PyTorch).
    
    - model_class required only for PyTorch.
    - For Sklearn: model_class should be None.
    """

    if path.endswith(".pt"):  # Torch model
        model = model_class()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        print(f"✔ Loaded Torch model → {path}")
        return model
    
    else:                    # Sklearn model
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"✔ Loaded Sklearn model → {path}")
        return model
