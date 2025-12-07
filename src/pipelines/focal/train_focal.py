import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.foacl.focal_loss import FocalLoss
from src.models.foacl.fraud_nn import FraudNN
from utils.preprocess import scale_all_features
from utils.save_load import save_model


def train_focal_model(X_train, y_train, X_test, y_test, epochs=50, lr=1e-3, batch_size=512, return_scores=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Scale ALL features
    X_train_scaled, X_test_scaled, _ = scale_all_features(X_train, X_test)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1).to(device)

    X_test_t  = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_np = y_test.values

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    model = FraudNN(X_train_t.shape[1]).to(device)
    criterion = FocalLoss(alpha=0.75, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    logits = model(X_test_t).detach().cpu()
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)

    f1 = f1_score(y_test_np, preds)

    # 🔥 Auto-save Torch model
    save_model(model, "focal_nn_model")

    if return_scores:
        return model, f1, probs

    return model, f1
