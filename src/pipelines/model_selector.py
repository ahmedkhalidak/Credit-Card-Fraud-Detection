from pipelines.evaluate import compare_models
import os

def select_best(save_result=True):

    print("\n=== Selecting Best Model ===")

    df = compare_models(plot=False, save_reports=False)
    best_model = df["f1-score positive class"].idxmax()
    best_f1    = df["f1-score positive class"].max()

    print("\n============================")
    print(" BEST MODEL SELECTED ")
    print("============================")
    print(f"Model : {best_model}")
    print(f"F1    : {best_f1:.4f}\n")

    if save_result:
        os.makedirs("reports", exist_ok=True)
        with open("reports/best_model.txt", "w") as f:
            f.write(f"Best Model: {best_model}\n")
            f.write(f"F1 Score : {best_f1:.4f}\n")

    return best_model, best_f1
