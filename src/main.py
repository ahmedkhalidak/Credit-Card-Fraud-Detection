from pipelines.trainer import train_single_model, train_with_resampling
from pipelines.evaluate import compare_models
from models.voting_classifier import get_voting
from models.xgboost_model import get_xgb
from models.random_forest import get_rf
from src.pipelines.focal.train_focal import train_focal_model
from pipelines.model_selector import select_best

def main_menu():
    print("\n=== Fraud Detection System ===")
    print("1 - Train Logistic Regression")
    print("2 - Train Random Forest")
    print("3 - Train XGBoost")
    print("4 - Down-sample majority class")
    print("5 - Over-sample minority (SMOTE)")
    print("6 - Both (SMOTE + ENN)")
    print("7 - Voting Classifier (RF + LR)")
    print("8 - Train Focal Loss Neural Network ")
    print("9 - Compare All Models")
    print("10 - Best Model Selection")
    print("0 - Exit")

    choice = input("\nChoose an option: ")
    return choice

def main():
    while True:
        choice = main_menu()

        if choice == "1":
            train_single_model("logistic")
        elif choice == "2":
            train_single_model("random_forest")
        elif choice == "3":
            train_single_model("xgboost")
        elif choice == "4":
            train_with_resampling("down")
        elif choice == "5":
            train_with_resampling("smote")
        elif choice == "6":
            train_with_resampling("both")
        elif choice == "7":
            model = get_voting()
            train_single_model(model=model, name="Voting Classifier")
        elif choice == "8":
            print("\n=== Training Focal Loss Neural Network ===")

            from pipelines.trainer import load_processed, prep_data
            from src.pipelines.focal.train_focal import train_focal_model

            train, test = load_processed()
            X_train, X_test, y_train, y_test = prep_data(train, test)

            model, f1 = train_focal_model(X_train, y_train, X_test, y_test)

            print(f"\nFocal Loss NN F1 Score = {f1}")

        elif choice == "9":
            compare_models()
        elif choice == "10":
            select_best()
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice! Try again.")

if __name__ == "__main__":
    main()
