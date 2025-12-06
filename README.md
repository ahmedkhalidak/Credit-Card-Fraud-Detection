# Credit Card Fraud Detection using Machine Learning (Imbalanced Data)




## 📂 Dataset Setup

This project uses the **Credit Card Fraud Detection** dataset (Kaggle).

**it is NOT included in this repository** 

➡️ You MUST download the dataset manually from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading `creditcard.csv`, place it in the following directory:
``` 
credit-card-fraud/
└── data/
     ├── raw/
     │     └── creditcard.csv   ← put the dataset here
     │
     └── processed/
```

## 🐍 Create Conda Environment

```
$ conda create -n fraud-env python=3.10 -y
```
```
$ conda activate fraud-env
```

## 📦 Install Dependencies
```
$ pip install -r requirements.txt
```