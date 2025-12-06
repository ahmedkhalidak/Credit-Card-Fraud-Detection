from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_voting():
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced")
    lr = LogisticRegression(class_weight="balanced")
    
    voting = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("lr", lr),
        ],
        voting="soft"
    )
    return voting
