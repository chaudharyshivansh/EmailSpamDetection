"""Train the email spam detection model and save it to spam_model.joblib.

Run:  python train.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from spam_utils import FEATURES, TARGET, make_pipeline

HERE = Path(__file__).parent
DATA_PATH = HERE / "data" / "email_spam_detection.csv"
MODEL_PATH = HERE / "spam_model.joblib"

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df[FEATURES], df[TARGET]


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in candidates.items():
        pipe = make_pipeline(model)
        score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean()
        results[name] = score
        print(f"{name:<22} CV F1 = {score:.4f}")

    best_name = max(results, key=results.get)
    print(f"\nBest model: {best_name}")

    best = make_pipeline(candidates[best_name]).fit(X_train, y_train)
    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    print("\nHold-out test set")
    print(classification_report(y_test, pred, target_names=["Not spam", "Spam"], digits=4))
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, pred))
    print(f"ROC AUC: {roc_auc_score(y_test, proba):.4f}   F1: {f1_score(y_test, pred):.4f}")

    # Refit on all data for the final saved model
    final = make_pipeline(candidates[best_name]).fit(X, y)
    joblib.dump(final, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
