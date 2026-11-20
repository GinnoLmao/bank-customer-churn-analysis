"""Training-only preprocessing and modest CV comparisons on a churn benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

SEED = 0
NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
           "HasCrCard", "IsActiveMember", "EstimatedSalary"]
CATEGORICAL = ["Geography", "Gender"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Exited"


def load_data(path):
    frame = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing benchmark columns: {sorted(missing)}")
    # The original notebook already drops identifiers; preserve that behavior and
    # make the final feature contract explicit so later derived columns cannot enter.
    frame = frame.loc[:, FEATURES + [TARGET]].copy()
    for name in NUMERIC:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if np.isinf(frame[NUMERIC].to_numpy(dtype=float)).any():
        raise ValueError("Infinite numeric values are not valid inputs")
    for name in CATEGORICAL:
        frame[name] = frame[name].map(lambda value: str(value).strip().lower() if pd.notna(value) else np.nan)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise")
    if frame[TARGET].isna().any() or set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Exited must contain both binary classes and no missing values")
    return frame[FEATURES], frame[TARGET].astype(int)


def preprocessing():
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", MinMaxScaler())]), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ], remainder="drop")


def make_pipeline(model):
    return Pipeline([("preprocess", preprocessing()), ("model", model)])


def probability_metrics(actual, probabilities, threshold=.5):
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    return {"accuracy": float(accuracy_score(actual, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "positive_precision": float(precision_score(actual, predicted, zero_division=0)),
            "positive_recall": float(recall_score(actual, predicted, zero_division=0)),
            "positive_f1": float(f1_score(actual, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(actual, probabilities)),
            "average_precision": float(average_precision_score(actual, probabilities)),
            "confusion_matrix_true_rows_predicted_columns": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
            "threshold": threshold}


def run(data: Path, output: Path):
    features, target = load_data(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=.2, random_state=SEED, stratify=target)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = {
        "Logistic regression": (LogisticRegression(solver="liblinear", max_iter=1000, random_state=SEED),
                                {"model__C": [.01, .1, 1.0], "model__class_weight": [None, "balanced"]}),
        "Decision tree": (DecisionTreeClassifier(random_state=SEED),
                          {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [10, 30],
                           "model__class_weight": [None, "balanced"]}),
    }
    scores, probabilities, searches, cv_rows = {}, {}, {}, []
    baseline = make_pipeline(DummyClassifier(strategy="prior"))
    baseline.fit(x_train, y_train)
    probabilities["Prior baseline"] = baseline.predict_proba(x_test)[:, 1]
    scores["Prior baseline"] = probability_metrics(y_test, probabilities["Prior baseline"])
    for name, (model, grid) in candidates.items():
        # Both imputation and encoding/scaling are refitted INSIDE every CV fold.
        search = GridSearchCV(make_pipeline(model), grid, cv=cv, scoring="average_precision",
                              refit=True, n_jobs=1, error_score="raise", return_train_score=False)
        search.fit(x_train, y_train)
        searches[name] = search
        probabilities[name] = search.predict_proba(x_test)[:, 1]
        scores[name] = probability_metrics(y_test, probabilities[name])
        scores[name]["training_cv_average_precision"] = float(search.best_score_)
        scores[name]["selected_parameters"] = search.best_params_
        for params, mean, std in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"],
                                     search.cv_results_["std_test_score"]):
            cv_rows.append({"model": name, "parameters": json.dumps(params, sort_keys=True),
                            "mean_cv_average_precision": float(mean), "std_cv_average_precision": float(std)})

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cv_rows).to_csv(output / "training_cv_comparison.csv", index=False)
    metric_names = ["accuracy", "balanced_accuracy", "positive_precision", "positive_recall",
                    "positive_f1", "roc_auc", "average_precision"]
    pd.DataFrame({name: {metric: value[metric] for metric in metric_names}
                  for name, value in scores.items()}).T.rename_axis("model").to_csv(output / "test_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for name, values in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, values)
        axes[0].plot(fpr, tpr, label=f"{name} ({scores[name]['roc_auc']:.3f})")
        if name == "Prior baseline":
            # The constant-prior predictor's AP is prevalence. Interpolating
            # its two PR endpoints would suggest a misleading diagonal curve.
            axes[1].axhline(y_test.mean(), linestyle="--", linewidth=1,
                            label=f"Prior baseline ({scores[name]['average_precision']:.3f})")
        else:
            precision, recall, _ = precision_recall_curve(y_test, values)
            axes[1].step(recall, precision, where="post",
                         label=f"{name} ({scores[name]['average_precision']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Holdout ROC; labels show ROC-AUC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Holdout precision–recall; labels show AP")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(output / "classification_curves.png", dpi=150)
    plt.close(fig)

    summary = {"task": "Educational bank-customer churn benchmark; no connection to an employer/client dataset",
               "input_file": data.name, "input_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
               "rows": len(features), "feature_names": FEATURES,
               "excluded_identifiers_and_artifacts": ["#", "RowNumber", "CustomerId", "Surname", "NumOfProducts_int"],
               "train_rows": len(x_train), "test_rows": len(x_test), "split_seed": SEED,
               "train_churn_prevalence": float(y_train.mean()), "test_churn_prevalence": float(y_test.mean()),
               "cv": "Five stratified training-only folds; entire preprocessing/model Pipeline searched by probability-based average precision",
               "test_use": "One final fixed 20% holdout; no threshold tuning or model selection on this holdout",
               "models": scores,
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scikit_learn": sklearn.__version__}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary, searches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/Churn_Modelling.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output)[0], indent=2))
