from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from .models import train_classifier, evaluate_classifier


def save_confusion_matrix(cm, classes, title, outfile):
    vmax = cm.max()
    fig, ax = plt.subplots(figsize=(10,6))

    im = ax.imshow(cm, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:.1f}", ha="center", va="center", color="black")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def kfold_evaluate(
    X,
    y,
    model_type,
    k=5,
    outdir="outputs/classification/heart",
    prefix="heart"
):
    os.makedirs(outdir, exist_ok=True)

    metrics_list = []
    cms = []

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = train_classifier(X_train, y_train, model_type=model_type)
        metrics = evaluate_classifier(clf, X_val, y_val, plot_cm=False)

        metrics_list.append({
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
        })

        cms.append(metrics["confusion_matrix"])

    avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()
    avg_cm = np.mean(cms, axis=0)

    csv_path = os.path.join(outdir, f"{prefix}_{model_type}_AVERAGE.csv")
    pd.DataFrame([avg_metrics]).to_csv(csv_path, index=False)

    cm_path = os.path.join(outdir, f"{prefix}_{model_type}_AVERAGE_CM.png")
    save_confusion_matrix(
        avg_cm,
        classes=np.unique(y),
        title=f"{prefix.upper()} - {model_type} (Averaged {k}-Fold CM)",
        outfile=cm_path
    )

    return avg_metrics, avg_cm, metrics_list, cms

