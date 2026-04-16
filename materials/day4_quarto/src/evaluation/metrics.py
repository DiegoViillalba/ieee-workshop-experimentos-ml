# src/evaluation/metrics.py
# Cálculo de métricas completas, baseline y generación de figuras.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def compute_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    pos_label: int = 0,
) -> dict[str, float]:
    """
    Entrena un clasificador de mayoría y devuelve sus métricas en test.

    # TODO: entender por qué el baseline es necesario.
    #       Un clasificador que siempre predice 'benigno' obtiene accuracy = 63%
    #       — parece razonable — pero su recall de maligno es 0.0.
    #       El baseline muestra el piso mínimo: cualquier modelo útil
    #       debe superar AMBAS métricas (recall Y AUC-ROC) simultáneamente.
    """
    dummy = DummyClassifier(strategy="most_frequent", random_state=0)
    dummy.fit(X_train, y_train)
    preds = dummy.predict(X_test)
    probs = dummy.predict_proba(X_test)[:, 1]   # P(benigno) para roc_auc_score

    return {
        "baseline/accuracy": float(np.mean(preds == y_test)),
        "baseline/recall":   recall_score(y_test, preds, pos_label=pos_label,
                                          zero_division=0.0),
        "baseline/auc":      roc_auc_score(y_test, probs),
    }


def compute_metrics(
    model:     BaseEstimator,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    pos_label: int = 0,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Calcula el conjunto completo de métricas en el test set.

    # TODO: interpretar métricas.
    #       recall_malignant : de todos los tumores malignos reales,
    #                          ¿qué fracción detectó el modelo?
    #       auc_roc          : capacidad discriminativa entre maligno y benigno,
    #                          independiente del umbral.
    #       AUC = 0.5 → el modelo no discrimina (igual que lanzar una moneda).
    #       AUC = 1.0 → separación perfecta.
    """
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]   # P(benigno) para roc_auc_score

    report = classification_report(
        y_test, preds,
        target_names=["maligno", "benigno"],
        output_dict=True,
    )
    return {
        "accuracy":            report["accuracy"],
        "recall_malignant":    report["maligno"]["recall"],
        "precision_malignant": report["maligno"]["precision"],
        "f1_malignant":        report["maligno"]["f1-score"],
        "recall_benign":       report["benigno"]["recall"],
        "auc_roc":             roc_auc_score(y_test, probs),
        "average_precision":   average_precision_score(y_test, probs,
                                                        pos_label=pos_label),
    }


def print_report(
    metrics:          dict[str, float],
    baseline_metrics: dict[str, float],
    cfg:              dict,
) -> None:
    """Imprime el reporte con evaluación de la hipótesis H₁."""
    target = cfg["metrics"]["target"]
    recall = metrics["recall_malignant"]
    auc    = metrics["auc_roc"]

    print("\n══════════════════════════════════════════")
    print("       Reporte de evaluación — Test set")
    print("══════════════════════════════════════════")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}   "
          f"(baseline: {baseline_metrics['baseline/accuracy']:.4f})")
    print(f"  Recall maligno    : {recall:.4f}   "
          f"(baseline: {baseline_metrics['baseline/recall']:.4f})")
    print(f"  Precision maligno : {metrics['precision_malignant']:.4f}")
    print(f"  AUC-ROC           : {auc:.4f}   "
          f"(baseline: {baseline_metrics['baseline/auc']:.4f})")
    print()

    h1_recall = recall >= target
    h1_auc    = auc    > 0.50
    h1        = h1_recall and h1_auc

    print("  Evaluación de la hipótesis H₁:")
    print(f"    recall ≥ {target}  : {'✓ CUMPLE' if h1_recall else '✗ NO CUMPLE'}")
    print(f"    AUC-ROC > 0.50  : {'✓ CUMPLE' if h1_auc    else '✗ NO CUMPLE'}")
    print()
    if h1:
        print("  → H₁ SOSTENIDA")
    else:
        print("  → H₀ NO RECHAZADA")
    print("══════════════════════════════════════════\n")


def save_figures(
    model:    BaseEstimator,
    X_test:   np.ndarray,
    y_test:   np.ndarray,
    save_dir: str | Path = "outputs/metrics",
    pos_label: int = 0,
) -> None:
    """Guarda matrices de confusión y curvas ROC / PR."""
    if not HAS_MPL:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]   # P(benigno) para roc_auc_score

    # Matrices de confusión
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm, display_labels=["maligno", "benigno"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Matriz de confusión (conteos)")
    cm_n = confusion_matrix(y_test, preds, normalize="true")
    ConfusionMatrixDisplay(cm_n, display_labels=["maligno", "benigno"]).plot(
        ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title("Matriz de confusión (normalizada)")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # ROC y PR
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fpr, tpr, _ = roc_curve(y_test, probs, pos_label=pos_label)
    auc = roc_auc_score(y_test, probs)
    axes[0].plot(fpr, tpr, lw=2, color="steelblue", label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Azar")
    axes[0].set(xlabel="FPR", ylabel="TPR", title="Curva ROC")
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(y_test, probs, pos_label=pos_label)
    ap = average_precision_score(y_test, probs, pos_label=pos_label)
    axes[1].plot(rec, prec, lw=2, color="coral", label=f"AP = {ap:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Curva PR")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "roc_pr_curves.png", dpi=150)
    plt.close()
    print(f"Figuras guardadas en: {save_dir}")


def save_metrics_json(
    metrics:          dict[str, float],
    baseline_metrics: dict[str, float],
    cfg:              dict,
    out_path:         str | Path = "outputs/metrics/test_metrics.json",
) -> None:
    """Guarda métricas en JSON compatible con dvc metrics."""
    out = {
        "metrics":  metrics,
        "baseline": baseline_metrics,
        "config": {
            "model":         cfg["model"]["type"],
            "class_weight":  cfg["model"]["class_weight"],
            "C":             cfg["model"].get("C", 1.0),
            "seed":          cfg["seed"],
        },
        "hypothesis": {
            "H1_recall_ge_090": metrics["recall_malignant"] >= cfg["metrics"]["target"],
            "H1_auc_gt_050":    metrics["auc_roc"] > 0.50,
            "H1_sustained": (
                metrics["recall_malignant"] >= cfg["metrics"]["target"]
                and metrics["auc_roc"] > 0.50
            ),
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Métricas guardadas: {out_path}")
