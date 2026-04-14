# src/evaluation/metrics.py
# Cálculo de métricas completas y generación de figuras de evaluación.
#
# PRINCIPIO de separación: este módulo no sabe nada del entrenamiento.
# Solo recibe predicciones y etiquetas, y devuelve métricas + figuras.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Predicciones ─────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Obtiene predicciones, etiquetas verdaderas y probabilidades.

    Returns:
        Tupla (preds, labels, probs) donde:
        - preds:  array de etiquetas predichas (0 o 1)
        - labels: array de etiquetas verdaderas
        - probs:  probabilidades para la clase PNEUMONIA (clase 1)
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(
    labels:    np.ndarray,
    preds:     np.ndarray,
    probs:     np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Calcula el conjunto completo de métricas de evaluación.

    Args:
        labels:    Etiquetas verdaderas (0=NORMAL, 1=PNEUMONIA).
        preds:     Predicciones del modelo.
        probs:     Probabilidades para clase positiva (PNEUMONIA).
        threshold: Umbral de decisión (default 0.5).

    Returns:
        Diccionario con todas las métricas escalares.

    # TODO: entender cómo se calcula recall.
    #       recall = TP / (TP + FN)
    #       En este contexto médico: TP = neumonías detectadas,
    #       FN = neumonías NO detectadas (el error más peligroso).

    # TODO: entender AUC-ROC.
    #       AUC-ROC mide la capacidad discriminativa del modelo
    #       independientemente del umbral de decisión.
    #       AUC=0.5 → el modelo no discrimina (igual que lanzar una moneda).
    #       AUC=1.0 → separación perfecta entre clases.
    """
    report = classification_report(
        labels, preds,
        target_names=["NORMAL", "PNEUMONIA"],
        output_dict=True,
    )
    return {
        "accuracy":              report["accuracy"],
        "precision_normal":      report["NORMAL"]["precision"],
        "recall_normal":         report["NORMAL"]["recall"],
        "f1_normal":             report["NORMAL"]["f1-score"],
        "precision_pneumonia":   report["PNEUMONIA"]["precision"],
        "recall_pneumonia":      report["PNEUMONIA"]["recall"],
        "f1_pneumonia":          report["PNEUMONIA"]["f1-score"],
        "auc_roc":               roc_auc_score(labels, probs),
        "average_precision":     average_precision_score(labels, probs),
    }


def print_report(metrics: dict[str, float], cfg: dict) -> None:
    """Imprime el reporte de evaluación con interpretación del criterio de éxito."""
    target = cfg["metrics"]["target"]
    recall = metrics["recall_pneumonia"]
    auc    = metrics["auc_roc"]

    print("\n══════════════════════════════════════════")
    print("         Reporte de evaluación — Test set")
    print("══════════════════════════════════════════")
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  Recall  PNEUMONIA  : {recall:.4f}  (objetivo ≥ {target})")
    print(f"  Precision PNEUMONIA: {metrics['precision_pneumonia']:.4f}")
    print(f"  F1 PNEUMONIA       : {metrics['f1_pneumonia']:.4f}")
    print(f"  AUC-ROC            : {auc:.4f}  (baseline trivial = 0.50)")
    print()

    # ── Evaluación de la hipótesis ──────────────────────────────────────────
    h1_recall = recall >= target
    h1_auc    = auc > 0.50
    h1        = h1_recall and h1_auc

    print("  Evaluación de la hipótesis H₁:")
    print(f"    recall ≥ {target}   : {'✓ CUMPLE' if h1_recall else '✗ NO CUMPLE'}")
    print(f"    AUC-ROC > 0.50 : {'✓ CUMPLE' if h1_auc    else '✗ NO CUMPLE'}")
    print()
    if h1:
        print("  → H₁ SOSTENIDA: el modelo supera los criterios clínicos.")
    else:
        print("  → H₀ NO RECHAZADA: el modelo no cumple los criterios clínicos.")
    print("══════════════════════════════════════════\n")


# ── Figuras ───────────────────────────────────────────────────────────────────

def save_evaluation_figures(
    labels:    np.ndarray,
    preds:     np.ndarray,
    probs:     np.ndarray,
    save_dir:  str | Path = "outputs/metrics",
) -> None:
    """
    Genera y guarda las figuras estándar de evaluación:
    - Matrices de confusión (conteos y normalizada)
    - Curvas ROC y Precision-Recall
    """
    if not HAS_MPL:
        print("matplotlib no disponible — omitiendo figuras.")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Matrices de confusión ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(cm, display_labels=["NORMAL", "PNEUMONIA"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title("Matriz de confusión (conteos)")

    cm_norm = confusion_matrix(labels, preds, normalize="true")
    ConfusionMatrixDisplay(cm_norm, display_labels=["NORMAL", "PNEUMONIA"]).plot(
        ax=axes[1], colorbar=False, cmap="Blues"
    )
    axes[1].set_title("Matriz de confusión (normalizada)")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # ── Curvas ROC y PR ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    axes[0].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Azar")
    axes[0].set_xlabel("Tasa de Falsos Positivos")
    axes[0].set_ylabel("Tasa de Verdaderos Positivos")
    axes[0].set_title("Curva ROC")
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    axes[1].plot(rec, prec, color="coral", lw=2, label=f"AP = {ap:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Curva Precision-Recall")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "roc_pr_curves.png", dpi=150)
    plt.close()
    print(f"Figuras guardadas en: {save_dir}")


def save_metrics_json(
    metrics:  dict[str, float],
    cfg:      dict,
    out_path: str | Path = "outputs/metrics/test_metrics.json",
) -> None:
    """Guarda las métricas en JSON compatible con DVC metrics."""
    out = {
        "metrics": metrics,
        "config": {
            "architecture":    cfg["model"]["architecture"],
            "freeze_backbone": cfg["model"]["freeze_backbone"],
            "w_pos":           cfg["loss"]["class_weights"]["pneumonia"],
            "lr":              cfg["training"]["learning_rate"],
            "seed":            cfg["seed"],
        },
        "hypothesis": {
            "H1_recall_ge_090": metrics["recall_pneumonia"] >= cfg["metrics"]["target"],
            "H1_auc_gt_050":    metrics["auc_roc"] > 0.50,
            "H1_sustained":     (metrics["recall_pneumonia"] >= cfg["metrics"]["target"])
                                 and (metrics["auc_roc"] > 0.50),
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Métricas guardadas: {out_path}")
