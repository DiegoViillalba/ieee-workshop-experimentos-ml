#!/usr/bin/env python
# scripts/evaluate.py
# Evaluación final en el test set — se ejecuta UNA SOLA VEZ al final.
#
# Uso:
#   python scripts/evaluate.py --config configs/baseline.yaml
#   python scripts/evaluate.py --config configs/baseline.yaml --no-wandb
#
# PRINCIPIO: el test set se evalúa solo al final.
# Nunca uses el test set para ajustar hiperparámetros.
# Para eso existe el val set.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import wandb

from src.evaluation.metrics import (
    compute_baseline,
    compute_metrics,
    print_report,
    save_figures,
    save_metrics_json,
)
from src.models.logistic    import load_model
from src.training.config    import load_config, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluación en test set — Workshop ML Día 2"
    )
    p.add_argument("--config",     default="configs/baseline.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/model.pkl")
    p.add_argument("--no-wandb",   action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    pos_label = cfg["data"]["positive_label"]

    # ── Cargar datos procesados ───────────────────────────────────────────────
    processed = Path("data/processed")
    X_train = np.load(processed / "X_train.npy")
    y_train = np.load(processed / "y_train.npy")
    X_test  = np.load(processed / "X_test.npy")
    y_test  = np.load(processed / "y_test.npy")
    print(f"Test set: {len(y_test)} muestras")

    # ── Cargar modelo ─────────────────────────────────────────────────────────
    model = load_model(args.checkpoint)

    # ── Baseline ──────────────────────────────────────────────────────────────
    # TODO: entender por qué calculamos un baseline.
    #       El DummyClassifier (en src/evaluation/metrics.py) siempre predice
    #       la clase más frecuente. Ver compute_baseline() para los detalles.
    #       El clasificador de mayoría siempre predice "benigno" (clase más frecuente).
    #       Su accuracy = 63% parece aceptable, pero su recall de maligno = 0.0.
    #       Para que nuestro modelo sea útil debe superar AMBAS métricas.
    baseline_metrics = compute_baseline(X_train, y_train, X_test, y_test, pos_label)
    print(f"\nBaseline (mayoría): "
          f"accuracy={baseline_metrics['baseline/accuracy']:.4f}  "
          f"recall={baseline_metrics['baseline/recall']:.4f}  "
          f"AUC={baseline_metrics['baseline/auc']:.4f}")

    # ── Métricas del modelo ────────────────────────────────────────────────────
    # TODO: interpretar métricas.
    #       Observa recall_malignant y AUC-ROC.
    #       ¿El modelo supera al baseline en ambas?
    #       ¿Cumple el criterio clínico recall ≥ 0.90?
    metrics = compute_metrics(model, X_test, y_test, pos_label,
                              cfg["metrics"]["threshold"])
    print_report(metrics, baseline_metrics, cfg)

    # ── Guardar resultados ────────────────────────────────────────────────────
    save_metrics_json(metrics, baseline_metrics, cfg)
    save_figures(model, X_test, y_test, pos_label=pos_label)

    # ── W&B ───────────────────────────────────────────────────────────────────
    if not args.no_wandb:
        wandb_cfg = cfg.get("wandb", {})
        run = wandb.init(
            project=wandb_cfg.get("project", "breast-cancer-classifier"),
            entity=wandb_cfg.get("entity"),
            tags=wandb_cfg.get("tags", []) + ["evaluation"],
            job_type="evaluate",
        )
        run.log({**metrics, **baseline_metrics})
        run.log({
            "test/confusion_matrix": wandb.Image("outputs/metrics/confusion_matrix.png"),
            "test/roc_pr_curves":    wandb.Image("outputs/metrics/roc_pr_curves.png"),
        })
        run.finish()

    print("Evaluación completada.")


if __name__ == "__main__":
    main()
