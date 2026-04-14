#!/usr/bin/env python
# scripts/evaluate.py
# Evaluación completa del modelo entrenado sobre el test set.
#
# Uso:
#   python scripts/evaluate.py --config configs/baseline.yaml
#   python scripts/evaluate.py --config configs/baseline.yaml --no-wandb
#
# PRINCIPIO: el test set se evalúa UNA SOLA VEZ al final.
# Nunca uses el test set para seleccionar hiperparámetros o comparar modelos.
# Para eso existe el val set.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wandb

from src.data.dataset        import get_dataloader
from src.evaluation.metrics  import (
    compute_metrics,
    get_predictions,
    print_report,
    save_evaluation_figures,
    save_metrics_json,
)
from src.models.resnet       import build_model, load_checkpoint
from src.training.config     import get_device, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación del clasificador de neumonía — Workshop ML Día 2"
    )
    parser.add_argument(
        "--config", type=str, default="configs/baseline.yaml",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="outputs/checkpoints/best_model.pt",
        help="Ruta al checkpoint del modelo entrenado.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Configuración y semilla ───────────────────────────────────────────────
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = build_model(
        architecture=cfg["model"]["architecture"],
        pretrained=False,          # cargaremos los pesos del checkpoint
        freeze_backbone=False,     # en evaluación no importa
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
    )
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)

    # ── Test set ──────────────────────────────────────────────────────────────
    print("\nCargando test set...")
    test_loader = get_dataloader(
        data_dir=cfg["data"]["raw_dir"],
        split="test",
        batch_size=cfg["training"]["batch_size"],
        image_size=cfg["data"]["image_size"],
        seed=cfg["seed"],
    )
    print(f"Test: {len(test_loader.dataset)} imágenes")

    # ── Predicciones ──────────────────────────────────────────────────────────
    print("\nGenerando predicciones sobre el test set...")
    preds, labels, probs = get_predictions(model, test_loader, device)
    # TODO: entender cómo se calcula recall — ver src/evaluation/metrics.py
    # TODO: entender AUC-ROC — ver compute_metrics() en el mismo módulo

    # ── Baseline de comparación ───────────────────────────────────────────────
    import numpy as np
    majority_class = int(np.bincount(labels).argmax())
    baseline_preds = np.full_like(labels, fill_value=majority_class)
    from sklearn.metrics import recall_score
    baseline_recall = recall_score(labels, baseline_preds, pos_label=1)
    # TODO: explicar por qué este baseline es válido.
    #       El clasificador de mayoría siempre predice la clase más frecuente (PNEUMONIA).
    #       Su recall = 1.0 parece perfecto, pero su AUC-ROC = 0.5 revela que no discrimina.
    #       Un modelo útil debe superar AMBAS métricas simultáneamente.
    print(f"\nBaseline (mayoría): recall={baseline_recall:.4f}, AUC-ROC=0.500")

    # ── Métricas ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(labels, preds, probs, threshold=cfg["metrics"]["threshold"])
    print_report(metrics, cfg)

    # ── Guardar resultados ────────────────────────────────────────────────────
    save_metrics_json(metrics, cfg)
    save_evaluation_figures(labels, preds, probs)

    # ── Log en W&B ────────────────────────────────────────────────────────────
    if not args.no_wandb:
        wandb_cfg = cfg.get("wandb", {})
        run = wandb.init(
            project=wandb_cfg.get("project", "pneumonia-classifier"),
            entity=wandb_cfg.get("entity"),
            tags=wandb_cfg.get("tags", []) + ["evaluation"],
            job_type="evaluate",
        )
        run.log({
            "test/recall_pneumonia":   metrics["recall_pneumonia"],
            "test/auc_roc":            metrics["auc_roc"],
            "test/precision_pneumonia":metrics["precision_pneumonia"],
            "test/f1_pneumonia":       metrics["f1_pneumonia"],
            "test/accuracy":           metrics["accuracy"],
            "baseline/recall":         baseline_recall,
        })
        run.log({
            "test/confusion_matrix": wandb.Image("outputs/metrics/confusion_matrix.png"),
            "test/roc_pr_curves":    wandb.Image("outputs/metrics/roc_pr_curves.png"),
        })
        run.finish()

    print("Evaluación completada.")


if __name__ == "__main__":
    main()
