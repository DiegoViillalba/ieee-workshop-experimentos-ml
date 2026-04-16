#!/usr/bin/env python
# scripts/train.py
# Entrenamiento del clasificador de cáncer de mama.
#
# Uso:
#   python scripts/train.py --config configs/baseline.yaml
#   python scripts/train.py --config configs/baseline.yaml --no-wandb

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wandb

from src.data.dataset      import dataset_summary, load_and_split
from src.models.logistic   import build_model, save_model
from src.training.config   import config_summary, load_config, set_seed
from src.training.trainer  import evaluate_split, train_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entrena el clasificador de cáncer de mama — Workshop ML Día 2"
    )
    p.add_argument("--config",    default="configs/baseline.yaml")
    p.add_argument("--no-wandb",  action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Configuración ─────────────────────────────────────────────────────
    cfg = load_config(args.config)
    print(config_summary(cfg))

    # ── 2. Reproducibilidad ──────────────────────────────────────────────────
    set_seed(cfg["seed"])
    # TODO: verificar cómo afecta el seed al resultado.
    #       Cambia seed en el YAML y vuelve a ejecutar. ¿Cambia el recall?

    # ── 3. Datos ─────────────────────────────────────────────────────────────
    print("\nCargando dataset...")
    dataset_summary()

    # TODO: ¿por qué dividir en train / val / test y no en solo train / test?
    #       Val se usa para ajustar hiperparámetros (como C o class_weight).
    #       Test se evalúa UNA SOLA VEZ al final para reportar el resultado real.
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        seed=cfg["seed"],
        save_dir="data/processed",
    )
    print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    # ── 4. Modelo ─────────────────────────────────────────────────────────────
    print("\nConstruyendo modelo...")
    model = build_model(cfg)
    # TODO: ¿por qué usar class_weight="balanced"?
    #       Ver src/models/logistic.py para la explicación completa.
    # REPLACE: cambia class_weight en el YAML y observa el efecto en recall.

    # ── 5. Entrenamiento ──────────────────────────────────────────────────────
    print("Entrenando...")
    model = train_model(model, X_train, y_train)

    # ── 6. Evaluación en validación ───────────────────────────────────────────
    val_metrics = evaluate_split(
        model, X_val, y_val,
        split="val",
        pos_label=cfg["data"]["positive_label"],
    )
    print(
        f"\nVal — accuracy: {val_metrics['val/accuracy']:.4f}  "
        f"recall: {val_metrics['val/recall']:.4f}  "
        f"AUC: {val_metrics['val/auc']:.4f}"
    )

    # ── 7. Guardar modelo ─────────────────────────────────────────────────────
    # DVC rastrea este output: cualquier cambio en código o config lo invalida
    save_model(model, "outputs/checkpoints/model.pkl")

    # ── 8. W&B ───────────────────────────────────────────────────────────────
    if not args.no_wandb:
        wandb_cfg = cfg.get("wandb", {})
        run = wandb.init(
            project=wandb_cfg.get("project", "breast-cancer-classifier"),
            entity=wandb_cfg.get("entity"),
            tags=wandb_cfg.get("tags", []),
            config={
                "seed":         cfg["seed"],
                "model":        cfg["model"]["type"],
                "class_weight": cfg["model"]["class_weight"],
                "C":            cfg["model"].get("C", 1.0),
                "max_iter":     cfg["model"]["max_iter"],
                "test_size":    cfg["data"]["test_size"],
            },
        )
        run.log(val_metrics)
        # TODO: comparar dos runs en W&B.
        #       Después de este run, cambia class_weight en el YAML y ejecuta de nuevo.
        #       Abre wandb.ai y compara las curvas val/recall de ambos runs.
        # REPLACE: cambia learning_rate y observa los efectos en las curvas de W&B.
        run.finish()

    print(f"\nEntrenamiento completado.")
    print(f"Siguiente paso: python scripts/evaluate.py --config {args.config}")


if __name__ == "__main__":
    main()
