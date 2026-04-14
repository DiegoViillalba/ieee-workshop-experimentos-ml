#!/usr/bin/env python
# scripts/train.py
# Script principal de entrenamiento — Día 2 Workshop ML
#
# Uso:
#   python scripts/train.py --config configs/baseline.yaml
#   python scripts/train.py --config configs/baseline.yaml --no-wandb
#
# Este script orquesta el pipeline de entrenamiento completo:
#   1. Cargar configuración desde YAML
#   2. Fijar semillas (reproducibilidad)
#   3. Preparar datos
#   4. Construir modelo
#   5. Entrenar con early stopping
#   6. Guardar checkpoint del mejor modelo
#   7. Registrar todo en W&B

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Añadir el directorio raíz al path para importar src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wandb

from src.data.dataset   import dataset_summary, get_dataloader
from src.models.resnet  import build_model, model_summary
from src.training.config  import config_summary, get_device, load_config, set_seed
from src.training.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento del clasificador de neumonía — Workshop ML Día 2"
    )
    parser.add_argument(
        "--config", type=str, default="configs/baseline.yaml",
        help="Ruta al archivo YAML de configuración.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Desactivar logging en Weights & Biases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Configuración ─────────────────────────────────────────────────────
    cfg = load_config(args.config)
    print(config_summary(cfg))

    # ── 2. Reproducibilidad ──────────────────────────────────────────────────
    set_seed(cfg["seed"])
    # TODO: verificar cómo afecta el seed a los resultados.
    #       Cambia seed en el YAML y ejecuta de nuevo. ¿Cambia el recall final?

    device = get_device()

    # ── 3. Datos ─────────────────────────────────────────────────────────────
    data_dir = cfg["data"]["raw_dir"]
    print(f"\nCargando datos desde: {data_dir}")

    # Mostrar resumen del dataset (si existe en disco)
    try:
        dataset_summary(data_dir)
    except Exception:
        print("  (dataset no encontrado en disco — asegúrate de descargarlo)")
        print("  Descarga: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia")

    # TODO: ¿qué impacto tiene batch_size en el entrenamiento?
    #       Cambia batch_size en el YAML y observa el tiempo por época y la estabilidad
    #       de las curvas de loss en W&B.
    train_loader = get_dataloader(
        data_dir=data_dir,
        split="train",
        batch_size=cfg["training"]["batch_size"],
        image_size=cfg["data"]["image_size"],
        seed=cfg["seed"],
    )
    val_loader = get_dataloader(
        data_dir=data_dir,
        split="val",
        batch_size=cfg["training"]["batch_size"],
        image_size=cfg["data"]["image_size"],
        seed=cfg["seed"],
    )

    # REPLACE: agrega data augmentation modificando get_transforms() en src/data/dataset.py
    #          Prueba añadir RandomGrayscale, GaussianBlur o ElasticTransform.
    #          ¿Mejora la generalización al test set?

    # ── 4. Modelo ─────────────────────────────────────────────────────────────
    print("\nConstruyendo modelo...")
    model = build_model(
        architecture=cfg["model"]["architecture"],
        pretrained=cfg["model"]["pretrained"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
    )
    model = model.to(device)
    model_summary(model)

    # TODO: entender qué capas se están fine-tuneando.
    #       model_summary() muestra cuántos parámetros tienen requires_grad=True.
    #       Con freeze_backbone=True debería ser ~1 026 (solo la capa final).
    # REPLACE: cambia freeze_backbone a false en el YAML para fine-tuning completo.
    #          Observa cuántos parámetros pasan a ser entrenables y cómo cambia
    #          el tiempo por época.

    # ── 5. W&B ───────────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        wandb_cfg = cfg.get("wandb", {})
        wandb_run = wandb.init(
            project=wandb_cfg.get("project", "pneumonia-classifier"),
            entity=wandb_cfg.get("entity"),
            tags=wandb_cfg.get("tags", []),
            config={
                "seed":            cfg["seed"],
                "architecture":    cfg["model"]["architecture"],
                "freeze_backbone": cfg["model"]["freeze_backbone"],
                "dropout":         cfg["model"]["dropout"],
                "epochs":          cfg["training"]["epochs"],
                "batch_size":      cfg["training"]["batch_size"],
                "lr":              cfg["training"]["learning_rate"],
                "weight_decay":    cfg["training"]["weight_decay"],
                "w_pos":           cfg["loss"]["class_weights"]["pneumonia"],
                "image_size":      cfg["data"]["image_size"],
            },
        )
        # TODO: comparar dos runs en W&B.
        #       Después de este run, modifica lr en el YAML y ejecuta de nuevo.
        #       Abre wandb.ai y compara las curvas val/recall de ambos runs.
        # REPLACE: cambia learning_rate y observa los efectos en las curvas de W&B.

    # ── 6. Entrenamiento ──────────────────────────────────────────────────────
    print("\nIniciando entrenamiento...")
    # REPLACE: modifica w_pos en el YAML y observa el recall.
    #          ¿Hay un punto en que subir el peso daña la precision?
    # TODO: identificar dónde se calcula recall — ver src/training/trainer.py
    # TODO: entender cómo se selecciona el mejor modelo — ver trainer.py, early stopping

    model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        wandb_run=wandb_run,
    )

    # ── 7. Cierre ─────────────────────────────────────────────────────────────
    if wandb_run is not None:
        artifact = wandb.Artifact("pneumonia-model", type="model")
        artifact.add_file("outputs/checkpoints/best_model.pt")
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    print(f"\nEntrenamiento completado.")
    print(f"Checkpoint guardado: outputs/checkpoints/best_model.pt")
    print(f"Siguiente paso: python scripts/evaluate.py --config {args.config}")


if __name__ == "__main__":
    main()
