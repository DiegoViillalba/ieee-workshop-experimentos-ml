# src/training/trainer.py
# Loop de entrenamiento completo con early stopping, logging en W&B
# y guardado del mejor checkpoint.
#
# PRINCIPIO de separación de responsabilidades:
#   Este módulo SOLO entrena. No carga datos, no define el modelo,
#   no evalúa en test. Cada responsabilidad vive en su propio módulo.

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


def build_criterion(cfg: dict, device: torch.device) -> nn.CrossEntropyLoss:
    """
    Construye la función de pérdida ponderada.

    Args:
        cfg:    Diccionario de configuración (debe tener cfg['loss']['class_weights']).
        device: Dispositivo donde se ejecutará la pérdida.

    Returns:
        nn.CrossEntropyLoss con pesos de clase.

    # TODO: por qué usamos pesos de clase.
    #       Sin pesos, CrossEntropyLoss trata NORMAL y PNEUMONIA como igualmente
    #       importantes. Con w_pos=2.0, cada error en PNEUMONIA penaliza el doble,
    #       empujando al modelo hacia mayor recall de la clase positiva.
    # REPLACE: modifica loss.class_weights.pneumonia en el YAML (prueba 3.0, 4.0).
    #          ¿Hay un punto en que subir el peso daña la precision?
    """
    w_normal    = cfg["loss"]["class_weights"]["normal"]
    w_pneumonia = cfg["loss"]["class_weights"]["pneumonia"]

    # Orden de clases: 0=NORMAL, 1=PNEUMONIA (orden alfabético de ImageFolder)
    weights = torch.tensor([w_normal, w_pneumonia], dtype=torch.float, device=device)
    return nn.CrossEntropyLoss(weight=weights)


def build_optimizer(model: nn.Module, cfg: dict) -> AdamW:
    """Construye el optimizador AdamW solo sobre parámetros entrenables."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return AdamW(
        trainable,
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )


# ── Epoch functions ───────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> dict[str, float]:
    """
    Ejecuta una época completa de entrenamiento.

    Returns:
        Diccionario con train/loss y train/acc de la época.

    # TODO: identificar dónde se calcula recall durante el entrenamiento.
    #       Pista: ¡no se calcula! El recall de entrenamiento no es informativo
    #       porque el modelo está optimizando activamente sobre esos datos.
    #       Solo medimos recall en validación (conjunto no visto).
    """
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds         = logits.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += images.size(0)

    return {
        "train/loss": running_loss / total,
        "train/acc":  correct / total,
    }


@torch.no_grad()
def evaluate_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    split:     str = "val",
) -> dict[str, float]:
    """
    Evalúa el modelo en un split dado sin actualizar pesos.

    Returns:
        Diccionario con loss, acc, recall y AUC-ROC del split.

    # TODO: entender cómo se calcula recall.
    #       recall = TP / (TP + FN)
    #       En diagnóstico médico, un FN (neumonía no detectada) es más costoso
    #       que un FP (alarma falsa). Por eso recall es nuestra métrica primaria.
    """
    model.eval()
    running_loss = 0.0
    all_preds    = []
    all_labels   = []
    all_probs    = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)
        probs  = torch.softmax(logits, dim=1)[:, 1]   # P(PNEUMONIA)
        preds  = logits.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    n = len(all_labels)
    return {
        f"{split}/loss":   running_loss / n,
        f"{split}/acc":    float(np.mean(np.array(all_preds) == np.array(all_labels))),
        f"{split}/recall": recall_score(all_labels, all_preds, pos_label=1),
        f"{split}/auc":    roc_auc_score(all_labels, all_probs),
    }


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          dict[str, Any],
    device:       torch.device,
    wandb_run:    Any | None = None,
) -> tuple[nn.Module, dict]:
    """
    Loop de entrenamiento completo con early stopping y logging en W&B.

    Guarda el mejor modelo en outputs/checkpoints/best_model.pt
    basado en val/recall (métrica primaria clínica).

    Args:
        model:        Modelo PyTorch inicializado y enviado al device.
        train_loader: DataLoader de entrenamiento.
        val_loader:   DataLoader de validación.
        cfg:          Configuración completa del experimento.
        device:       Dispositivo de cómputo.
        wandb_run:    Objeto wandb.Run para logging (None = sin logging).

    Returns:
        Tupla (mejor_modelo, historial_de_métricas).

    # TODO: entender cómo se selecciona el mejor modelo.
    #       Pista: guardamos el checkpoint cuando val/recall > best_recall.
    #       ¿Por qué usamos recall y no accuracy para seleccionar el mejor modelo?
    """
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    criterion = build_criterion(cfg, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    best_recall      = 0.0
    patience_counter = 0
    history: dict[str, list] = {
        "train/loss": [], "train/acc": [],
        "val/loss":   [], "val/acc":   [],
        "val/recall": [], "val/auc":   [],
    }

    print(f"\nIniciando entrenamiento: {cfg['training']['epochs']} épocas")
    print(f"Early stopping: patience={cfg['training']['patience']}")
    print(f"Checkpoint: {checkpoint_path}\n")

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics   = evaluate_epoch(model, val_loader, criterion, device, split="val")
        scheduler.step()

        elapsed = time.time() - t0
        metrics = {
            **train_metrics,
            **val_metrics,
            "epoch":        epoch,
            "lr":           scheduler.get_last_lr()[0],
            "epoch_time_s": elapsed,
        }

        # Log en W&B
        if wandb_run is not None:
            wandb_run.log(metrics)
            # TODO: comparar dos runs en W&B.
            #       Ejecuta el entrenamiento dos veces con distintos lr y compara
            #       las curvas val/recall en el dashboard.

        # Guardar historial
        for k in history:
            history[k].append(metrics[k])

        # Imprimir progreso
        print(
            f"Época {epoch:3d}/{cfg['training']['epochs']} | "
            f"loss {train_metrics['train/loss']:.4f} | "
            f"val_recall {val_metrics['val/recall']:.4f} | "
            f"val_auc {val_metrics['val/auc']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Early stopping + guardado del mejor modelo
        current_recall = val_metrics["val/recall"]
        if current_recall > best_recall:
            best_recall = current_recall
            torch.save(model.state_dict(), checkpoint_path)
            if wandb_run is not None:
                wandb_run.summary["best_val_recall"] = best_recall
                wandb_run.summary["best_epoch"]      = epoch
            print(f"  ✓ Nuevo mejor modelo guardado (recall={best_recall:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["patience"]:
                print(f"\nEarly stopping en época {epoch}. Mejor recall: {best_recall:.4f}")
                break

    # Cargar los mejores pesos antes de devolver
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"\nEntrenamiento terminado. Mejor val/recall: {best_recall:.4f}")

    # Guardar historial en disco
    metrics_path = Path("outputs/metrics/train_history.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)

    return model, history
