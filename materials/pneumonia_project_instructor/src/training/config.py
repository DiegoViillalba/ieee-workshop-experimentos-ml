# src/training/config.py
# Carga y valida la configuración desde un archivo YAML.
#
# PRINCIPIO: todo el código recibe un objeto config, nunca valores hardcodeados.
# Esto garantiza que cualquier cambio de hiperparámetro quede registrado en el YAML
# y sea rastreable vía Git + W&B.

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ── Carga ────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Carga un archivo YAML y devuelve un diccionario de configuración.

    Args:
        config_path: Ruta al archivo YAML (p.ej. 'configs/baseline.yaml').

    Returns:
        Diccionario con todos los parámetros del experimento.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan campos obligatorios.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config no encontrado: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    """Verifica que los campos críticos estén presentes."""
    required = ["seed", "model", "training", "loss", "metrics"]
    for field in required:
        if field not in cfg:
            raise ValueError(
                f"Campo obligatorio ausente en config: '{field}'. "
                f"Revisa configs/default.yaml para la estructura esperada."
            )


# ── Reproducibilidad ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Fija todas las semillas de aleatoriedad del experimento.

    Afecta: random, numpy, torch CPU, torch CUDA y cuDNN.

    Args:
        seed: Entero para inicializar los generadores. Leer desde config["seed"].

    # TODO: verificar cómo afecta el seed a los resultados.
    #       Cambia el valor en configs/baseline.yaml y vuelve a ejecutar.
    #       ¿Cambia el recall final? ¿En cuánto?
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # para setups multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[reproducibilidad] Semilla fijada: {seed}")


# ── Utilidades ───────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Devuelve el dispositivo disponible (GPU si existe, CPU si no)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dispositivo] Usando: {device}")
    if torch.cuda.is_available():
        print(f"[dispositivo] GPU: {torch.cuda.get_device_name(0)}")
    return device


def config_summary(cfg: dict) -> str:
    """Genera un resumen legible de la configuración activa."""
    lines = [
        "╔══════════════════════════════════════════╗",
        "║        Configuración del experimento      ║",
        "╚══════════════════════════════════════════╝",
        f"  Semilla          : {cfg['seed']}",
        f"  Arquitectura     : {cfg['model']['architecture']}",
        f"  Backbone frozen  : {cfg['model']['freeze_backbone']}",
        f"  Épocas           : {cfg['training']['epochs']}",
        f"  Batch size       : {cfg['training']['batch_size']}",
        f"  Learning rate    : {cfg['training']['learning_rate']}",
        f"  w_pos (pneumonia): {cfg['loss']['class_weights']['pneumonia']}",
        f"  Métrica primaria : {cfg['metrics']['primary']} ≥ {cfg['metrics']['target']}",
    ]
    return "\n".join(lines)
