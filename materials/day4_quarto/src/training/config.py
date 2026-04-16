# src/training/config.py
# Carga de configuración YAML y utilidades de reproducibilidad.

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Carga un archivo YAML y valida los campos obligatorios.

    Args:
        config_path: Ruta al archivo YAML (p.ej. 'configs/baseline.yaml').

    Returns:
        Diccionario con todos los parámetros del experimento.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config no encontrado: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    required = ["seed", "data", "model", "metrics"]
    for field in required:
        if field not in cfg:
            raise ValueError(
                f"Campo obligatorio ausente en config: '{field}'. "
                f"Revisa configs/default.yaml para la estructura esperada."
            )
    return cfg


def set_seed(seed: int = 42) -> None:
    """
    Fija semillas en random y numpy para reproducibilidad.

    # TODO: verificar cómo afecta el seed al resultado.
    #       Cambia seed: 42 → 99 en el YAML y vuelve a ejecutar.
    #       ¿Cambia el recall? ¿Cuánto? ¿Qué dice eso sobre la estabilidad
    #       del modelo y del split?
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    print(f"[reproducibilidad] Semilla fijada: {seed}")


def config_summary(cfg: dict) -> str:
    """Genera un resumen legible de la configuración activa."""
    lines = [
        "╔══════════════════════════════════════╗",
        "║     Configuración del experimento     ║",
        "╚══════════════════════════════════════╝",
        f"  Semilla        : {cfg['seed']}",
        f"  Modelo         : {cfg['model']['type']}",
        f"  class_weight   : {cfg['model']['class_weight']}",
        f"  C              : {cfg['model'].get('C', 1.0)}",
        f"  max_iter       : {cfg['model']['max_iter']}",
        f"  test_size      : {cfg['data']['test_size']}",
        f"  Métrica primaria: {cfg['metrics']['primary']} ≥ {cfg['metrics']['target']}",
    ]
    return "\n".join(lines)
