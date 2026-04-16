# src/data/dataset.py
# Carga y división del dataset breast cancer de sklearn.
#
# Usamos sklearn.datasets.load_breast_cancer porque:
#   - Es tabular (no requiere GPU ni transformaciones de imagen)
#   - Es binario: maligno (0) vs benigno (1)
#   - Es pequeño (~570 muestras): se entrena en < 1 segundo en CPU
#   - No requiere descarga — está incluido en scikit-learn
#   - Tiene desbalance real (63/37) que justifica class_weight

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_split(
    test_size:      float = 0.20,
    val_size:       float = 0.15,
    seed:           int   = 42,
    save_dir:       str | Path | None = "data/processed",
) -> tuple[np.ndarray, ...]:
    """
    Carga breast_cancer, escala y divide en train/val/test.

    Estratificado: misma proporción maligno/benigno en los tres splits.

    Args:
        test_size:  Fracción de datos para test (evaluación final).
        val_size:   Fracción de train para validación (selección de modelo).
        seed:       Semilla para reproducibilidad.
        save_dir:   Si se indica, guarda arrays .npy en ese directorio.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test

    # TODO: ¿qué hace StandardScaler y por qué es necesario?
    #       La regresión logística usa distancias en el espacio de features.
    #       Sin escalar, features con rangos grandes (p.ej. área del tumor en mm²)
    #       dominarían sobre features pequeñas. Escalar las pone en el mismo rango.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    # y: 0 = maligno, 1 = benigno

    # División estratificada train+val / test
    X_trv, X_test, y_trv, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    # División estratificada train / val
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trv, y_trv,
        test_size=val_frac,
        stratify=y_trv,
        random_state=seed,
    )

    # Escalado — fit SOLO en train, transform en val y test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "X_train.npy", X_train)
        np.save(save_dir / "X_val.npy",   X_val)
        np.save(save_dir / "X_test.npy",  X_test)
        np.save(save_dir / "y_train.npy", y_train)
        np.save(save_dir / "y_val.npy",   y_val)
        np.save(save_dir / "y_test.npy",  y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def dataset_summary() -> None:
    """Imprime un resumen del dataset original antes del split."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    n_total  = len(y)
    n_mal    = (y == 0).sum()
    n_ben    = (y == 1).sum()
    print("Dataset: Breast Cancer Wisconsin (sklearn)")
    print(f"  Muestras totales : {n_total}")
    print(f"  Maligno  (y=0)   : {n_mal}  ({100*n_mal/n_total:.1f}%)")
    print(f"  Benigno  (y=1)   : {n_ben}  ({100*n_ben/n_total:.1f}%)")
    print(f"  Features         : {X.shape[1]}")
