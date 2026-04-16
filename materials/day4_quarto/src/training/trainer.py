# src/training/trainer.py
# Lógica de entrenamiento y evaluación en val set.

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import recall_score, roc_auc_score


def train_model(
    model:   BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> BaseEstimator:
    """
    Entrena el modelo sobre los datos de entrenamiento.

    # TODO: ¿qué ocurre internamente durante model.fit()?
    #       La regresión logística minimiza la cross-entropy ponderada
    #       usando el algoritmo L-BFGS (un método cuasi-Newton).
    #       Con este dataset converge en < 100 iteraciones.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_split(
    model:    BaseEstimator,
    X:        np.ndarray,
    y:        np.ndarray,
    split:    str = "val",
    pos_label: int = 0,   # 0 = maligno = clase positiva de interés
) -> dict[str, float]:
    """
    Evalúa el modelo en un split dado.

    Args:
        model:     Modelo entrenado.
        X:         Features del split.
        y:         Etiquetas del split.
        split:     Nombre del split para las claves del diccionario.
        pos_label: Etiqueta de la clase positiva (0=maligno).

    Returns:
        Diccionario con accuracy, recall y AUC-ROC del split.

    # TODO: identificar dónde se calcula recall.
    #       recall = TP / (TP + FN)
    #       En diagnóstico médico, un FN (cáncer no detectado) es mucho
    #       más grave que un FP (alarma falsa que genera una biopsia adicional).
    #       Por eso recall es nuestra métrica primaria, no accuracy.
    """
    preds = model.predict(X)
    # AUC-ROC: pasamos P(benigno=1) — sklearn por defecto trata columna 1
    # como la clase positiva para roc_auc_score. El AUC es simétrico:
    # discriminar maligno de benigno es lo mismo en ambas direcciones.
    probs = model.predict_proba(X)[:, 1]   # P(benigno) para roc_auc_score

    return {
        f"{split}/accuracy": float(np.mean(preds == y)),
        f"{split}/recall":   recall_score(y, preds, pos_label=pos_label),
        f"{split}/auc":      roc_auc_score(y, probs),
    }
