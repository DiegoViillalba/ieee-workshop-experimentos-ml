# src/models/logistic.py
# Construcción del modelo de regresión logística.
#
# ¿Por qué regresión logística?
#   - Interpretable: los coeficientes tienen significado directo
#   - Rápida: converge en milisegundos en este dataset
#   - Probabilística: devuelve P(maligno) en lugar de solo 0/1
#   - Buen baseline para datos tabulares antes de modelos más complejos
#
# ¿Por qué no una red neuronal aquí?
#   Con 569 muestras y 30 features, una red neuronal sobreajustaría fácilmente.
#   La regresión logística es la elección correcta para empezar.

from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression


def build_model(cfg: dict) -> LogisticRegression:
    """
    Construye el modelo según la configuración YAML.

    Args:
        cfg: Diccionario leído desde configs/baseline.yaml.
             Se usa cfg['model'] para los parámetros del modelo.

    Returns:
        LogisticRegression configurado y listo para entrenar.

    # TODO: entender qué hace class_weight="balanced".
    #       scikit-learn calcula automáticamente:
    #       w_clase = n_total / (n_clases × n_muestras_clase)
    #       Eso hace que los errores en la clase minoritaria (maligno)
    #       pesen más en la función de pérdida → mayor recall.
    # REPLACE: cambia class_weight a None y observa qué pasa con el recall.
    # REPLACE: cambia el tipo de modelo a RandomForestClassifier y compara.
    """
    m = cfg["model"]
    # REPLACE: cambia class_weight en el YAML y re-ejecuta.
    return LogisticRegression(
        C=m.get("C", 1.0),
        class_weight=m.get("class_weight", "balanced"),
        solver=m.get("solver", "lbfgs"),
        max_iter=m.get("max_iter", 200),
        random_state=cfg["seed"],
    )


def save_model(model: LogisticRegression, path: str | Path) -> None:
    """Serializa el modelo entrenado en formato pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo guardado: {path}")


def load_model(path: str | Path) -> LogisticRegression:
    """Carga un modelo serializado desde disco."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Modelo cargado: {path}")
    return model
