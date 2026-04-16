# tests/test_model.py
# Pruebas unitarias para el modelo y el pipeline de datos.
#
# Ejecutar: pytest tests/ -v
#
# TODO: extender pruebas:
#       - test con class_weight=None
#       - test con distintos valores de C
#       - test que las probabilidades sumen 1

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset    import load_and_split
from src.models.logistic import build_model, load_model, save_model
from src.training.config import load_config
from src.training.trainer import evaluate_split, train_model


@pytest.fixture(scope="module")
def sample_data():
    """Carga y divide el dataset para todos los tests del módulo."""
    return load_and_split(test_size=0.2, val_size=0.15, seed=42, save_dir=None)


@pytest.fixture(scope="module")
def trained_model(sample_data):
    """Entrena un modelo una vez y lo reutiliza en todos los tests."""
    X_train, _, _, y_train, _, _ = sample_data
    cfg = {
        "seed": 42,
        "model": {"type": "logistic_regression", "class_weight": "balanced",
                  "solver": "lbfgs", "max_iter": 200, "C": 1.0},
    }
    model = build_model(cfg)
    return train_model(model, X_train, y_train)


# ── Dataset ───────────────────────────────────────────────────────────────────

def test_data_shapes(sample_data):
    """Los splits tienen el número correcto de muestras y features."""
    X_train, X_val, X_test, y_train, y_val, y_test = sample_data
    assert X_train.shape[1] == 30, "Breast cancer tiene 30 features"
    assert len(X_train) == len(y_train)
    assert len(X_val)   == len(y_val)
    assert len(X_test)  == len(y_test)
    assert len(X_train) > len(X_test), "Train debe ser mayor que test"


def test_stratification(sample_data):
    """La proporción de clases es similar en todos los splits (stratify)."""
    _, _, _, y_train, y_val, y_test = sample_data
    for y, name in [(y_train, "train"), (y_val, "val"), (y_test, "test")]:
        prop = (y == 0).mean()
        assert 0.30 <= prop <= 0.50, (
            f"Proporción de maligno en {name} fuera de rango: {prop:.3f}"
        )


# ── Modelo ───────────────────────────────────────────────────────────────────

def test_model_predicts(trained_model, sample_data):
    """El modelo entrenado produce predicciones con la forma correcta."""
    _, _, X_test, _, _, y_test = sample_data
    preds = trained_model.predict(X_test)
    assert preds.shape == y_test.shape
    assert set(preds).issubset({0, 1}), "Las predicciones deben ser 0 o 1"


def test_model_probabilities(trained_model, sample_data):
    """predict_proba devuelve probabilidades válidas (suman 1 por fila)."""
    _, _, X_test, _, _, _ = sample_data
    probs = trained_model.predict_proba(X_test)
    assert probs.shape[1] == 2
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_recall_meets_target(trained_model, sample_data):
    """El modelo baseline debe alcanzar recall ≥ 0.90 en maligno."""
    _, _, X_test, _, _, y_test = sample_data
    metrics = evaluate_split(trained_model, X_test, y_test,
                             split="test", pos_label=0)
    recall = metrics["test/recall"]
    assert recall >= 0.90, (
        f"Recall {recall:.4f} < 0.90 — el baseline no cumple el criterio clínico"
    )


def test_model_serialization(trained_model, tmp_path):
    """El modelo se guarda y recarga correctamente con pickle."""
    path = tmp_path / "model.pkl"
    save_model(trained_model, path)
    loaded = load_model(path)
    assert path.exists()
    assert path.stat().st_size > 0

    # Los dos modelos producen las mismas predicciones
    X_dummy = np.random.randn(10, 30)
    np.testing.assert_array_equal(
        trained_model.predict(X_dummy),
        loaded.predict(X_dummy),
    )
