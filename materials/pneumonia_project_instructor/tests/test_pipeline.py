# tests/test_pipeline.py
# Pruebas de integración: verifican la existencia y estructura de los outputs
# generados por el pipeline.
#
# Ejecutar después de `dvc repro` o de los scripts manualmente:
#   python -m pytest tests/test_pipeline.py -v
#
# TODO: extender pruebas para verificar:
#       - Que el checkpoint es un state_dict válido de ResNet-18
#       - Que test_metrics.json contiene las claves esperadas
#       - Que el recall reportado es coherente con el umbral clínico

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

CHECKPOINT_PATH   = Path("outputs/checkpoints/best_model.pt")
TRAIN_HISTORY     = Path("outputs/metrics/train_history.json")
TEST_METRICS      = Path("outputs/metrics/test_metrics.json")
CONFUSION_MATRIX  = Path("outputs/metrics/confusion_matrix.png")
ROC_PR_CURVES     = Path("outputs/metrics/roc_pr_curves.png")
BASELINE_CONFIG   = Path("configs/baseline.yaml")
DEFAULT_CONFIG    = Path("configs/default.yaml")
DVC_YAML          = Path("dvc.yaml")


# ── Tests de existencia de archivos ───────────────────────────────────────────

def test_baseline_config_exists():
    """configs/baseline.yaml debe existir antes de ejecutar el pipeline."""
    assert BASELINE_CONFIG.exists(), (
        f"No se encontró {BASELINE_CONFIG}. "
        "Asegúrate de que el repositorio está correctamente inicializado."
    )


def test_default_config_exists():
    """configs/default.yaml debe existir."""
    assert DEFAULT_CONFIG.exists()


def test_dvc_yaml_exists():
    """dvc.yaml debe existir para poder ejecutar dvc repro."""
    assert DVC_YAML.exists()


@pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason="Checkpoint no encontrado — ejecuta train.py primero"
)
def test_checkpoint_exists():
    """outputs/checkpoints/best_model.pt debe existir tras el entrenamiento."""
    assert CHECKPOINT_PATH.exists(), (
        f"Checkpoint no encontrado en {CHECKPOINT_PATH}. "
        "Ejecuta: python scripts/train.py --config configs/baseline.yaml"
    )
    assert CHECKPOINT_PATH.stat().st_size > 0, "El checkpoint está vacío"


@pytest.mark.skipif(
    not TEST_METRICS.exists(),
    reason="test_metrics.json no encontrado — ejecuta evaluate.py primero"
)
def test_test_metrics_structure():
    """outputs/metrics/test_metrics.json debe tener la estructura esperada."""
    with open(TEST_METRICS) as f:
        data = json.load(f)

    required_keys = ["metrics", "config", "hypothesis"]
    for key in required_keys:
        assert key in data, f"Falta la clave '{key}' en test_metrics.json"

    required_metrics = [
        "recall_pneumonia", "auc_roc", "accuracy",
        "precision_pneumonia", "f1_pneumonia"
    ]
    for m in required_metrics:
        assert m in data["metrics"], f"Falta la métrica '{m}' en test_metrics.json"

    # Verificar que recall es un número en [0, 1]
    recall = data["metrics"]["recall_pneumonia"]
    assert 0.0 <= recall <= 1.0, f"Recall fuera de rango: {recall}"


@pytest.mark.skipif(
    not TEST_METRICS.exists(),
    reason="test_metrics.json no encontrado"
)
def test_hypothesis_fields_present():
    """test_metrics.json debe incluir la evaluación de la hipótesis."""
    with open(TEST_METRICS) as f:
        data = json.load(f)
    h = data["hypothesis"]
    assert "H1_recall_ge_090" in h
    assert "H1_auc_gt_050" in h
    assert "H1_sustained" in h
    assert isinstance(h["H1_sustained"], bool)


@pytest.mark.skipif(
    not TRAIN_HISTORY.exists(),
    reason="train_history.json no encontrado — ejecuta train.py primero"
)
def test_train_history_has_recall():
    """El historial de entrenamiento debe incluir val/recall por época."""
    with open(TRAIN_HISTORY) as f:
        history = json.load(f)
    assert "val/recall" in history, "No se encontró val/recall en el historial"
    assert len(history["val/recall"]) > 0, "Historial de val/recall vacío"


@pytest.mark.skipif(
    not CONFUSION_MATRIX.exists(),
    reason="confusion_matrix.png no encontrado — ejecuta evaluate.py primero"
)
def test_figures_exist():
    """Las figuras de evaluación deben existir tras correr evaluate.py."""
    assert CONFUSION_MATRIX.exists()
    assert ROC_PR_CURVES.exists()
    # TODO: extender pruebas.
    #       Verifica que las imágenes no están corruptas (son PNG válidos).
    #       Puedes usar PIL: Image.open(path).verify()
