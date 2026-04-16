# tests/test_pipeline.py
# Pruebas de integración: verifican outputs generados por el pipeline.
#
# Ejecutar después de dvc repro o de los scripts manualmente:
#   pytest tests/test_pipeline.py -v
#
# TODO: extender pruebas:
#       - verificar que recall en test_metrics.json ≥ 0.90
#       - verificar que H1_sustained es True

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHECKPOINT    = Path("outputs/checkpoints/model.pkl")
TEST_METRICS  = Path("outputs/metrics/test_metrics.json")
CONF_MATRIX   = Path("outputs/metrics/confusion_matrix.png")
ROC_CURVES    = Path("outputs/metrics/roc_pr_curves.png")
BASELINE_CFG  = Path("configs/baseline.yaml")
DEFAULT_CFG   = Path("configs/default.yaml")
DVC_YAML      = Path("dvc.yaml")
PROCESSED     = Path("data/processed")


def test_configs_exist():
    """Los archivos de configuración deben existir."""
    assert BASELINE_CFG.exists(), "configs/baseline.yaml no encontrado"
    assert DEFAULT_CFG.exists(),  "configs/default.yaml no encontrado"


def test_dvc_yaml_exists():
    """dvc.yaml debe existir para poder ejecutar dvc repro."""
    assert DVC_YAML.exists()


@pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="model.pkl no encontrado — ejecuta train.py primero",
)
def test_checkpoint_exists_and_nonempty():
    """El checkpoint debe existir y no estar vacío."""
    assert CHECKPOINT.stat().st_size > 0, "model.pkl está vacío"


@pytest.mark.skipif(
    not (PROCESSED / "X_train.npy").exists(),
    reason="datos procesados no encontrados — ejecuta dvc repro primero",
)
def test_processed_data_exists():
    """Los 6 arrays del pipeline deben existir en data/processed/."""
    for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        assert (PROCESSED / f"{name}.npy").exists(), f"{name}.npy no encontrado"


@pytest.mark.skipif(
    not TEST_METRICS.exists(),
    reason="test_metrics.json no encontrado — ejecuta evaluate.py primero",
)
def test_metrics_structure():
    """test_metrics.json debe tener la estructura correcta."""
    with open(TEST_METRICS) as f:
        data = json.load(f)
    for key in ["metrics", "baseline", "config", "hypothesis"]:
        assert key in data, f"Falta clave '{key}' en test_metrics.json"
    for m in ["recall_malignant", "auc_roc", "accuracy"]:
        assert m in data["metrics"], f"Falta métrica '{m}'"
    assert 0.0 <= data["metrics"]["recall_malignant"] <= 1.0


@pytest.mark.skipif(
    not TEST_METRICS.exists(),
    reason="test_metrics.json no encontrado",
)
def test_hypothesis_fields():
    """La evaluación de hipótesis debe estar presente y ser booleana."""
    with open(TEST_METRICS) as f:
        data = json.load(f)
    h = data["hypothesis"]
    assert "H1_recall_ge_090" in h
    assert "H1_auc_gt_050"    in h
    assert "H1_sustained"     in h
    assert isinstance(h["H1_sustained"], bool)


@pytest.mark.skipif(
    not CONF_MATRIX.exists(),
    reason="confusion_matrix.png no encontrado — ejecuta evaluate.py primero",
)
def test_figures_exist():
    """Las figuras deben existir tras correr evaluate.py."""
    assert CONF_MATRIX.exists()
    assert ROC_CURVES.exists()
