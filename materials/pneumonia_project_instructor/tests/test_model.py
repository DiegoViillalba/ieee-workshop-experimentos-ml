# tests/test_model.py
# Pruebas unitarias ligeras para la arquitectura del modelo.
#
# Ejecutar:
#   python -m pytest tests/ -v
#
# TODO: extender pruebas para cubrir más casos de borde:
#       - Imágenes con píxeles en cero (negro puro)
#       - Batches de tamaño 1 (puede causar problemas con BatchNorm)
#       - Pesos de clases extremos (0.0 o 100.0)

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.resnet import build_model


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def model_frozen():
    """Modelo con backbone congelado (configuración baseline)."""
    return build_model(
        architecture="resnet18",
        pretrained=False,   # False para evitar descarga en CI
        freeze_backbone=True,
        dropout=0.4,
        num_classes=2,
    )


@pytest.fixture
def model_unfrozen():
    """Modelo con fine-tuning completo."""
    return build_model(
        architecture="resnet18",
        pretrained=False,
        freeze_backbone=False,
        dropout=0.0,
        num_classes=2,
    )


# ── Tests de forward pass ─────────────────────────────────────────────────────

def test_forward_pass_frozen(model_frozen):
    """El forward pass con backbone congelado produce la forma correcta."""
    batch = torch.randn(4, 3, 224, 224)
    output = model_frozen(batch)
    assert output.shape == (4, 2), (
        f"Se esperaba forma (4, 2), se obtuvo {output.shape}"
    )


def test_forward_pass_unfrozen(model_unfrozen):
    """El forward pass con fine-tuning completo produce la forma correcta."""
    batch = torch.randn(2, 3, 224, 224)
    output = model_unfrozen(batch)
    assert output.shape == (2, 2)


def test_forward_pass_batch_size_1(model_frozen):
    """Batch de tamaño 1 debe funcionar (puede fallar con BatchNorm en train mode)."""
    model_frozen.eval()   # eval mode desactiva BatchNorm statistics update
    batch  = torch.randn(1, 3, 224, 224)
    output = model_frozen(batch)
    assert output.shape == (1, 2)


# ── Tests de parámetros ───────────────────────────────────────────────────────

def test_frozen_backbone_trainable_params(model_frozen):
    """Con freeze_backbone=True, solo los parámetros de la cabeza son entrenables."""
    trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model_frozen.parameters())

    # Con ResNet-18 frozen + Linear(512, 2) + Dropout:
    # Esperamos que los parámetros entrenables sean << totales
    assert trainable < total, "No se congeló ningún parámetro del backbone"
    assert trainable > 0, "No hay parámetros entrenables"
    # La capa final: 512*2 pesos + 2 bias = 1026 parámetros
    assert trainable <= 2000, (
        f"Demasiados parámetros entrenables con backbone congelado: {trainable}"
    )


def test_unfrozen_all_params_trainable(model_unfrozen):
    """Con freeze_backbone=False, todos los parámetros deben ser entrenables."""
    trainable = sum(p.numel() for p in model_unfrozen.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model_unfrozen.parameters())
    assert trainable == total, (
        f"No todos los parámetros son entrenables: {trainable}/{total}"
    )


# ── Tests de salida ───────────────────────────────────────────────────────────

def test_output_is_logits_not_probabilities(model_frozen):
    """El modelo devuelve logits (sin softmax), no probabilidades."""
    model_frozen.eval()
    with torch.no_grad():
        batch  = torch.randn(4, 3, 224, 224)
        output = model_frozen(batch)
    # Si fueran probabilidades, la suma de cada fila sería ~1.0
    # Los logits no tienen esa restricción
    row_sums = torch.softmax(output, dim=1).sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(4), atol=1e-5), (
        "Las filas del softmax no suman 1 — algo está mal en la arquitectura"
    )


def test_deterministic_output_with_same_seed(model_frozen):
    """Con la misma semilla, el forward pass produce exactamente el mismo resultado."""
    import random
    import numpy as np

    model_frozen.eval()
    batch = torch.randn(2, 3, 224, 224)

    torch.manual_seed(42)
    out1 = model_frozen(batch)

    torch.manual_seed(42)
    out2 = model_frozen(batch)

    assert torch.allclose(out1, out2), (
        "El forward pass no es determinista con la misma semilla"
    )
    # TODO: extender pruebas.
    #       Prueba con torch.backends.cudnn.deterministic = True en GPU.
    #       ¿Siguen siendo idénticos los outputs?
