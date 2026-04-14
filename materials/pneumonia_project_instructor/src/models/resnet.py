# src/models/resnet.py
# Definición de la arquitectura ResNet-18 con cabeza de clasificación personalizada.
#
# Variable independiente del experimento baseline:
#   ResNet-18 preentrenada en ImageNet + fine-tuning de la capa final
#   con pérdida ponderada (w_pos = 2.0).

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(
    architecture: str  = "resnet18",
    pretrained:   bool = True,
    freeze_backbone: bool = True,
    dropout:      float = 0.4,
    num_classes:  int   = 2,
) -> nn.Module:
    """
    Construye ResNet-18 con cabeza de clasificación personalizada.

    Flujo de datos:
        imagen (3×224×224)
        → ResNet-18 backbone (congelado si freeze_backbone=True)
        → GlobalAveragePooling (automático en ResNet)
        → Dropout(p=dropout)
        → Linear(512, num_classes)
        → logits (sin softmax — la pérdida la aplica internamente)

    Args:
        architecture:    Nombre de la arquitectura ('resnet18', 'resnet50').
        pretrained:      Cargar pesos preentrenados en ImageNet.
        freeze_backbone: Si True, congela todos los pesos excepto la capa final.
        dropout:         Probabilidad de dropout en la cabeza de clasificación.
        num_classes:     Número de clases de salida (2 para NORMAL/PNEUMONIA).

    Returns:
        Modelo PyTorch listo para enviar a dispositivo y entrenar.

    # TODO: entender qué capas se están fine-tuneando.
    #       Ejecuta model_summary() para ver los parámetros entrenables.
    #       Con freeze_backbone=True: ¿cuántos parámetros tienen requires_grad=True?
    # REPLACE: cambia freeze_backbone a False para fine-tuning completo.
    #          ¿Mejora el recall? ¿Cuánto más tarda?
    # REPLACE: cambia a 'resnet50' para un backbone más profundo.
    #          Considera que necesitarás más datos o más regularización.
    """
    weights_enum = {
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
    }
    builder_fn = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
    }

    if architecture not in builder_fn:
        raise ValueError(
            f"Arquitectura '{architecture}' no soportada. "
            f"Opciones: {list(builder_fn.keys())}"
        )

    weights = weights_enum[architecture] if pretrained else None
    model   = builder_fn[architecture](weights=weights)

    # Congelar backbone si se solicita
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Reemplazar la capa fully-connected final
    in_features = model.fc.in_features   # 512 para ResNet-18
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    # Los parámetros de model.fc siempre tienen requires_grad=True
    # (nn.Sequential crea nuevos parámetros que no fueron congelados)

    return model


def model_summary(model: nn.Module) -> None:
    """Imprime un resumen del modelo: parámetros totales y entrenables."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print("Resumen del modelo:")
    print(f"  Parámetros totales     : {total:>12,}")
    print(f"  Parámetros entrenables : {trainable:>12,}")
    print(f"  Parámetros congelados  : {frozen:>12,}")
    print(f"  % entrenables          : {100*trainable/total:>11.2f}%")


def load_checkpoint(
    model:           nn.Module,
    checkpoint_path: str,
    device:          torch.device,
) -> nn.Module:
    """
    Carga los pesos de un checkpoint guardado.

    Args:
        model:           Instancia del modelo (misma arquitectura que el checkpoint).
        checkpoint_path: Ruta al archivo .pt generado por el entrenamiento.
        device:          Dispositivo destino (cpu o cuda).

    Returns:
        Modelo con los pesos del checkpoint cargados.
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Checkpoint cargado: {checkpoint_path}")
    return model
