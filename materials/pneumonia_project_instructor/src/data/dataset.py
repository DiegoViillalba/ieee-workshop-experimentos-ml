# src/data/dataset.py
# Carga, preprocesamiento y DataLoaders para el dataset de Kermany et al. (2018).
#
# El dataset Chest X-Ray Images tiene la siguiente estructura en disco:
#   data/raw/chest_xray/
#     train/NORMAL/     train/PNEUMONIA/
#     val/NORMAL/       val/PNEUMONIA/
#     test/NORMAL/      test/PNEUMONIA/
#
# Descarga: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


# ── Transformaciones ──────────────────────────────────────────────────────────

def get_transforms(
    split: Literal["train", "val", "test"],
    image_size: int = 224,
) -> transforms.Compose:
    """
    Devuelve las transformaciones de imagen para cada split.

    Train: aumentación de datos para regularización.
    Val / Test: solo resize + normalización (sin aumentación).

    Args:
        split: 'train', 'val' o 'test'.
        image_size: Tamaño al que se redimensionan todas las imágenes (px).

    Returns:
        Pipeline de transformaciones de torchvision.

    # TODO: ¿qué impacto tiene cada transformación en el entrenamiento?
    #       Prueba quitar RandomHorizontalFlip. ¿Cambia el recall?
    # REPLACE: agrega más aumentación para mejorar la generalización.
    #          Opciones: RandomRotation, ColorJitter, GaussianBlur.
    """
    # Estadísticas de normalización de ImageNet
    # (usadas porque el backbone fue preentrenado en ImageNet)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    normalize     = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloader(
    data_dir: str | Path,
    split: Literal["train", "val", "test"],
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    balance: bool = False,
    seed: int = 42,
) -> DataLoader:
    """
    Construye un DataLoader para el split indicado.

    Args:
        data_dir: Directorio raíz del dataset (contiene train/, val/, test/).
        split: 'train', 'val' o 'test'.
        batch_size: Número de imágenes por batch.
        image_size: Tamaño de imagen tras el preprocesamiento.
        num_workers: Subprocesos para carga paralela de datos.
        balance: Si True, usa muestreo ponderado para balancear clases en train.
        seed: Semilla para el generador del DataLoader.

    Returns:
        DataLoader configurado y listo para iterar.

    # TODO: ¿qué impacto tiene batch_size?
    #       Batch pequeño = más actualizaciones por época pero más ruido en el gradiente.
    #       Batch grande = gradientes más estables pero menos actualizaciones.
    """
    dataset = datasets.ImageFolder(
        root=Path(data_dir) / split,
        transform=get_transforms(split, image_size),
    )

    sampler = None
    shuffle = split == "train"

    if balance and split == "train":
        # WeightedRandomSampler: equilibra las clases durante el entrenamiento
        # Es una alternativa a class_weights en la función de pérdida.
        counts  = np.bincount([s[1] for s in dataset.samples])
        weights = 1.0 / counts
        sample_weights = torch.tensor(
            [weights[s[1]] for s in dataset.samples], dtype=torch.float
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False   # sampler y shuffle son mutuamente excluyentes

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


# ── Utilidades ────────────────────────────────────────────────────────────────

def dataset_summary(data_dir: str | Path) -> None:
    """Imprime un resumen del dataset (tamaño y proporción de clases por split)."""
    data_dir = Path(data_dir)
    print("Dataset: Chest X-Ray (Kermany et al., 2018)")
    print(f"Ubicación: {data_dir.resolve()}")
    print()
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"  {split:6s}: [no encontrado]")
            continue
        ds = datasets.ImageFolder(root=split_dir)
        counts = np.bincount([s[1] for s in ds.samples])
        total  = len(ds)
        print(f"  {split:6s}: {total:5d} imágenes", end="")
        for cls, n in zip(ds.classes, counts):
            print(f"  |  {cls}: {n} ({100*n/total:.1f}%)", end="")
        print()
