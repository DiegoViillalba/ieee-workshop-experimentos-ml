# Clasificador de cáncer de mama
### Workshop ML — Día 2: De la pregunta científica al experimento ejecutable

---

## Pregunta científica

> ¿Un modelo de regresión logística con ponderación de clases alcanza
> **recall ≥ 0.90** para la clase maligna y supera a un clasificador de
> mayoría en AUC-ROC?

**H₁:** `recall_malignant ≥ 0.90` **y** `auc_roc > 0.50` en el test set.

---

## Instalación

```bash
# Con uv (recomendado)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev]"

# O con pip
pip install -r requirements.txt
```

Sin GPU. Sin descarga de datos. Ejecución en < 5 segundos.

---

## Ejecutar el experimento

```bash
# Pipeline completo con DVC
dvc repro

# Scripts individuales
python scripts/train.py    --config configs/baseline.yaml
python scripts/evaluate.py --config configs/baseline.yaml

# Sin W&B
python scripts/train.py    --config configs/baseline.yaml --no-wandb
python scripts/evaluate.py --config configs/baseline.yaml --no-wandb
```

---

## El único archivo que modifican los alumnos

`configs/baseline.yaml` — copiar y cambiar **un parámetro**:

```bash
cp configs/baseline.yaml configs/experiment_01.yaml
# Editar experiment_01.yaml
python scripts/train.py --config configs/experiment_01.yaml
```

| Ejercicio | Campo | Cambio sugerido |
|---|---|---|
| 1 | `model.class_weight` | `"balanced"` → `null` |
| 2 | `model.C` | `1.0` → `0.01` o `100.0` |
| 3 | `seed` | `42` → `99` |

---

## Estructura

```
breast-cancer-classifier/
├── src/
│   ├── data/dataset.py           # carga y split
│   ├── models/logistic.py        # build_model(), save/load
│   ├── training/config.py        # load_config(), set_seed()
│   ├── training/trainer.py       # train_model(), evaluate_split()
│   └── evaluation/metrics.py     # métricas, figuras, JSON
├── scripts/
│   ├── train.py                  # entrenamiento
│   └── evaluate.py               # evaluación en test set
├── configs/
│   ├── default.yaml              # referencia de todos los campos
│   └── baseline.yaml             # el que modifican los alumnos
├── outputs/
│   ├── checkpoints/model.pkl     # generado por train.py
│   └── metrics/                  # JSON + figuras
├── tests/
├── quarto/slides.qmd             # slides del instructor
├── docs/instructor_guide.md      # guía completa del instructor
├── pyproject.toml
└── dvc.yaml
```

---

## Tests

```bash
pytest tests/test_model.py -v    # siempre deben pasar
pytest tests/ -v                 # completo (pipeline tests se saltan si faltan outputs)
```

---

## Referencia

Dataset: Breast Cancer Wisconsin — `sklearn.datasets.load_breast_cancer()`
