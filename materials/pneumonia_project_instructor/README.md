# Clasificador de neumonía en rayos X
### Workshop ML — Día 2: De la pregunta científica al experimento ejecutable

---

## Pregunta científica

> ¿Un modelo ResNet-18 con fine-tuning sobre ImageNet y pérdida ponderada
> (w_pos = 2.0) alcanza **recall ≥ 0.90** para la clase PNEUMONIA en el
> test set de Kermany et al. (2018), con AUC-ROC mayor al de un clasificador
> de mayoría?

**Hipótesis H₁:** `recall_pneumonia ≥ 0.90` **y** `auc_roc > 0.50` en el test set.

---

## Instalación con uv (recomendado)

Este proyecto usa **[uv](https://docs.astral.sh/uv/)** como gestor de paquetes
y **[ruff](https://docs.astral.sh/ruff/)** como linter y formatter.

```bash
# 1. Instalar uv (una sola vez por máquina)
curl -LsSf https://astral.sh/uv/install.sh | sh
# En Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clonar el repositorio
git clone <url-del-repo>
cd pneumonia-classifier

# 3. Crear entorno virtual .venv en el directorio del proyecto
uv venv --python 3.10
# Activa el entorno:
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 4. Instalar dependencias (modo editable + herramientas de desarrollo)
uv pip install -e ".[dev]"

# 5. Verificar la instalación
python -c "import torch; print(torch.__version__)"
ruff --version

# 6. Descargar dataset
# https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
# Colocar en: data/raw/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/
```

### Alternativa: instalación con conda

```bash
# Si prefieres Conda en lugar de uv:
conda create -n pneumonia-clf python=3.10 -y
conda activate pneumonia-clf
pip install -r requirements.txt    # requirements.txt generado por uv export
```

> **¿Por qué uv y no conda?** Ver la sección [uv vs. conda](#uv-vs-conda) al final.

---

## Formato y linting con ruff

```bash
# Verificar estilo del código (linting)
ruff check src/ scripts/ tests/

# Formatear automáticamente
ruff format src/ scripts/ tests/

# Verificar y corregir en un solo paso
ruff check --fix src/ scripts/ tests/
```

Ruff reemplaza flake8, isort y black en un solo binario escrito en Rust.
Es entre 10× y 100× más rápido que las herramientas equivalentes en Python.

---

## Reproducir el experimento

```bash
# Opción A: Pipeline completo con DVC (recomendado)
dvc repro

# Opción B: Scripts individuales
python scripts/train.py    --config configs/baseline.yaml
python scripts/evaluate.py --config configs/baseline.yaml

# Opción C: Sin W&B (para pruebas rápidas)
python scripts/train.py    --config configs/baseline.yaml --no-wandb
python scripts/evaluate.py --config configs/baseline.yaml --no-wandb
```

---

## Estructura del proyecto

```
pneumonia-classifier/
├── .venv/                        # Entorno virtual local (uv, ignorado por Git)
├── configs/
│   ├── default.yaml              # Parámetros base del experimento
│   └── baseline.yaml             # Configuración del experimento baseline
├── data/
│   ├── raw/chest_xray/           # Dataset original (versionado por DVC)
│   └── processed/                # Datos preprocesados
├── src/
│   ├── data/dataset.py           # Carga de datos y DataLoaders
│   ├── models/resnet.py          # Arquitectura ResNet-18
│   ├── training/
│   │   ├── config.py             # Carga de configuración y set_seed()
│   │   └── trainer.py            # Loop de entrenamiento
│   └── evaluation/metrics.py     # Métricas y figuras de evaluación
├── scripts/
│   ├── train.py                  # Entrenamiento
│   └── evaluate.py               # Evaluación en test set
├── outputs/
│   ├── checkpoints/              # best_model.pt
│   └── metrics/                  # Métricas JSON y figuras
├── tests/
│   ├── test_model.py             # Forward pass y parámetros
│   └── test_pipeline.py          # Existencia de outputs
├── quarto/
│   └── slides.qmd                # Slides del instructor (RevealJS)
├── pyproject.toml                # Dependencias (uv) + config ruff + pytest
├── requirements.txt              # Generado con: uv pip freeze > requirements.txt
├── dvc.yaml                      # Pipeline reproducible
└── README.md
```

---

## Ejecutar tests

```bash
# Con pytest (configurado en pyproject.toml)
pytest

# Con uv (sin necesidad de activar el entorno manualmente)
uv run pytest
```

---

## Resultados del baseline

| Métrica               | Valor     | Criterio      |
|-----------------------|-----------|---------------|
| Recall PNEUMONIA      | —         | ≥ 0.90        |
| AUC-ROC               | —         | > 0.50        |
| Precision PNEUMONIA   | —         | —             |
| Accuracy              | —         | —             |

> Los resultados se llenarán tras ejecutar el pipeline con `dvc repro`.

---

## Generar nuevos experimentos

```bash
# 1. Copia el config baseline
cp configs/baseline.yaml configs/experiment_01.yaml

# 2. Modifica UN parámetro (p.ej. learning_rate)
# REPLACE: edita el campo que quieres experimentar

# 3. Formatea el código antes de hacer commit
ruff format src/ scripts/

# 4. Ejecuta
python scripts/train.py --config configs/experiment_01.yaml

# 5. Compara en W&B
# Abre wandb.ai y compara las curvas val/recall de ambos runs
```

---

## Control de versiones

```bash
# Inicializar Git y DVC
git init
dvc init

# Primer commit
git add .
git commit -m "feat: initial experiment setup — resnet18 baseline"

# TODO: usar commits como historial del experimento.
#       Cada configuración significativa merece un commit.
#       Convención: "feat: experimento X — descripción del cambio"

# Versionado del dataset con DVC
dvc add data/raw/chest_xray
git add data/raw/chest_xray.dvc .gitignore
git commit -m "data: add chest_xray dataset via DVC"
```

---

## uv vs. conda

| Criterio | uv + .venv | conda |
|---|---|---|
| **Velocidad de instalación** | 10–100× más rápido | Más lento (solver SAT) |
| **Tamaño en disco** | ~50 MB por entorno | ~500 MB–2 GB por entorno |
| **Reproducibilidad** | `uv.lock` exacto (hashes) | `environment.yml` menos preciso |
| **Paquetes no-Python** | ✗ No gestiona CUDA, MKL, etc. | ✓ Gestiona binarios nativos |
| **Integración con PyPI** | ✓ Nativa y completa | Mezcla conda-forge + PyPI |
| **Activación del entorno** | Manual o `uv run` | `conda activate` |
| **CI/CD y Docker** | ✓ Ideal (sin conda overhead) | Añade complejidad |
| **Soporte de torch + CUDA** | Requiere wheel correcto de PyPI | Más sencillo con `pytorch` channel |
| **Curva de aprendizaje** | Baja (similar a pip) | Media (ecosistema propio) |

**Regla práctica:**

- Usa **uv** si tu proyecto vive en PyPI, trabajas en CI/CD o Docker, y controlas la versión de CUDA de otra forma (p.ej. en Colab o con drivers del sistema).
- Usa **conda** si dependes de binarios nativos no-Python (CUDA toolkit, MKL, GDAL, libsndfile) o si tu equipo ya trabaja con Anaconda/Miniconda.

En este workshop usamos **uv** porque el entorno de Colab ya tiene CUDA, y uv nos da reproducibilidad exacta con el lockfile.

---

## Referencia

Kermany, D.S. et al. (2018). Identifying Medical Diagnoses and Treatable Diseases
by Image-Based Deep Learning. *Cell*, 172(5), 1122–1131.
