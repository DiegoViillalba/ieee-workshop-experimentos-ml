# Guía del instructor — Día 2 Workshop ML
## "De la idea al experimento"

Esta guía es **solo para el instructor**. Describe exactamente qué hacer antes
de la sesión, durante la sesión, y qué archivos deben modificar los alumnos.

---

## 1. Antes de la sesión (preparación local)

### 1.1 Instalar uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verificar
uv --version   # debe mostrar uv 0.x.x
```

### 1.2 Clonar el proyecto y crear el entorno

```bash
git clone <url-del-repositorio>
cd pneumonia-classifier

# Crear entorno virtual en .venv
uv venv --python 3.10
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Instalar dependencias
uv pip install -e ".[dev]"

# Verificar que todo instaló correctamente
python -c "import torch, torchvision, wandb, sklearn; print('OK')"
ruff --version
pytest --version
```

### 1.3 Descargar el dataset

```bash
# Opción A — kaggle CLI (recomendada)
pip install kaggle
# Coloca tu kaggle.json en ~/.kaggle/kaggle.json
kaggle datasets download paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/raw/

# Opción B — descarga manual
# 1. Ve a https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
# 2. Descarga el ZIP
# 3. Descomprime en data/raw/chest_xray/

# Verificar estructura esperada:
python -c "
from src.data.dataset import dataset_summary
dataset_summary('data/raw/chest_xray')
"
# Debe imprimir los conteos de cada split:
# train:  5216 | NORMAL: 1341 (25.7%) | PNEUMONIA: 3875 (74.3%)
# val  :    16 | NORMAL:    8 (50.0%) | PNEUMONIA:    8 (50.0%)
# test :   624 | NORMAL:  234 (37.5%) | PNEUMONIA:  390 (62.5%)
```

### 1.4 Configurar Weights & Biases

```bash
# Crear cuenta gratuita en https://wandb.ai
# Luego ejecutar:
wandb login
# Pegar el API key cuando se solicite (está en https://wandb.ai/authorize)

# Verificar
python -c "import wandb; print(wandb.api.api_key[:8] + '...')"
```

### 1.5 Configurar tu entidad W&B en el YAML (opcional)

```bash
# Edita configs/baseline.yaml y cambia:
#   wandb:
#     entity: null   ← reemplaza null con tu nombre de usuario de W&B
#                      si null, usará tu cuenta personal (funciona igual)
```

### 1.6 Inicializar Git y DVC

```bash
git init
git add .
git commit -m "feat: initial experiment setup — resnet18 baseline"

dvc init
dvc add data/raw/chest_xray
git add data/raw/chest_xray.dvc .gitignore
git commit -m "data: add chest_xray dataset via DVC"
```

### 1.7 Ejecutar los tests antes de la sesión

```bash
# Verifica que el modelo y la estructura del proyecto están correctos
uv run pytest tests/ -v

# Resultado esperado:
# tests/test_model.py::test_forward_pass_frozen            PASSED
# tests/test_model.py::test_forward_pass_unfrozen          PASSED
# tests/test_model.py::test_forward_pass_batch_size_1      PASSED
# tests/test_model.py::test_frozen_backbone_trainable_params PASSED
# tests/test_model.py::test_unfrozen_all_params_trainable  PASSED
# tests/test_model.py::test_output_is_logits_not_probabilities PASSED
# tests/test_model.py::test_deterministic_output_with_same_seed PASSED
# tests/test_pipeline.py::test_baseline_config_exists      PASSED
# tests/test_pipeline.py::test_default_config_exists       PASSED
# tests/test_pipeline.py::test_dvc_yaml_exists             PASSED
```

### 1.8 Hacer un run completo de prueba (la noche anterior)

```bash
# Run 1 — baseline (freeze_backbone=true, w_pos=2.0)
uv run python scripts/train.py --config configs/baseline.yaml

# Evaluación
uv run python scripts/evaluate.py --config configs/baseline.yaml

# Verificar que se generaron los outputs
ls outputs/checkpoints/best_model.pt
ls outputs/metrics/test_metrics.json
ls outputs/metrics/confusion_matrix.png
ls outputs/metrics/roc_pr_curves.png

# Leer los resultados
python -c "
import json
with open('outputs/metrics/test_metrics.json') as f:
    d = json.load(f)
print('recall:', d['metrics']['recall_pneumonia'])
print('auc_roc:', d['metrics']['auc_roc'])
print('H1 sostenida:', d['hypothesis']['H1_sustained'])
"
```

---

## 2. Archivos que los alumnos deben modificar

Esta es la lista exacta de los **únicos archivos** que los alumnos necesitan
tocar durante la sesión. El resto del código ya funciona.

### Archivo 1 — `configs/baseline.yaml` ← EL MÁS IMPORTANTE

Este es el único archivo que los alumnos **deben** editar para generar
sus propios experimentos. Cada alumno copia el baseline y cambia **un parámetro**.

```yaml
# configs/baseline.yaml — lo que está configurado y por qué

seed: 42                     # ← los alumnos NO cambian esto en el baseline

model:
  freeze_backbone: true      # ← EJERCICIO 1: cambiar a false y observar

training:
  learning_rate: 0.0001      # ← EJERCICIO 2: cambiar a 0.001 o 0.00001
  batch_size: 32             # ← EJERCICIO 3: cambiar a 16 o 64
  epochs: 5                  # ← dejar en 5 para que termine en la sesión

loss:
  class_weights:
    pneumonia: 2.0           # ← EJERCICIO 4: cambiar a 3.0 o 1.0
```

**Cómo se crea un experimento nuevo:**

```bash
# El alumno ejecuta esto en su terminal:
cp configs/baseline.yaml configs/mi_experimento.yaml
# Edita configs/mi_experimento.yaml con su IDE o nano
python scripts/train.py --config configs/mi_experimento.yaml
```

### Archivo 2 — `src/data/dataset.py` (opcional, para alumnos avanzados)

```python
# Líneas 42-46: la función get_transforms() tiene un bloque marcado con REPLACE
# Los alumnos pueden agregar aumentación de datos aquí:

# REPLACE: agrega más aumentación para mejorar la generalización.
# Ejemplo: añadir después de ColorJitter:
transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
# o:
transforms.RandomGrayscale(p=0.1),
```

### Archivo 3 — `src/models/resnet.py` (opcional, para alumnos avanzados)

```python
# Línea 46: cambiar la arquitectura base
# REPLACE: cambia a 'resnet50' para un backbone más profundo.
# El alumno lo haría cambiando en su YAML:
#   model:
#     architecture: "resnet50"
# No necesita tocar el .py directamente.
```

### Resumen: lo que los alumnos modifican vs. lo que ya funciona

| Archivo | ¿Lo modifica el alumno? | ¿Cómo? |
|---|---|---|
| `configs/baseline.yaml` | **Sí — obligatorio** | Copiar y cambiar un parámetro |
| `configs/default.yaml` | No | Solo lectura — referencia |
| `scripts/train.py` | No | Ya funciona completo |
| `scripts/evaluate.py` | No | Ya funciona completo |
| `src/data/dataset.py` | Opcional (REPLACE) | Agregar aumentación |
| `src/models/resnet.py` | No — vía YAML | `architecture: resnet50` |
| `src/training/trainer.py` | No | Ya funciona |
| `src/evaluation/metrics.py` | No | Ya funciona |
| `dvc.yaml` | No | Ya funciona |

---

## 3. Secuencia exacta de comandos durante la sesión

### Bloque 1 (0:00–0:15) — Setup en vivo

```bash
# El instructor ejecuta esto en pantalla compartida

# Mostrar que el entorno está activo
which python         # debe apuntar a .venv/bin/python
python --version     # Python 3.10.x

# Mostrar que ruff está listo
ruff check src/ scripts/   # debe salir sin errores

# Mostrar la estructura del proyecto
find . -name "*.py" | grep -v __pycache__ | grep -v .venv | sort
```

### Bloque 2 (0:15–0:35) — Dataset e infraestructura

```bash
# Mostrar el dataset
python -c "from src.data.dataset import dataset_summary; dataset_summary('data/raw/chest_xray')"

# Mostrar los tests
uv run pytest tests/ -v
```

### Bloque 3 (0:35–1:00) — Modelo

```bash
# Mostrar el modelo y sus parámetros
python -c "
from src.models.resnet import build_model, model_summary
model = build_model(freeze_backbone=True)
model_summary(model)
print()
model2 = build_model(freeze_backbone=False)
model_summary(model2)
"
# Resultado esperado:
#   Con freeze=True:  Entrenables: 1,026 / 11,177,538
#   Con freeze=False: Entrenables: 11,177,538 / 11,177,538
```

### Bloque 4 (1:00–1:20) — Pipeline e infraestructura

```bash
# Mostrar el pipeline DVC
dvc dag

# Mostrar la configuración activa
cat configs/baseline.yaml

# Mostrar que la configuración se lee en el código
python -c "
from src.training.config import load_config, config_summary
cfg = load_config('configs/baseline.yaml')
print(config_summary(cfg))
"
```

### Bloque 5 (1:20–1:40) — Entrenamiento en vivo (Run 1)

```bash
# Run 1 — baseline exacto
uv run python scripts/train.py --config configs/baseline.yaml

# Mientras entrena, abrir el dashboard de W&B en el navegador
# y mostrar las curvas en tiempo real

# Resultado esperado por época (5 épocas, ~1 min en Colab T4):
# Época  1/5 | loss 0.4521 | val_recall 0.8461 | val_auc 0.9012 | 58.3s
# Época  2/5 | loss 0.3814 | val_recall 0.9205 | val_auc 0.9287 | 55.1s
# ...
# ✓ Nuevo mejor modelo guardado (recall=0.9385)
```

### Bloque 6 (1:40–2:00) — Evaluación + Run 2

```bash
# Evaluar Run 1
uv run python scripts/evaluate.py --config configs/baseline.yaml

# Crear Run 2 (el instructor cambia learning_rate en vivo)
cp configs/baseline.yaml configs/experiment_lr_high.yaml
# Editar experiment_lr_high.yaml: learning_rate: 0.001

uv run python scripts/train.py --config configs/experiment_lr_high.yaml

# Comparar en W&B: mostrar ambos runs en el dashboard
```

### Cierre — Pipeline completo con DVC

```bash
# Ejecutar el pipeline completo de una vez
dvc repro

# Ver las métricas
dvc metrics show
dvc params diff

# Commit del experimento
git add configs/experiment_lr_high.yaml outputs/metrics/test_metrics.json
git commit -m "exp: lr=0.001 vs baseline — comparar recall en W&B"
```

---

## 4. Solución de problemas frecuentes

### "No se encontró data/raw/chest_xray"

```bash
# Verificar la estructura del dataset
ls data/raw/chest_xray/
# Debe mostrar: train/  val/  test/

ls data/raw/chest_xray/train/
# Debe mostrar: NORMAL/  PNEUMONIA/

# Si el ZIP se descomprimió en un subdirectorio extra:
mv data/raw/chest_xray/chest_xray/* data/raw/chest_xray/
```

### "wandb: ERROR No valid API key"

```bash
wandb login
# Copiar el API key desde https://wandb.ai/authorize
# Alternativa: ejecutar con --no-wandb para la demo
uv run python scripts/train.py --config configs/baseline.yaml --no-wandb
```

### "CUDA out of memory"

```bash
# Reducir batch_size en el YAML:
# training:
#   batch_size: 16    # en lugar de 32
```

### "ModuleNotFoundError: No module named 'src'"

```bash
# Asegurarse de ejecutar desde el directorio raíz del proyecto
cd pneumonia-classifier
uv pip install -e ".[dev]"
# o usar:
PYTHONPATH=. python scripts/train.py --config configs/baseline.yaml
```

### Tests fallan con "Checkpoint no encontrado"

```bash
# Normal antes del primer entrenamiento.
# Los tests de pipeline se saltan automáticamente si no existe best_model.pt.
# Solo los tests de modelo (test_model.py) deben pasar desde el inicio.
uv run pytest tests/test_model.py -v
```

---

## 5. Configuraciones pre-generadas para los ejercicios

Estos archivos están listos para que los alumnos los usen como punto de partida:

### `configs/exercise_freeze_off.yaml`
Ejercicio: descongelar el backbone y observar el efecto.

### `configs/exercise_lr_high.yaml`
Ejercicio: subir el learning rate y observar la convergencia.

### `configs/exercise_wpos_high.yaml`
Ejercicio: subir el peso de clase positiva y observar el recall.

Ejecutarlos con:

```bash
uv run python scripts/train.py --config configs/exercise_<nombre>.yaml
uv run python scripts/evaluate.py --config configs/exercise_<nombre>.yaml
```

---

## 6. Diferencias entre versión alumno y versión instructor

| | Versión alumno | Versión instructor |
|---|---|---|
| `configs/` | baseline.yaml | + 3 configs de ejercicios pre-generados |
| `INSTRUCTOR_SETUP.md` | No existe | Este archivo |
| `scripts/instructor_demo.py` | No existe | Script de demo que ejecuta 2 runs automáticamente |
| Resultados en `outputs/` | Vacíos | Pre-generados para mostrar en clase |
| `README.md` | Instructivo para alumnos | Mismo |
