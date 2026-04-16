# Guía del instructor — Día 2 Workshop ML
## "De la idea al experimento"

> Documento de uso **exclusivo del instructor**.
> Los alumnos no reciben este archivo.

---

## Índice

1. [Objetivo de la sesión](#1-objetivo-de-la-sesión)
2. [Flujo sugerido — timeline de 2 horas](#2-flujo-sugerido--timeline-de-2-horas)
3. [Qué explicar en cada bloque](#3-qué-explicar-en-cada-bloque)
4. [Preguntas clave para los alumnos](#4-preguntas-clave-para-los-alumnos)
5. [Lugares donde intervenir en el código](#5-lugares-donde-intervenir-en-el-código)
6. [Errores comunes](#6-errores-comunes)
7. [Ejercicio sugerido](#7-ejercicio-sugerido)
8. [Criterios de éxito en clase](#8-criterios-de-éxito-en-clase)
9. [Preparación antes de la sesión](#9-preparación-antes-de-la-sesión)
10. [Verificación rápida pre-sesión](#10-verificación-rápida-pre-sesión)

---

## 1. Objetivo de la sesión

Al terminar la sesión, cada alumno debe ser capaz de:

1. Traducir una pregunta científica en código ejecutable con configuración en YAML.
2. Ejecutar un pipeline reproducible con DVC (3 etapas: prepare → train → evaluate).
3. Comparar al menos 2 runs en W&B y explicar la diferencia en términos de la hipótesis.
4. Interpretar recall, AUC-ROC y baseline en el contexto del problema clínico.
5. Sostener o rechazar H₁ con evidencia numérica.

**Lo que NO es objetivo de esta sesión:** optimizar métricas al máximo, aprender todos los modelos de sklearn, ni entender todos los detalles matemáticos de la regresión logística. El foco es el proceso experimental, no el modelo.

---

## 2. Flujo sugerido — timeline de 2 horas

### `0:00–0:15` — Recapitulación y contexto (15 min)

- Conectar con el Día 1: la pregunta científica ya está definida, hoy la convertimos en código.
- Mostrar la estructura del proyecto (`find . -name "*.py" | sort`).
- Mostrar `configs/baseline.yaml` y leer los campos en voz alta conectándolos con la pregunta científica.
- Mostrar que no hay valores hardcodeados en ningún `.py`.

**Mensaje central:** el YAML es la traducción directa de la pregunta científica. Cada campo responde a una decisión tomada en el Día 1.

### `0:15–0:35` — Dataset y reproducibilidad (20 min)

- Ejecutar `python -c "from src.data.dataset import dataset_summary; dataset_summary()"` y leer los conteos en voz alta.
- Señalar el desbalance (37% maligno) y preguntar: ¿qué pasaría si no hacemos nada con este desbalance?
- Mostrar `src/data/dataset.py`: la función `load_and_split` con `stratify=y`.
- Mostrar `set_seed()` en `src/training/config.py` y explicar las 3 líneas.

**Pregunta para la clase:** ¿por qué el test set se evalúa solo al final y no durante el ajuste de hiperparámetros?

### `0:35–0:55` — Modelo y función de pérdida (20 min)

- Mostrar `src/models/logistic.py`: la función `build_model()`.
- Explicar `class_weight="balanced"` con la fórmula: el peso de maligno = n / (2 × n_maligno).
- Ejecutar en vivo el modelo con `class_weight=None` (comentar el YAML) y mostrar el recall.
- Restaurar `class_weight="balanced"` y mostrar la diferencia.

**Mensaje central:** `class_weight` no cambia el modelo, cambia qué errores el modelo considera más graves durante el entrenamiento.

### `0:55–1:20` — Pipeline, DVC y W&B (25 min)

- Mostrar `dvc.yaml` y el grafo con `dvc dag`.
- Ejecutar `dvc repro` en vivo. Mostrar que tarda < 5 segundos.
- Abrir el dashboard de W&B y mostrar el Run 1.
- Preguntar: si cambio `C` en el YAML y vuelvo a ejecutar `dvc repro`, ¿qué etapas se re-ejecutan?
- Crear `configs/no_balance.yaml` con `class_weight: null` y ejecutar el Run 2.
- Mostrar los 2 runs comparados en W&B.

### `1:20–1:45` — Métricas e hipótesis (25 min)

- Ejecutar `python scripts/evaluate.py --config configs/baseline.yaml`.
- Leer el reporte impreso en pantalla, campo por campo.
- Preguntar: ¿el modelo cumple H₁? ¿Cómo lo sabemos?
- Mostrar `outputs/metrics/test_metrics.json` y leer `hypothesis.H1_sustained`.
- Mostrar las figuras: matriz de confusión y curva ROC.

### `1:45–2:00` — Cierre y reflexión (15 min)

- Preguntar: si H₁ no se sostuvo, ¿cuál sería el siguiente experimento?
- Conectar con el Día 3: reproducibilidad y análisis de errores.
- Verificar que todos tienen `dvc repro` corriendo y ≥ 2 runs en W&B.

---

## 3. Qué explicar en cada bloque

### Dataset (`src/data/dataset.py`)

El dataset se elige por sus **propiedades pedagógicas**, no porque sea el más difícil:

- **Tabular**: no requiere GPU ni transformaciones de imagen. El pipeline corre en segundos.
- **Desbalance real (37/63)**: motiva el uso de `class_weight` y explica por qué accuracy engaña.
- **Incluido en sklearn**: no hay pasos de descarga ni de preparación. El alumno puede enfocarse en el experimento.
- **Pequeño**: permite iterar rápido y comparar muchos runs en una sola sesión.

Señalar que la misma estructura de código (YAML → split → modelo → métricas → DVC → W&B) se aplicará al clasificador de rayos X del Día 2 original, que sí usa GPU.

### Modelo (`src/models/logistic.py`)

Tres preguntas que el instructor debe poder responder:

1. **¿Por qué regresión logística y no random forest o SVM?**
   Porque es el modelo más simple que puede responder la pregunta. Si el modelo más simple ya cumple H₁, no necesitamos complejidad adicional. Si no la cumple, tenemos un baseline claro para comparar.

2. **¿Qué hace exactamente `class_weight="balanced"`?**
   sklearn calcula automáticamente `w_i = n_total / (n_clases × n_i)` para cada clase. Para maligno (212 muestras): `w = 569 / (2 × 212) = 1.34`. Para benigno (357 muestras): `w = 569 / (2 × 357) = 0.80`. Esto hace que cada mujer con cáncer "valga" 1.67× más en la función de pérdida que una muestra benigna.

3. **¿Qué es `C` y qué cambia si lo modificamos?**
   `C` es el inverso de la regularización. `C=1.0` es el valor por defecto. `C=0.01` añade mucha regularización (coeficientes más pequeños, modelo más simple). `C=100.0` casi no regulariza (coeficientes pueden ser muy grandes, riesgo de sobreajuste).

### Métricas (`src/evaluation/metrics.py`)

Explicar las tres métricas en el orden correcto:

1. **Baseline primero**: mostrar que accuracy = 62.6% no significa nada si recall = 0.0.
2. **Recall**: TP / (TP + FN). De todos los cánceres reales, ¿cuántos detectó el modelo?
3. **AUC-ROC**: independiente del umbral. Mide si el modelo puede separar maligno de benigno mejor que el azar. AUC = 0.5 → azar puro. AUC = 1.0 → separación perfecta.

El criterio H₁ exige las dos condiciones simultáneamente precisamente para evitar soluciones triviales: un modelo que predice siempre maligno tiene recall = 1.0 pero AUC = 0.5.

### Baseline (`DummyClassifier` en `src/evaluation/metrics.py`)

El baseline no es una crítica al modelo — es el punto de referencia mínimo. Cualquier modelo que no supere al baseline no está aportando valor. El instructor debe mostrar explícitamente que el baseline tiene recall = 0.0 antes de mostrar los resultados del modelo, para que la comparación sea significativa.

---

## 4. Preguntas clave para los alumnos

Estas preguntas funcionan bien como disparadores de discusión. No requieren respuesta inmediata — el objetivo es que los alumnos piensen antes de continuar.

### Sobre el modelo

> **¿Por qué este modelo?**
> Pedir a los alumnos que defiendan la elección antes de explicarla. Respuesta esperada: es interpretable, rápido, y suficiente para el tamaño del dataset.

> **¿Qué pasaría si usaras un modelo más complejo (random forest, red neuronal)?**
> Con 569 muestras, probablemente sobreajustaría. La regresión logística tiene mucho menos riesgo de sobreajuste.

### Sobre recall

> **¿Qué significa recall = 0.95 en este contexto clínico?**
> De cada 100 mujeres con cáncer de mama, el modelo detectó 95 y dejó pasar 5. ¿Es eso suficiente? ¿Cuándo sería inaceptable?

> **¿Preferirías un modelo con recall = 0.99 y precision = 0.60, o recall = 0.90 y precision = 0.95?**
> Respuesta: depende del contexto. Para screening inicial (primera detección), recall alto. Para diagnóstico final (decidir cirugía), precision alta.

### Sobre el baseline

> **¿Por qué comparamos contra un clasificador que siempre predice "benigno"?**
> Porque si no lo hacemos, no sabemos si el modelo aprendió algo. Un modelo que solo memoriza la distribución de clases es inútil, aunque tenga accuracy alta.

> **¿Cuándo diríamos que el modelo "no aprendió nada"?**
> Cuando sus métricas son iguales o peores que el baseline en todas las dimensiones.

### Sobre el experimento

> **Si el modelo no cumple H₁, ¿cuál sería el siguiente experimento?**
> Respuesta pedagógica: cambiar UNA variable (`class_weight`, `C`, o `seed`) y volver a ejecutar. No cambiar dos cosas al mismo tiempo.

> **¿Por qué el test set se evalúa solo una vez al final?**
> Si lo usamos para ajustar hiperparámetros, las métricas de test quedan infladas. Ya no son una estimación independiente del error de generalización.

---

## 5. Lugares donde intervenir en el código

Cada `# TODO` y `# REPLACE` es un punto donde el instructor puede pausar y preguntar antes de que el alumno lo lea solo.

### Intervenciones de alta prioridad

| Archivo | Línea | Intervención |
|---|---|---|
| `configs/baseline.yaml` | `class_weight` | Preguntar: ¿qué hace este campo? Antes de ejecutar. |
| `src/models/logistic.py` | `# TODO: entender class_weight` | Explicar la fórmula de pesos. |
| `src/evaluation/metrics.py` | `# TODO: entender baseline` | Mostrar que baseline recall = 0.0 antes de mostrar el modelo. |
| `scripts/train.py` | `# TODO: ¿por qué dividir en train/val/test?` | Conectar con el principio de no usar test para selección. |

### Intervenciones opcionales (para grupos avanzados)

| Archivo | Línea | Intervención |
|---|---|---|
| `src/data/dataset.py` | `# TODO: ¿qué hace StandardScaler?` | Explicar por qué escalar importa para regresión logística. |
| `src/training/trainer.py` | `# TODO: ¿qué ocurre durante model.fit()?` | Explicar L-BFGS y convergencia. |
| `src/evaluation/metrics.py` | `# TODO: interpretar AUC-ROC` | Explicar la interpretación probabilística: P(maligno > benigno). |

---

## 6. Errores comunes

### Error 1: Confundir accuracy con recall

**Síntoma:** el alumno ve accuracy = 96% y declara que el modelo funciona bien sin mirar recall.

**Corrección:** mostrar el baseline. Accuracy = 62.6% con recall = 0.0. Si el alumno acepta accuracy sin recall, está aceptando un modelo que no detecta ningún cáncer.

**Prevención:** mostrar el baseline **antes** de mostrar los resultados del modelo. Así el alumno ya tiene el contexto cuando ve accuracy alta.

### Error 2: No entender el baseline

**Síntoma:** el alumno pregunta "¿para qué sirve el clasificador de mayoría?" o dice que el baseline es "demasiado fácil de superar".

**Corrección:** ejecutar el baseline en vivo y mostrar recall = 0.0. Preguntar: "¿usarías este modelo en un hospital?" La respuesta obvia es no — y eso hace que el baseline sea relevante.

### Error 3: No modificar la configuración

**Síntoma:** el alumno ejecuta el pipeline con `baseline.yaml` dos veces sin cambiar nada y reporta "dos runs iguales en W&B".

**Corrección:** recordar la regla del Día 1: cambiar UNA variable independiente por experimento. El segundo run debe tener exactamente un campo diferente. Revisar el YAML antes de ejecutar.

### Error 4: Cambiar dos parámetros al mismo tiempo

**Síntoma:** el alumno cambia `class_weight: null` y `C: 0.01` en el mismo run y no puede atribuir el cambio a ninguna de las dos variables.

**Corrección:** este es el aliasing del Día 1. Mostrar que con dos cambios simultáneos no se puede saber cuál fue la causa de la diferencia en recall.

### Error 5: Usar el test set para ajustar hiperparámetros

**Síntoma:** el alumno ejecuta `evaluate.py` varias veces durante el ajuste de `C` y reporta el mejor resultado de test como el resultado final.

**Corrección:** el test set se ejecuta UNA SOLA VEZ. Cualquier ajuste de hiperparámetros debe hacerse mirando `val/recall` en W&B, no `test/recall_malignant` en el JSON final.

### Error 6: `ModuleNotFoundError: No module named 'src'`

**Síntoma:** error al ejecutar `scripts/train.py` sin activar el entorno.

**Corrección:**
```bash
source .venv/bin/activate
# o
uv run python scripts/train.py --config configs/baseline.yaml
```

---

## 7. Ejercicio sugerido

### Ejercicio principal (todos los alumnos)

**Duración:** 15 minutos  
**Objetivo:** generar un segundo run y compararlo en W&B.

**Instrucciones para el alumno:**

1. Copia el baseline:
   ```bash
   cp configs/baseline.yaml configs/sin_balanceo.yaml
   ```

2. Edita `configs/sin_balanceo.yaml` y cambia **solo este campo**:
   ```yaml
   model:
     class_weight: null    # era "balanced"
   ```

3. Ejecuta:
   ```bash
   python scripts/train.py --config configs/sin_balanceo.yaml
   python scripts/evaluate.py --config configs/sin_balanceo.yaml
   ```

4. Abre W&B y compara los dos runs. Responde:
   - ¿Cuál tiene mayor recall de maligno?
   - ¿Cuál tiene mayor accuracy?
   - ¿Cuál sostienes H₁? ¿Por qué?

**Respuesta esperada:**

| Métrica | `class_weight="balanced"` | `class_weight=null` |
|---|---|---|
| Recall maligno | ~0.97 ✓ | ~0.85 ✗ |
| AUC-ROC | ~0.99 ✓ | ~0.99 ✓ |
| Accuracy | ~0.96 | ~0.97 |
| **H₁** | **SOSTENIDA** | **NO RECHAZADA** |

### Ejercicio avanzado (grupos que terminan antes)

**Objetivo:** explorar el trade-off recall vs. precision ajustando `C`.

1. Copia el baseline y cambia `C: 0.01` (más regularización).
2. Ejecuta y compara recall en W&B.
3. Repite con `C: 100.0`.
4. Preguntar: ¿en qué rango de C se sostiene H₁? ¿Es el modelo sensible a este parámetro?

---

## 8. Criterios de éxito en clase

Al finalizar la sesión, cada alumno debe poder verificar:

```bash
# 1. Pipeline corre de extremo a extremo
dvc repro && echo "✓ pipeline OK"

# 2. Modelo generado
ls outputs/checkpoints/model.pkl && echo "✓ modelo OK"

# 3. Métricas evaluadas con hipótesis
python -c "
import json
d = json.load(open('outputs/metrics/test_metrics.json'))
print('recall:', round(d['metrics']['recall_malignant'], 4))
print('H1_sustained:', d['hypothesis']['H1_sustained'])
" && echo "✓ métricas OK"

# 4. Tests pasan
pytest tests/test_model.py -q && echo "✓ tests OK"

# 5. ≥ 2 runs en W&B
# Verificar visualmente en wandb.ai
```

---

## 9. Preparación antes de la sesión

### Instalar entorno

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd breast-cancer-classifier
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Verificar que el proyecto corre en < 5 segundos

```bash
time python scripts/train.py --config configs/baseline.yaml --no-wandb
# Debe terminar en < 5 segundos
```

### Configurar W&B

```bash
wandb login
# Pegar API key desde https://wandb.ai/authorize
```

### Ejecutar un run completo de prueba

```bash
python scripts/train.py    --config configs/baseline.yaml
python scripts/evaluate.py --config configs/baseline.yaml

python -c "
import json
d = json.load(open('outputs/metrics/test_metrics.json'))
print('recall:', d['metrics']['recall_malignant'])
print('H1:', d['hypothesis']['H1_sustained'])
"
```

Resultado esperado: `recall ≈ 0.97`, `H1: True`.

---

## 10. Verificación rápida pre-sesión

Ejecutar esto 10 minutos antes de que lleguen los alumnos:

```bash
# 1. Entorno y paquete
python -c "from src.models.logistic import build_model; print('✓ src OK')"

# 2. Pipeline
dvc repro --no-commit 2>&1 | tail -3 && echo "✓ dvc OK"

# 3. Métricas
python -c "
import json; d = json.load(open('outputs/metrics/test_metrics.json'))
r = d['metrics']['recall_malignant']
assert r >= 0.90, f'recall {r} < 0.90'
print(f'✓ recall={r:.4f} H1={d[\"hypothesis\"][\"H1_sustained\"]}')"

# 4. Tests
pytest tests/test_model.py -q && echo "✓ tests OK"

# 5. W&B
python -c "import wandb; wandb.login(anonymous='never'); print('✓ W&B OK')"
```

Si las 5 líneas terminan con `✓`, la sesión puede comenzar.

---

## 11. Día 4 — Comunicación de proyectos de ML con Quarto

### Objetivo de la sesión

Entender Quarto como sistema unificado de documentación que permite generar
HTML, PDF y presentaciones desde un único source, conectando el pipeline
del Días 1–3 con su comunicación a distintas audiencias.

Al terminar la sesión, cada alumno debe ser capaz de:

1. Entender la arquitectura de un proyecto Quarto (`.qmd`, `_quarto.yml`, `sections/`).
2. Ejecutar `quarto render` y `quarto preview`.
3. Agregar contenido a un documento existente.
4. Entender cómo controlar la ejecución de código (`echo`, `eval`, `cache`).
5. Publicar el sitio en GitHub Pages.

---

### Timeline sugerido (2 horas)

| Tiempo | Bloque | Meta docente |
|---|---|---|
| `0:00–0:15` | Problema central | Por qué comunicar es parte del proceso científico |
| `0:15–0:35` | Quarto como sistema | Un source → múltiples salidas |
| `0:35–0:55` | Estructura del paper | Secciones, `include`, referencias |
| `0:55–1:20` | Reproducibilidad | Código + resultados en el mismo archivo |
| `1:20–1:45` | Actividad hands-on | Construir documento base |
| `1:45–2:00` | Publicación y cierre | GitHub Pages + ciclo completo |

---

### Puntos clave para explicar

**Un source, múltiples formatos:**

El principio central del Día 4. Mostrar en vivo que `quarto render paper.qmd`
genera HTML, y `quarto render paper.qmd --to pdf` genera PDF — desde el
mismo archivo. Contrastar con el flujo típico (Word + PowerPoint + Jupyter).

**Reproducibilidad del documento:**

Cuando `test_metrics.json` cambia (porque el pipeline se re-ejecutó con
nuevos parámetros), `quarto render` actualiza automáticamente las tablas
y figuras del paper. El documento nunca está desincronizado con los resultados.

**Modularidad con `{{< include >}}`:**

Cada sección del paper vive en su propio archivo. Distintos colaboradores
pueden trabajar en `sections/results.qmd` y `sections/discussion.qmd`
sin conflictos de merge en Git.

**Temas y estilos:**

Hoy se usan temas default. El CSS personalizado se añade en iteraciones
posteriores. Enfatizar que el contenido y el diseño están desacoplados —
cambiar de `theme: default` a `theme: cosmo` no requiere tocar ningún párrafo.

---

### Qué mostrar en vivo

```bash
# 1. Ver la estructura del proyecto
find quarto/ -type f | sort

# 2. Preview en tiempo real
cd quarto/
quarto preview paper.qmd

# 3. Render HTML
quarto render paper.qmd --to html

# 4. Render PDF (requiere tinytex)
quarto install tinytex   # solo la primera vez
quarto render paper.qmd --to pdf

# 5. Preview de las slides de alumnos
quarto preview presentation.qmd

# 6. Ver el output en docs/
ls docs/
```

---

### Errores comunes

**"No encuentro el archivo JSON de métricas"**

El paper.qmd busca `../outputs/metrics/test_metrics.json`. Esto supone que
`quarto render` se ejecuta desde el directorio `quarto/`. Si se ejecuta
desde la raíz del proyecto, ajustar las rutas. Alternativa: usar rutas
absolutas o la variable `here::here()`.

Corrección:
```bash
cd quarto/
quarto render paper.qmd
```

**"LaTeX no instalado"**

El PDF requiere LaTeX. Instalar con:
```bash
quarto install tinytex
```

**"El preview no actualiza"**

Asegurarse de guardar el archivo `.qmd` antes de que el servidor de preview
lo detecte. Si no actualiza, reiniciar con `Ctrl+C` y `quarto preview` de nuevo.

**"`include` no funciona"**

Las rutas en `{{< include sections/intro.qmd >}}` son relativas al archivo
que hace el include, no al directorio de trabajo. Si el archivo está en
`quarto/paper.qmd` y hace `{{< include sections/intro.qmd >}}`, busca
`quarto/sections/intro.qmd`. Correcto.

---

### Criterios de éxito Día 4

Al finalizar la sesión:

```bash
# 1. El sitio renderiza sin errores
cd quarto/ && quarto render && echo "✓ render OK"

# 2. La presentación de alumnos funciona
quarto render presentation.qmd && echo "✓ presentation OK"

# 3. Las slides del instructor funcionan
quarto render slides_instructor.qmd && echo "✓ instructor slides OK"

# 4. El paper tiene las 8 secciones
grep -c "^# " paper.qmd  # debe ser >= 8

# 5. Los outputs están en docs/
ls docs/index.html && echo "✓ GitHub Pages ready"
```

---

### Ejercicio principal (actividad hands-on)

**Duración:** 20 minutos

Los alumnos construyen su propio documento Quarto desde cero.

**Paso a paso:**

```bash
# 1. Crear proyecto
mkdir mi_experimento && cd mi_experimento
quarto create project website --no-open

# 2. Editar index.qmd — agregar pregunta científica y una tabla

# 3. Ejecutar
quarto preview

# 4. Cambiar tema en _quarto.yml
# theme: default → theme: cosmo

# 5. Agregar un bloque de código Python con resultado del experimento
```

**Lo que el alumno debe observar:**

- El mismo `.qmd` genera distintos formatos con un solo comando
- Cambiar el tema no requiere tocar el contenido
- El código se ejecuta y el resultado aparece en el documento automáticamente
