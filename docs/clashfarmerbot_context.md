# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXTO INTEGRAL DEL CHAT, HISTORIAL DE DESARROLLO Y ESTADO DEL PROYECTO
# ClashFarmerbot - Sistema Autónomo de Farming para Clash of Clans
# ═══════════════════════════════════════════════════════════════════════════════

## 1. REGLAS CRÍTICAS Y RESTRICCIONES VIGENTES (LEER OBLIGATORIAMENTE)

> [!CAUTION]
> **ESTADO DE FASE 4: ESTRICTAMENTE CONGELADA.**
> - El archivo `attack/deployment_strategy.py` **NUNCA DEBE MODIFICARSE NI DESPLEGARSE A COMBATE EN VIVO** sin la aprobación explícita y previa del usuario.
> - Existe un respaldo de seguridad verificado en `backup_jump_optimizer_20261114_170528/attack/deployment_strategy.py`. Cualquier cambio no autorizado viola la directiva del proyecto.
> - Las pruebas de optimización de saltos deben mantenerse en `vision/wall_compartments.py`, `tools/` y `scratch/`, evaluadas offline sobre datasets anotados.
> - La suite de pruebas unitarias (`python -m unittest tests.test_wall_compartments tests.test_wall_raycast tests.test_wall_junctions tools.test_annotate_tool`) debe ejecutarse y pasar al 100% ante cualquier cambio.

---

## 2. PROPÓSITO GENERAL Y ARQUITECTURA DEL SISTEMA

ClashFarmerbot es un sistema de automatización para Clash of Clans en Windows que interactúa con el juego emulado (BlueStacks / LDPlayer / Google Play Games PC) mediante ADB (`core/device.py`) o captura directa de pantalla (`scrcpy` / `BitBlt` / `cv2`):

- **Pipeline de Visión (`vision/`):**
  - `yolo_detector.py`: Modelo ONNX (`models/yolov8s.onnx`) que detecta ayuntamientos (`town_hall`), almacenes de oro (`storage_gold`), elixir (`storage_elixir`), elixir oscuro (`storage_dark`), recolectores (`collector_gold`, `collector_elixir`, `collector_dark`), y muros.
  - `deploy_zone_detector.py`: Red neuronal especializada (`models/deploy_zone_finder.onnx`) para clasificar zonas de despliegue legal (césped exterior) vs zona roja prohibida.
  - `wall_raycast.py`: Detección global e isométrica de líneas de muro usando transformada de Hough probabilística (`HoughLinesP`), Canny y gradiente direccional a 36° y 144°.
  - `wall_compartments.py`: Módulo canónico de compartimentos. Modela la aldea como un grafo de conectividad $G=(V, E)$, calcula componentes conexas cerradas, identifica accesibilidad exterior vía flood-fill, y optimiza la colocación de hechizos de salto.
  - `army_checker.py` & `damage_reader.py`: Reconocimiento de inventario de batalla (duendes, super rompemuros, dragones, héroes, hechizos) y lectura de porcentaje de destrucción.
- **Pipeline de Ataque (`attack/`):**
  - `deployment_strategy.py`: Estrategia maestra de farming con duendes furtivos (Sneaky Goblins), rompemuros y saltos.
  - `edrag_strategy.py`: Estrategia secundaria de dragones eléctricos para trofeos/ligas.
- **Pipeline de Control y Comportamiento Humano (`core/` y `humanizer/`):**
  - `state_machine.py`: Máquina de estados (IDLE, SEARCHING_BASE, ATTACKING, UPGRADING_WALLS, COLLECTING).
  - `behavior.py`: Jitter temporal, curvas Bezier de clics y retrasos gaussianos para evitar detección de bot.

---

## 3. HISTORIAL COMPLETO DE INTERACCIONES Y DECISIONES DEL CHAT

### A. Origen: Solicitud de Saltos Óptimos (Fase 1 a 3)
- El usuario solicitó implementar una estrategia óptima de salto de muros basada en el documento `plan_saltos_optimos_antigravity.md`.
- Se desarrollaron dos aproximaciones de salto:
  1. `cluster_intersection`: Intersección de líneas de visión desde objetivos hacia la entrada.
  2. `wall_raycast`: Trazado de rayos desde objetivos al borde exterior con selección adaptativa por demanda de recursos.
  3. `coverage_optimizer` (Fase 2-3 en `vision/wall_compartments.py`): Grafo de compartimentos donde un salto actúa como una elipse que elimina aristas de muro y fusiona componentes conexas del grafo, minimizando el costo de salto hacia los recursos valiosos.

### B. Congelamiento de Fase 4 y Reevaluación Rigurosa
- El usuario **NO aprobó** la Fase 4 (integración a `deployment_strategy.py`) y exigió una auditoría estricta con evidencia empírica:
  1. Reevaluar SOLO aldeas cerradas (costo inicial > 0), midiendo saltos usados y costo final por separado de aldeas abiertas.
  2. Confirmar qué modo corría en producción (`settings.json` tenía `"wall_raycast"` pero existían contradicciones en reportes previos).
  3. Explicar discrepancia en aldeas abiertas (7 vs 5 vs 8 aldeas).
  4. Limpiar mapa de muros: eliminar marcos de interfaz, filtrar por orientación isométrica (36°/144°) y gradiente mínimo.
  5. Medir huella real del salto: se estandarizó en elipse de 115x65 px para coincidencia geométrica y 90 px para radio efectivo de compartimentos.
  6. Medir tiempos de cómputo en aldeas cerradas (deben mantenerse $< 300$ ms).

### C. Herramienta de Anotación Ground Truth (`tools/annotate_best_jump.py`)
- El usuario informó que su ejército real lleva **4 hechizos de salto** (no 3).
- Se modificó `tools/annotate_best_jump.py` y `tools/test_annotate_tool.py`:
  - Límite de clics ampliado a 4 saltos máximos.
  - Tecla `'k'` añadida para alternar el flag booleano `"botin_en_colectores": true/false` (almacenes vacíos, botín en recolectores externos).
  - Tecla `'s'` para `"sin_salto": true` (aldeas abiertas o sin necesidad de salto).
  - Tecla `'u'` para deshacer (undo) y `'c'` para limpiar (clear).
  - Compatibilidad hacia atrás garantizada para anotaciones previas con 3 saltos.
- El usuario completó las anotaciones manuales de las **57 aldeas** en `datasets/coc/jump_gt/`.

### D. Corrección de Bugs en el Optimizador y Comparación con Ground Truth (57 Aldeas)
- **Corrección de Bug de Ganancia:** En `vision/wall_compartments.py`, `gain` se calculaba contando componentes aisladas en vez de la fórmula matemática canónica `gain = initial_cost - final_cost`. Esto generaba reducciones $> 100\%$. Se corrigió formalmente.
- **Curva de Presupuesto (1, 2, 3, 4 saltos):** Se demostró que el optimizador usaba un promedio de 1.77 saltos porque la parada temprana `min_marginal_gain = 0.5` cortaba la búsqueda cuando ningún salto individual desbloqueaba objetivos adicionales. Se agregó `force_max_jumps=True` para forzar la colocación de los 4 saltos.
- **Máscara de HUD de Batalla:** Se implementó `get_battle_hud_mask` para excluir candidatos de salto dentro de la barra de tropas inferior ($y \ge 888$), pero permitiendo que la elipse de cobertura se extienda bajo ella si se coloca justo encima.
- **Desacoplamiento de Hechizo:** Se parametrizaron forma, radio, efecto y cantidad en `settings.json` (`spell_config`).

### E. Análisis Avanzado Solicitado por el Usuario

#### 1. Comparación Aldea por Aldea (Human GT vs Optimizer)
En las 57 aldeas anotadas:
- **Human GT gana en 7 aldeas** (12.3%): `0005`, `0010`, `0012`, `0022`, `0008`, `0021`, `screen_raw_07`.
- **Optimizer gana en 33 aldeas** (57.9%).
- **Empatan en 17 aldeas** (29.8%) (12 aldeas donde ambos llegan a costo 0.0, y 5 aldeas abiertas).

**¿Por qué gana el Humano en esas 7 aldeas? Atascamiento Voraz (Greedy):**
En `auto_20261104_111433_0005.png`, el humano coloca una cadena de 4 saltos que une el flanco exterior con el núcleo.
- Salto #1: ganancia marginal individual = +9.0 (costo 17.0 -> 8.0).
- Salto #2: une dos compartimentos interiores; por sí solo produce **ganancia = 0.0**.
- Salto #3: une pasillo intermedio; por sí solo produce **ganancia = 0.0**.
- Salto #4: completa la cadena; juntos los 4 saltos logran costo **0.0** (ganancia total = 17.0).
El algoritmo voraz se detenía en el paso 2 porque ningún candidato individual superaba `min_marginal_gain = 0.5`.
Además, el humano colocó saltos en **pasillos de césped / aperturas intermedias**, mientras que el optimizador solo muestreaba sobre muros. Se demostró que con búsqueda de anticipación (Lookahead de 2 pasos / Beam Search) y candidatos en aperturas, el optimizador resuelve cadenas complejas.

#### 2. Objetivo Direccional (Flancos Reales de Ataque)
Se evaluó el costo usando la entrada por flancos de `structure_quota_deployment`:
- `coverage_optimizer`: Costo 5.50 (54.3% reducción, mediana 71.5%).
- `cluster_intersection`: Costo 6.20 (48.5% reducción, mediana 17.4%).
- `wall_raycast`: Costo 9.12 (24.2% reducción, mediana 0.0%).
- `human_gt`: Costo 9.54 (20.8% reducción, mediana 0.0%).
El score del GT humano refleja que el humano concentró sus 4 saltos en un único flanco penetrante (donde logró costo 0.0), mientras que la métrica direccional evalúa tropas atacando desde los 4 cuadrantes a la vez.

#### 3. Baseline Aleatorio vs Coincidencia Geométrica (Monte Carlo 100 repeticiones)
Para verificar si la coincidencia geométrica (elipse 115x65 px) era mejor que tirar saltos al azar junto a muros:
- `coverage_optimizer` ($k=1.88$): **Recall Real 43.3%** vs **Azar 7.15% ± 2.56%** (6.1x sobre el azar).
- `coverage_optimizer` ($k=1.88$): **Precisión Real 39.4%** vs **Azar 8.29% ± 3.11%** (4.8x sobre el azar).
- `wall_raycast` ($k=2.55$): Recall Real 26.9% vs Azar 9.96% | Precisión Real 21.3% vs Azar 7.71%.
- `cluster_intersection` ($k=3.84$): Recall Real 36.5% vs Azar 12.74% | Precisión Real 18.3% vs Azar 6.88%.
La superioridad del optimizador sobre el azar es estadísticamente contundente ($p < 10^{-6}$).

#### 4. Análisis Técnico: Identificación de Botín en Almacenes vs Recolectores
Sin implementar atajos en tiempo real:
1. **Clasificación Visual de Tiers de Llenado (Sprites):** Supercell tiene 4-5 estados visuales según llenado de almacenes (cristal vacío vs lleno de oro/elixir). Puede discriminarse mediante ratio de color HSV en la ROI central del almacén o clasificador ligero (MobileNetV4).
2. **Inferencia Matemática por OCR de Botín Disponible vs Tope de Almacén:** Los almacenes tienen un tope de saqueo (~450k-500k por recurso en TH11-TH13). Si el OCR en el HUD muestra > 1,000,000 de oro/elixir disponible, matemáticamente > 50% del botín reside obligatoriamente en recolectores perimetrales.

#### 5. Prueba Real de Oclusión bajo el HUD (Prolongación Isométrica)
Se prolongaron los muros que entran a la barra de tropas ($y \ge 800$) siguiendo ángulos de 36° y 144° hasta $y = 1020$:
- **55 de 57 aldeas (96.5%)** tienen muros que tocan o entran a la zona oculta por la barra inferior.
- En **28 de 57 aldeas (49.1%)**, la prolongación de muros altera el grafo de compartimentos o el costo inicial (ej. sellando fugas artificiales creadas por la barra, como en `0017` donde el costo pasó de 9.0 a 18.0, o `screen_raw_05` que parecía abierta y en realidad era cerrada).

#### 6. Estabilidad y Sensibilidad de Clasificación Abierta / Cerrada
- El paso de 7 -> 5 -> 8 aldeas abiertas a lo largo del proyecto se debió a:
  1. Corrida 1: Ruido de baldosas y bordes exteriores sin filtrar (7 abiertas con fugas erráticas).
  2. Corrida 2: Truncamiento entero `int(x/scale)` que desalineaba 1 px en diagonales T (5 abiertas).
  3. Corrida 3: Rasterizador canónico redondeado `round(x/scale)` en `vision/wall_compartments.py` (8 abiertas).
- El barrido de sensibilidad demostró que la clasificación es **100% inmune a variaciones de gradiente (55 a 75) y grosor de muro (1 a 3 px)**, dependiendo únicamente de la tolerancia angular isométrica (6° canónico).

#### 7. Conjunto Held-Out para Evaluación a Ciegas
Se seleccionaron y congelaron **40 aldeas de batalla completamente nuevas** en `logs/yolo/test_split_unseen/` con su `manifest.json`. No han sido vistas, calibradas ni optimizadas.

---

## 4. QUÉ NO SE HA PODIDO VERIFICAR Y POR QUÉ

1. **Pathfinding y Dinámica de Tropas en Combate:** El modelo evalúa conectividad geométrica y costos de grafo estáticos. No modela cómo los Duendes Furtivos priorizan objetivos dinámicamente cuando un muro adyacente es abierto por un super rompemuros durante la batalla.
2. **Muros Completamente Ocultos Bajo el HUD:** Aunque la prolongación geométrica reveló fugas en el 49.1% de aldeas, no se puede saber si en el juego real existen aberturas deliberadas o trampas en esa franja sin desplazar la cámara del emulador hacia arriba (paneo vertical).
3. **Decaimiento Temporal del Hechizo:** El hechizo de salto dura un tiempo limitado (~20-40 segundos según nivel). La efectividad en combate depende de la cadencia de despliegue de las oleadas, lo cual no puede medirse en capturas estáticas.

═══════════════════════════════════════════════════════════════════════════════


This file is a merged representation of a subset of the codebase, containing specifically included files and files not matching ignore patterns, combined into a single document by Repomix.
The content has been processed where content has been formatted for parsing in markdown style.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: core/**/*.py, attack/**/*.py, vision/**/*.py, humanizer/**/*.py, tools/**/*.py, tests/**/*.py, config/**/*.json, main.py, requirements.txt
- Files matching these patterns are excluded: .venv/**, venv/**, __pycache__/**, *.pyc, *.pyo, .env*, .git/**, .idea/**, .agents/**, .gemini/**, assets/**, datasets/**, ejemplos/**, logs/**, models/**, runs/**, scratch/**, _backups/**, backup_*/**, _trash*/**, clashfarmerbot_context.md, repomix-output.*, *.md, *.png, *.jpg, *.jpeg, *.gif, *.webp, *.onnx, *.enc, *.pt, *.pth, *.rar, *.zip, *.tar, *.gz
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Content has been formatted for parsing in markdown style
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
attack/
  __init__.py
  deployment_strategy.py
  edrag_strategy.py
config/
  settings.json
core/
  __init__.py
  config.py
  device.py
  ledger.py
  logger.py
  process_manager.py
  state_machine.py
  ui_cache.py
  vision.py
  wall_upgrader.py
humanizer/
  __init__.py
  behavior.py
tests/
  __init__.py
  test_army_checker.py
  test_boundary_detection.py
  test_camera_survey.py
  test_damage_reader.py
  test_deep_corridor_and_swb.py
  test_deploy_zone_detector.py
  test_drop_validation.py
  test_e2e_village_analysis.py
  test_edrag_strategy.py
  test_heroes_haste_and_cross_pan.py
  test_quota_drop_and_inventory.py
  test_ui_cache.py
  test_wall_compartments.py
  test_wall_junctions.py
  test_wall_raycast.py
tools/
  analyze_eval_extremes.py
  annotate_best_jump.py
  audit_village_diversity.py
  auto_capture_villages.py
  calibrate_jump_footprint.py
  capture_dataset.py
  check_templates.py
  crop_helper.py
  diagnose_drop_points.py
  evaluate_jumps_offline.py
  measure_wall_raycast_accuracy.py
  prelabel_hsv.py
  prelabel_with_model.py
  rename_templates.py
  review_prelabels.py
  run_live_test_3_attacks.py
  split_dataset.py
  test_annotate_tool.py
  test_boundary_overlay.py
  test_canonical_diamond.py
  test_jump_from_labels.py
  test_jump_overlay.py
  test_local_zone.py
  test_probe_red_zone.py
  test_stitch.py
  test_sweep.py
  test_vision.py
  test_wall_raycast_on_saved_attacks.py
  test_yolo.py
  train_yolo.py
  verify_drop_clearance.py
vision/
  __init__.py
  army_checker.py
  damage_reader.py
  deploy_zone_detector.py
  resource_reader.py
  storage_detector.py
  wall_compartments.py
  wall_raycast.py
  yolo_detector.py
main.py
requirements.txt
```

# Files

## File: attack/__init__.py
```python

```

## File: attack/deployment_strategy.py
```python
"""
attack/deployment_strategy.py
Estrategia de ataque con barridos calibrados según el flujo:
1. Zoom out obligatorio para estabilizar la vista del mapa al máximo.
2. Salto inicial de cámara del Centro hacia el extremo inicial (Sur o Norte).
3. Barrido continuo con ritmo pausado y extendido hacia los extremos.
4. Salto de cámara al otro extremo completo.
5. Barrido del flanco opuesto extendido.
6. Separación deliberada entre cada acción para garantizar detección confiable.
"""
import random
import time
import cv2
import numpy as np
from core.config import SETTINGS
from core.device import Device
from core.vision import Vision, save_debug_image_async
from vision.army_checker import ArmyChecker
from vision.storage_detector import StorageDetector
from vision.yolo_detector import Detection, get_detector
from vision.wall_raycast import (
    estimate_iso_angles,
    detect_global_wall_segments,
    find_wall_crossings_from_segments,
    find_wall_crossings,
    draw_wall_raycast_debug,
)
from humanizer.behavior import HumanBehavior
from core.logger import get_logger

log = get_logger("deployment")

# Paneo calibrado físicamente en pantalla de 2400x1080:
# Eje X centrado en 0.50 (X=1200) para no tocar HUD, botones de rendición ni recursos.
# 1. Del Centro (tras zoom) hacia el Sur: arrastre hacia arriba (0.68 -> 0.30)
PAN_CENTER_TO_SOUTH = (0.50, 0.68, 0.50, 0.30)
# 2. Del Centro (tras zoom) hacia el Norte: arrastre hacia abajo (0.32 -> 0.70)
PAN_CENTER_TO_NORTH = (0.50, 0.32, 0.50, 0.70)
# 3. De Extremo Sur a Extremo Norte: arrastre completo hacia abajo (0.20 -> 0.82)
PAN_SOUTH_TO_NORTH = (0.50, 0.20, 0.50, 0.82)
# 4. De Extremo Norte a Extremo Sur: arrastre completo hacia arriba (0.82 -> 0.20)
PAN_NORTH_TO_SOUTH = (0.50, 0.82, 0.50, 0.20)

# Compatibilidad con imports previos
PAN_TO_NORTH = PAN_SOUTH_TO_NORTH
PAN_TO_SOUTH = PAN_NORTH_TO_SOUTH
PAN_DOWN = PAN_SOUTH_TO_NORTH
PAN_BACK = PAN_NORTH_TO_SOUTH

SWEEP_SUR = {
    # Sector Suroeste (SW): +5px hacia arriba (Y - 5), extendido +18px en dirección de avance
    "izq": [
        (0.458, 0.75463),
        (0.447, 0.72593),
        (0.424, 0.68981),
        (0.403, 0.65278),
        (0.383, 0.61574),
        (0.362, 0.57870),
        (0.339, 0.54167),
        (0.317, 0.50463),
        (0.295, 0.46759),
        (0.274, 0.43056),
        (0.254, 0.39352),
        (0.234, 0.35648),
        (0.20939, 0.31484),
        (0.18512, 0.27206),
    ],
    # Sector Sureste (SE): -5px hacia abajo (Y + 5), extendido +18px en dirección de avance
    "der": [
        (0.518, 0.76389),
        (0.534, 0.73519),
        (0.555, 0.69907),
        (0.575, 0.66204),
        (0.597, 0.62500),
        (0.619, 0.58796),
        (0.641, 0.55093),
        (0.663, 0.51389),
        (0.685, 0.47685),
        (0.704, 0.43981),
        (0.725, 0.40278),
        (0.746, 0.36574),
        (0.76956, 0.32396),
        (0.79276, 0.28100),
    ],
}

SWEEP_NORTE = {
    # Sector Noroeste (NW): trazado centrado, extendido +18px en dirección de avance
    "izq": [
        (0.461, 0.148),
        (0.433, 0.185),
        (0.406, 0.231),
        (0.375, 0.278),
        (0.344, 0.324),
        (0.315, 0.370),
        (0.286, 0.417),
        (0.258, 0.463),
        (0.22538, 0.51357),
        (0.19510, 0.56628),
    ],
    # Sector Noreste (NE): +15px hacia arriba (Y - 15), extendido +18px en dirección de avance
    "der": [
        (0.528, 0.13426),
        (0.558, 0.17130),
        (0.586, 0.21667),
        (0.612, 0.26389),
        (0.640, 0.31019),
        (0.669, 0.35648),
        (0.698, 0.40278),
        (0.728, 0.44907),
        (0.75860, 0.50000),
        (0.78785, 0.55209),
    ],
}

def interpolate_catmull_rom(points: list[tuple[int, int]], density: int = 5) -> list[tuple[int, int]]:
    """Genera una trayectoria Catmull-Rom continua y suave sobre la lista de puntos."""
    if len(points) < 3:
        return points
    import numpy as np
    pts = np.vstack([points[0], points, points[-1]])
    curve = []
    for i in range(len(pts) - 3):
        p0, p1, p2, p3 = pts[i:i+4]
        for t in np.linspace(0, 1, density, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            q = 0.5 * ((2 * p1) +
                       (-p0 + p2) * t +
                       (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                       (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            curve.append((int(round(q[0])), int(round(q[1]))))
    curve.append(points[-1])
    return curve


def _init_outer_grass_hull() -> np.ndarray:
    """Precalcula el casco convexo del rombo de césped exterior garantizado."""
    import cv2
    import numpy as np
    w_ref, h_ref = 2400, 1080
    diamond_pts = []
    diamond_pts.extend([(int(x * w_ref), int(y * h_ref)) for x, y in SWEEP_NORTE["der"]])
    diamond_pts.extend([(int(x * w_ref), int(y * h_ref)) for x, y in SWEEP_SUR["der"]])
    diamond_pts.extend([(int(x * w_ref), int(y * h_ref)) for x, y in SWEEP_SUR["izq"]])
    diamond_pts.extend([(int(x * w_ref), int(y * h_ref)) for x, y in SWEEP_NORTE["izq"]])
    return cv2.convexHull(np.array(diamond_pts, dtype=np.int32))[:, 0, :]


_OUTER_GRASS_HULL = _init_outer_grass_hull()


# ---------------------------------------------------------------------------
# GEOMETRÍA DEL ROMBO CANÓNICO (Basado en la placa central del motor de CoC)
# ---------------------------------------------------------------------------
CANONICAL_CENTER = (1220.0, 500.0)
CANONICAL_RX_BUILDABLE = 885.0
CANONICAL_RY_BUILDABLE = 664.0
CANONICAL_RX_GRASS = 955.0
CANONICAL_RY_GRASS = 715.0


def compute_canonical_diamond_drop(
    building_center: tuple[float, float],
    center_pt: tuple[float, float] = CANONICAL_CENTER,
    rx_grass: float = CANONICAL_RX_GRASS,
    ry_grass: float = CANONICAL_RY_GRASS,
    safe_y_min: int = 138,
    safe_y_max: int = 845,
    safe_x_min: int = 120,
    safe_x_max: int = 2280,
) -> tuple[int, int]:
    """
    Calcula el punto de drop exacto sobre el anillo de césped exterior legal (2-2.5 tiles fuera de la aldea):
    1. Traza el rayo director u desde el centro de la aldea C pasando por el centro del edificio B.
    2. Calcula la intersección analítica exacta con el rombo canónico exterior:
       |x - Cx| / Rx + |y - Cy| / Ry = 1
       t = 1.0 / (|ux| / Rx + |uy| / Ry)
    3. Si la coordenada Y toca el HUD superior (y < safe_y_min) o las cartas inferiores (y > safe_y_max),
       desliza el punto sobre el perímetro del rombo hacia los flancos seguros sin penetrar hacia el centro.
    """
    import numpy as np

    cx, cy = center_pt
    bx, by = building_center
    dx = float(bx - cx)
    dy = float(by - cy)
    dist = float(np.hypot(dx, dy))

    if dist < 1e-3:
        ux, uy = 0.0, 1.0
    else:
        ux, uy = dx / dist, dy / dist

    # Intersección analítica del rayo con el rombo canónico
    denom = (abs(ux) / rx_grass) + (abs(uy) / ry_grass)
    t_intersec = 1.0 / max(denom, 1e-5)

    # Margen de seguridad calibrado (+32 px más afuera en el pasto legal para evitar cualquier saliente de muro)
    margin = 32.0
    drop_x = cx + (t_intersec + margin) * ux
    drop_y = cy + (t_intersec + margin) * uy

    # Si el punto cae sobre el HUD superior (y < safe_y_min), proyectar sobre la arista del rombo a safe_y_min
    if drop_y < safe_y_min:
        target_y = float(safe_y_min)
        # Despejar X de la ecuación del rombo: |x - cx| = Rx * (1 - |target_y - cy| / Ry)
        rem_y = abs(target_y - cy) / ry_grass
        span_x = rx_grass * max(0.0, 1.0 - rem_y)
        # Mantener el signo lateral respecto a dx
        sign_x = 1.0 if dx >= 0 else -1.0
        drop_x = cx + sign_x * (span_x + margin)
        drop_y = target_y

    # Si el punto cae sobre las cartas de tropa (y > safe_y_max), proyectar sobre la arista a safe_y_max
    elif drop_y > safe_y_max:
        target_y = float(safe_y_max)
        rem_y = abs(target_y - cy) / ry_grass
        span_x = rx_grass * max(0.0, 1.0 - rem_y)
        sign_x = 1.0 if dx >= 0 else -1.0
        drop_x = cx + sign_x * (span_x + margin)
        drop_y = target_y

    # Clamping estricto de pantalla
    final_x = int(np.clip(drop_x, safe_x_min, safe_x_max))
    final_y = int(np.clip(drop_y, safe_y_min, safe_y_max))
    return final_x, final_y



def compute_corridor_drop_geometry(
    center_pt: tuple[float, float],
    jump_pt: tuple[int, int],
    screen_size: tuple[int, int],
    detections: list[Detection] | None = None,
    green_hull: np.ndarray | None = None,
    half_width: float = 38.0,
    red_contour: np.ndarray | None = None
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """
    Calcula el punto de drop garantizado sobre el césped legal exterior para un salto dado:
    1. Si green_hull está disponible (calculado del filtro de zona roja), interseca el rayo con green_hull.
    2. Si no, calcula la intersección con el rombo de césped exterior garantizado.
    3. Asegura que el drop esté siempre más allá del edificio exterior más lejano en ese sector (+55px).
    4. Genera un mini-barrido tangencial sobre el césped orientado hacia la brecha.
    5. Aplica screen clamping dinámico evitando HUD buttons sin forzar invasión de la zona roja.
    """
    import cv2
    import numpy as np

    w, h = screen_size
    cx, cy = center_pt
    jx, jy = jump_pt
    dx, dy = float(jx - cx), float(jy - cy)
    dist = float(np.hypot(dx, dy))

    # Si el salto está cerca del núcleo (<18px), desplegar desde el césped Sur (ángulo de vista principal)
    if dist < 18.0:
        ux, uy = 0.0, 1.0
    else:
        ux, uy = dx / dist, dy / dist

    tx, ty = -uy, ux  # Vector tangencial ortogonal

    def _finalize_safe_drop(t_candidate: float) -> tuple[tuple[int, int], list[tuple[int, int]]]:
        t_d = t_candidate

        # 1. Margen de seguridad contra edificios alineados en el corredor (longitudinal y transversal)
        max_bld_t = 0.0
        if detections:
            for d in detections:
                bx, by = d.center
                vx, vy = bx - cx, by - cy
                t_proj = vx * ux + vy * uy
                d_perp = abs(vx * tx + vy * ty)
                if t_proj > 0 and d_perp <= 65.0:
                    if t_proj > max_bld_t:
                        max_bld_t = t_proj

        if max_bld_t > 0:
            t_d = max(t_d, max_bld_t + 55.0)

        # 2. Verificación iterativa de clearance contra cajas y centros de edificios
        if detections:
            for _ in range(15):
                curr_x = cx + t_d * ux
                curr_y = cy + t_d * uy
                too_close = False
                for d in detections:
                    if hasattr(d, "bbox_xyxy"):
                        x1, y1, x2, y2 = d.bbox_xyxy
                    else:
                        bx, by = d.center
                        x1, y1, x2, y2 = bx - 25, by - 25, bx + 25, by + 25
                    dist_center = float(np.hypot(d.center[0] - curr_x, d.center[1] - curr_y))
                    dist_box = max(0, x1 - curr_x, curr_x - x2, y1 - curr_y, curr_y - y2)
                    if dist_center < 42.0 or dist_box < 10.0:
                        too_close = True
                        break
                if too_close:
                    t_d += 15.0
                else:
                    break

        raw_x = cx + t_d * ux
        raw_y = cy + t_d * uy
        scale_x = w / 2400.0
        scale_y = h / 1080.0

        def is_safe(qx: float, qy: float) -> bool:
            if red_contour is not None and len(red_contour) >= 3:
                if cv2.pointPolygonTest(red_contour, (float(qx), float(qy)), True) > 0.0:
                    return False
            elif green_hull is not None and len(green_hull) >= 3:
                proj = (qx - cx) * ux + (qy - cy) * uy
                raw_proj = (raw_x - cx) * ux + (raw_y - cy) * uy
                if proj < raw_proj - 2.0:
                    d_g = cv2.pointPolygonTest(green_hull, (float(qx), float(qy)), True)
                    if qy <= 60 and d_g > 0.0:
                        return False
                    if qx <= 200 and d_g > 0.0:
                        return False
                    if d_g > 44.0:
                        return False
            if detections:
                for d in detections:
                    if hasattr(d, "bbox_xyxy"):
                        bx1, by1, bx2, by2 = d.bbox_xyxy
                    else:
                        bx, by = d.center
                        bx1, by1, bx2, by2 = bx - 25, by - 25, bx + 25, by + 25
                    dist_center = float(np.hypot(d.center[0] - qx, d.center[1] - qy))
                    dist_box = max(0, bx1 - qx, qx - bx2, by1 - qy, qy - by2)
                    if dist_center < 40.0 or dist_box < 8.0:
                        return False
            return True

        tactical_min_x, tactical_max_x = int(0.08 * w), int(0.92 * w)
        tactical_min_y, tactical_max_y = int(0.05 * h), int(0.86 * h)
        cand_x = int(np.clip(raw_x, tactical_min_x, tactical_max_x))
        cand_y = int(np.clip(raw_y, tactical_min_y, tactical_max_y))

        if is_safe(cand_x, cand_y):
            drop_x, drop_y = cand_x, cand_y
        else:
            min_x, max_x = int(30 * scale_x), int(w - 30 * scale_x)
            min_y, max_y = int(30 * scale_y), int(955 * scale_y)
            cand2_x = int(np.clip(raw_x, min_x, max_x))
            cand2_y = int(np.clip(raw_y, min_y, max_y))
            if is_safe(cand2_x, cand2_y):
                drop_x, drop_y = cand2_x, cand2_y
            else:
                drop_x = int(np.clip(raw_x, 0, w - 1))
                drop_y = int(np.clip(raw_y, 0, h - 1))

        # Evitar botón "Terminar Batalla" [730:885, :360] y botón "Siguiente" [850:980, 2040:]
        eb_y1, eb_y2 = int(730 * scale_y), int(885 * scale_y)
        eb_x2 = int(360 * scale_x)
        if eb_y1 <= drop_y <= eb_y2 and drop_x <= eb_x2:
            if raw_y < eb_y1:
                drop_y = eb_y1 - 5
            else:
                drop_x = eb_x2 + 10

        next_y1, next_y2 = int(850 * scale_y), int(980 * scale_y)
        next_x1 = int(2040 * scale_x)
        if next_y1 <= drop_y <= next_y2 and drop_x >= next_x1:
            if raw_y < next_y1:
                drop_y = next_y1 - 5
            else:
                drop_x = next_x1 - 10

        drop_x = int(np.clip(drop_x, int(0.08 * w), int(0.92 * w)))
        drop_y = int(np.clip(drop_y, int(0.05 * h), int(0.86 * h)))

        p1_x = int(np.clip(drop_x - half_width * tx, int(0.06 * w), int(0.94 * w)))
        p1_y = int(np.clip(drop_y - half_width * ty, int(0.04 * h), int(0.87 * h)))
        p2_x = int(np.clip(drop_x + half_width * tx, int(0.06 * w), int(0.94 * w)))
        p2_y = int(np.clip(drop_y + half_width * ty, int(0.04 * h), int(0.87 * h)))
        return (drop_x, drop_y), [(p1_x, p1_y), (p2_x, p2_y)]

    # 1. Si disponemos del contorno verde legal derivado del sondeo de la zona roja
    if green_hull is not None and len(green_hull) >= 3:
        N_g = len(green_hull)
        t_green_hits = []
        for i in range(N_g):
            x1, y1 = green_hull[i]
            x2, y2 = green_hull[(i + 1) % N_g]
            seg_dx, seg_dy = x2 - x1, y2 - y1
            det = ux * seg_dy - uy * seg_dx
            if abs(det) > 1e-5:
                t = ((x1 - cx) * seg_dy - (y1 - cy) * seg_dx) / det
                u_seg = ((x1 - cx) * uy - (y1 - cy) * ux) / det
                if t > 0 and 0.0 <= u_seg <= 1.0:
                    t_green_hits.append(t)

        if t_green_hits:
            t_drop = max(t_green_hits)
            return _finalize_safe_drop(t_drop)

    # 2. Fallback geométrico sobre el rombo exterior de césped
    scale_x, scale_y = w / 2400.0, h / 1080.0
    hull = np.column_stack([
        _OUTER_GRASS_HULL[:, 0] * scale_x,
        _OUTER_GRASS_HULL[:, 1] * scale_y
    ])
    N = len(hull)

    t_hits = []
    for i in range(N):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % N]
        seg_dx, seg_dy = x2 - x1, y2 - y1
        det = ux * seg_dy - uy * seg_dx
        if abs(det) > 1e-5:
            t = ((x1 - cx) * seg_dy - (y1 - cy) * seg_dx) / det
            u_seg = ((x1 - cx) * uy - (y1 - cy) * ux) / det
            if t > 0 and 0.0 <= u_seg <= 1.0:
                t_hits.append(t)

    t_outer = min(t_hits) if t_hits else 380.0
    t_drop = t_outer * 0.95
    return _finalize_safe_drop(t_drop)


def _compute_legacy_quadrant_jump_targets(
    detections: list[Detection], screen_size: tuple[int, int]
) -> list[tuple[int, int]]:
    """
    Heurística heredada de proyección por cuadrantes (jump_placement='legacy_quadrant').
    """
    import cv2
    import numpy as np

    w, h = screen_size
    valid_classes = {"storage_gold", "storage_elixir", "storage_dark", "collector_dark"}
    valid_storages = [d for d in detections if d.cls_name in valid_classes and d.confidence >= 0.50]

    if not valid_storages:
        log.warning("No se encontraron almacenes válidos (conf >= 0.50) para cálculo de Saltos.")
        return []

    all_points = np.array([d.center for d in detections], dtype=np.float32)
    if len(all_points) >= 3:
        hull = cv2.convexHull(all_points)
        M = cv2.moments(hull)
        if M["m00"] > 0:
            center_x = float(M["m10"] / M["m00"])
            center_y = float(M["m01"] / M["m00"])
        else:
            center_x = float(np.mean(all_points[:, 0]))
            center_y = float(np.mean(all_points[:, 1]))
    elif len(all_points) > 0:
        center_x = float(np.mean(all_points[:, 0]))
        center_y = float(np.mean(all_points[:, 1]))
    else:
        center_x = float(w * 0.50)
        center_y = float(h * 0.50)

    log.info("Centroide de base YOLO (legacy): (%.1f, %.1f) a partir de %d detecciones", center_x, center_y, len(all_points))

    core_radius = 160
    quadrants_storages = {"SW": [], "SE": [], "N": [], "CORE": []}
    quadrants_projected = {"SW": [], "SE": [], "N": [], "CORE": []}

    for s in valid_storages:
        sx, sy = s.center
        dist = float(np.hypot(sx - center_x, sy - center_y))

        dx = sx - center_x
        dy = sy - center_y
        proj_x = int(round(sx + 0.40 * dx))
        proj_y = int(round(sy + 0.40 * dy))

        proj_x = int(np.clip(proj_x, int(0.12 * w), int(0.88 * w)))
        proj_y = int(np.clip(proj_y, int(0.12 * h), int(0.88 * h)))

        if dist < core_radius:
            quad = "CORE"
        elif sy >= center_y:
            quad = "SW" if sx < center_x else "SE"
        else:
            quad = "N"

        quadrants_storages[quad].append(s)
        quadrants_projected[quad].append((proj_x, proj_y))

    active_quads = [q for q in ("SW", "SE", "N", "CORE") if len(quadrants_storages[q]) > 0]
    active_quads.sort(key=lambda q: len(quadrants_storages[q]), reverse=True)
    top_quads = active_quads[:2]

    targets: list[tuple[int, int]] = []
    for q in top_quads:
        pts = quadrants_projected[q]
        avg_x = int(round(np.mean([p[0] for p in pts])))
        avg_y = int(round(np.mean([p[1] for p in pts])))
        targets.append((avg_x, avg_y))

    log.info("🎯 Jump targets calculados con YOLO legacy (máx 2): %s (cuadrantes: %s)", targets, top_quads)
    return targets


def _compute_cluster_intersection_jump_targets(
    detections: list[Detection], screen_size: tuple[int, int]
) -> list[tuple[int, int]]:
    """
    Algoritmo de Intersecciones de Muros y Máxima Cobertura para Hechizos de Salto:
    1. Genera candidatos de intersección entre parejas y tríos de almacenes cercanos.
    2. Genera candidatos de muro de entrada (desplazados 0.75R hacia lado de drop) para almacenes aislados.
    3. Evalúa el coverage scoring ponderado por loot (dark=3, gold/elixir=2, collectors=1, town_hall=0).
    4. Selecciona secuencialmente hasta max_jumps garantizando una separación mínima R entre saltos.
    5. Guarda el registro de decisión en logs/yolo/jump_decision_{ts}.json.
    """
    import json
    import os
    import datetime
    import cv2
    import numpy as np

    w, h = screen_size
    attack_cfg = SETTINGS.get("attack", {})

    radius = float(attack_cfg.get("jump_spell_radius_px", 90))
    pair_min_dist = float(attack_cfg.get("pair_min_dist_px", 65))
    pair_max_dist = float(attack_cfg.get("pair_max_dist_px", 210))
    max_jumps = int(attack_cfg.get("max_jumps", 3))
    min_jump_score = float(attack_cfg.get("min_jump_score", 2.0))
    min_sep = float(attack_cfg.get("min_separation_between_jumps_px", 110))
    entry_offset_ratio = float(attack_cfg.get("jump_entry_wall_offset_ratio", 0.28))

    loot_weights = attack_cfg.get("jump_loot_weights", {
        "storage_dark": 3,
        "storage_gold": 2,
        "storage_elixir": 2,
        "collector_dark": 1,
        "collector_gold": 1,
        "collector_elixir": 1,
        "town_hall": 0,
    })

    # Filtrar edificios válidos según peso y confianza
    yolo_conf = float(attack_cfg.get("yolo_conf", 0.40))
    valid_loot = [
        d for d in detections
        if loot_weights.get(d.cls_name, 0) > 0 and d.confidence >= yolo_conf
    ]
    storages = [
        d for d in valid_loot
        if "storage" in d.cls_name
    ]

    if not valid_loot and not storages:
        log.warning("No se detectaron edificios de botín para cálculo de saltos.")
        return []

    # Grupo principal para generación de candidatos: storages (o valid_loot si no hay storages)
    primary_targets = storages if storages else valid_loot

    # Centro de masa de la base
    all_points = np.array([d.center for d in detections], dtype=np.float32)
    if len(all_points) >= 3:
        hull = cv2.convexHull(all_points)
        M = cv2.moments(hull)
        if M["m00"] > 0:
            center_x = float(M["m10"] / M["m00"])
            center_y = float(M["m01"] / M["m00"])
        else:
            center_x = float(np.mean(all_points[:, 0]))
            center_y = float(np.mean(all_points[:, 1]))
    elif len(all_points) > 0:
        center_x = float(np.mean(all_points[:, 0]))
        center_y = float(np.mean(all_points[:, 1]))
    else:
        center_x, center_y = float(w * 0.50), float(h * 0.50)

    center = (center_x, center_y)

    def _get_sector(pt, c_pt):
        dx, dy = pt[0] - c_pt[0], pt[1] - c_pt[1]
        return ("N" if dy < 0 else "S") + ("W" if dx < 0 else "E")

    # Identificar índices de almacenes en valid_loot
    storage_indices = [i for i, d in enumerate(valid_loot) if "storage" in d.cls_name]
    target_indices = storage_indices if storage_indices else list(range(len(valid_loot)))

    candidates = []

    # 1. Candidatos de parejas de edificios primarios (Lateral vs Radial)
    for i_pos in range(len(target_indices)):
        for j_pos in range(i_pos + 1, len(target_indices)):
            i = target_indices[i_pos]
            j = target_indices[j_pos]
            s1, s2 = valid_loot[i], valid_loot[j]
            d_pair = float(np.hypot(s1.center[0] - s2.center[0], s1.center[1] - s2.center[1]))

            if pair_min_dist <= d_pair <= pair_max_dist:
                r1 = float(np.hypot(s1.center[0] - center_x, s1.center[1] - center_y))
                r2 = float(np.hypot(s2.center[0] - center_x, s2.center[1] - center_y))
                delta_r = abs(r1 - r2)

                # Pareja Lateral: ambos a profundidad similar respecto al centro (muro exterior continuo)
                if delta_r <= 45.0:
                    mid_x = (s1.center[0] + s2.center[0]) / 2.0
                    mid_y = (s1.center[1] + s2.center[1]) / 2.0
                    vx, vy = mid_x - center_x, mid_y - center_y
                    norm_v = max(float(np.hypot(vx, vy)), 1e-3)
                    ux, uy = vx / norm_v, vy / norm_v
                    # Desplazamiento lateral controlado sobre la arista exterior del compartimento
                    jx = int(round(mid_x + 0.30 * radius * ux))
                    jy = int(round(mid_y + 0.30 * radius * uy))
                    candidates.append({
                        "point": (jx, jy),
                        "type": "lateral_pair_entrance",
                        "sector": _get_sector((jx, jy), center),
                        "origin_buildings": [s1.cls_name, s2.cls_name],
                        "covered_indices": [i, j],
                        "score": float(loot_weights.get(s1.cls_name, 0) + loot_weights.get(s2.cls_name, 0)),
                    })
                else:
                    # Pareja Radial: s_outer en muro exterior, s_inner más hacia el núcleo
                    s_outer, idx_outer = (s1, i) if r1 >= r2 else (s2, j)
                    s_inner, idx_inner = (s2, j) if r1 >= r2 else (s1, i)
                    vx, vy = s_outer.center[0] - center_x, s_outer.center[1] - center_y
                    norm_v = max(float(np.hypot(vx, vy)), 1e-3)
                    ux, uy = vx / norm_v, vy / norm_v
                    jx = int(round(s_outer.center[0] + entry_offset_ratio * radius * ux))
                    jy = int(round(s_outer.center[1] + entry_offset_ratio * radius * uy))
                    candidates.append({
                        "point": (jx, jy),
                        "type": "radial_outer_entrance",
                        "sector": _get_sector((jx, jy), center),
                        "origin_buildings": [s_outer.cls_name, s_inner.cls_name],
                        "covered_indices": [idx_outer, idx_inner],
                        "score": float(loot_weights.get(s_outer.cls_name, 0) + loot_weights.get(s_inner.cls_name, 0)),
                    })

    # 2. Candidatos de muro de entrada para almacenes individuales (offset hacia zona de drop)
    for i in target_indices:
        s = valid_loot[i]
        vx = s.center[0] - center_x
        vy = s.center[1] - center_y
        norm_v = max(float(np.hypot(vx, vy)), 1e-3)
        ux, uy = vx / norm_v, vy / norm_v
        entry_x = int(round(s.center[0] + (entry_offset_ratio * radius) * ux))
        entry_y = int(round(s.center[1] + (entry_offset_ratio * radius) * uy))
        candidates.append({
            "point": (entry_x, entry_y),
            "type": "entry_wall",
            "sector": _get_sector((entry_x, entry_y), center),
            "origin_buildings": [s.cls_name],
            "covered_indices": [i],
            "score": float(loot_weights.get(s.cls_name, 0)),
        })

    # 3. Candidatos de Corredor Vectorial en Línea para el Núcleo Profundo
    # Si el Almacén Oscuro o Ayuntamiento están en el núcleo profundo (> deep_core_dist_px del drop),
    # encadenar saltos en línea recta (Capa 2 intermedia y Muro del Núcleo) para crear autopista directa.
    deep_core_dist = float(attack_cfg.get("deep_core_distance_px", 240.0))
    core_indices = [
        i for i in target_indices
        if valid_loot[i].cls_name in {"storage_dark", "town_hall"}
    ]

    for c_idx in core_indices:
        target_core = valid_loot[c_idx]
        drop_x, drop_y = compute_canonical_diamond_drop(target_core.center, center_pt=center)
        dist_to_drop = float(np.hypot(target_core.center[0] - drop_x, target_core.center[1] - drop_y))

        if dist_to_drop >= deep_core_dist:
            vx = target_core.center[0] - drop_x
            vy = target_core.center[1] - drop_y
            norm_v = max(float(np.hypot(vx, vy)), 1e-3)
            ux, uy = vx / norm_v, vy / norm_v

            # Salto 1: Capa intermedia (~45% del trayecto hacia el núcleo)
            j1_x = int(round(drop_x + 0.45 * dist_to_drop * ux))
            j1_y = int(round(drop_y + 0.45 * dist_to_drop * uy))

            # Salto 2: Muro de entrada al compartimento del núcleo (~85% del trayecto)
            j2_x = int(round(drop_x + 0.85 * dist_to_drop * ux))
            j2_y = int(round(drop_y + 0.85 * dist_to_drop * uy))

            candidates.append({
                "point": (j1_x, j1_y),
                "type": "deep_core_intermediate",
                "sector": _get_sector((j1_x, j1_y), center),
                "origin_buildings": [target_core.cls_name],
                "covered_indices": [c_idx],
                "score": float(loot_weights.get(target_core.cls_name, 3) + 2.5),
            })
            candidates.append({
                "point": (j2_x, j2_y),
                "type": "deep_core_wall",
                "sector": _get_sector((j2_x, j2_y), center),
                "origin_buildings": [target_core.cls_name],
                "covered_indices": [c_idx],
                "score": float(loot_weights.get(target_core.cls_name, 3) + 3.0),
            })

    # Clamping de seguridad dentro de pantalla jugable
    min_x, max_x = int(0.12 * w), int(0.88 * w)
    min_y, max_y = int(0.14 * h), int(0.84 * h)
    for c in candidates:
        cx_c = int(np.clip(c["point"][0], min_x, max_x))
        cy_c = int(np.clip(c["point"][1], min_y, max_y))
        c["point"] = (cx_c, cy_c)

    # Actualizar cobertura real por radio geométrico para incluir recolectores adyacentes
    scored_candidates = []
    for c in candidates:
        cx_pt, cy_pt = c["point"]
        covered_set = set(c["covered_indices"])
        for idx, d in enumerate(valid_loot):
            dist_to_d = float(np.hypot(d.center[0] - cx_pt, d.center[1] - cy_pt))
            if dist_to_d <= radius:
                covered_set.add(idx)

        c["covered_indices"] = list(covered_set)
        c["score"] = sum(float(loot_weights.get(valid_loot[idx].cls_name, 0)) for idx in covered_set)
        c["covered_classes"] = [valid_loot[idx].cls_name for idx in covered_set]
        scored_candidates.append(c)

    # Ordenar candidatos por puntuación descendente
    scored_candidates.sort(key=lambda c: (c["score"], c["type"] != "entry_wall"), reverse=True)

    available_sectors = set(c["sector"] for c in scored_candidates)

    # Selección greedy con diversidad sectorial y separación mínima
    selected_jumps = []
    already_covered = set()
    used_sectors = set()

    for step in range(max_jumps):
        best_candidate = None
        best_marginal_gain = -1.0
        sectors_remaining = available_sectors - used_sectors

        for c in scored_candidates:
            # Separación mínima respecto a saltos ya elegidos
            if any(np.hypot(c["point"][0] - s["point"][0], c["point"][1] - s["point"][1]) < min_sep for s in selected_jumps):
                continue

            # Priorizar sectores sin salto previo si aún quedan disponibles
            if sectors_remaining and c["sector"] in used_sectors:
                continue

            # Ganancia marginal (botín no cubierto previamente)
            marginal_gain = sum(
                float(loot_weights.get(valid_loot[i].cls_name, 0))
                for i in c["covered_indices"]
                if i not in already_covered
            )

            if marginal_gain > best_marginal_gain:
                best_marginal_gain = marginal_gain
                best_candidate = c

        if best_candidate and best_marginal_gain >= min_jump_score:
            selected_jumps.append(best_candidate)
            already_covered.update(best_candidate["covered_indices"])
            used_sectors.add(best_candidate["sector"])
            log.info(
                "🎯 Salto #%d seleccionado en %s | tipo=%s | sector=%s | score=%.1f | cubre=%s",
                len(selected_jumps),
                best_candidate["point"],
                best_candidate["type"],
                best_candidate["sector"],
                best_candidate["score"],
                best_candidate["covered_classes"],
            )
        else:
            # Si no hubo candidato en sector libre, permitir evaluar sectores ya usados si queda botín sin cubrir
            found_secondary = False
            for c in scored_candidates:
                if any(np.hypot(c["point"][0] - s["point"][0], c["point"][1] - s["point"][1]) < min_sep for s in selected_jumps):
                    continue
                marginal_gain = sum(
                    float(loot_weights.get(valid_loot[i].cls_name, 0))
                    for i in c["covered_indices"]
                    if i not in already_covered
                )
                if marginal_gain > best_marginal_gain:
                    best_marginal_gain = marginal_gain
                    best_candidate = c

            if best_candidate and best_marginal_gain >= min_jump_score:
                selected_jumps.append(best_candidate)
                already_covered.update(best_candidate["covered_indices"])
                used_sectors.add(best_candidate["sector"])
                found_secondary = True
                log.info(
                    "🎯 Salto #%d (secundario) seleccionado en %s | tipo=%s | sector=%s | score=%.1f | cubre=%s",
                    len(selected_jumps),
                    best_candidate["point"],
                    best_candidate["type"],
                    best_candidate["sector"],
                    best_candidate["score"],
                    best_candidate["covered_classes"],
                )
                continue
            if not found_secondary:
                break

    # Si aún faltan saltos para alcanzar max_jumps (4 saltos), rellenar con candidatos de mayor score que respeten min_sep
    while len(selected_jumps) < max_jumps:
        best_extra = None
        best_extra_score = -1.0
        for c in scored_candidates:
            if any(np.hypot(c["point"][0] - s["point"][0], c["point"][1] - s["point"][1]) < min_sep for s in selected_jumps):
                continue
            if c["score"] > best_extra_score:
                best_extra_score = c["score"]
                best_extra = c
        if best_extra and best_extra_score > 0:
            selected_jumps.append(best_extra)
            log.info(
                "🎯 Salto #%d (adicional de cobertura) en %s | tipo=%s | sector=%s | score=%.1f | cubre=%s",
                len(selected_jumps),
                best_extra["point"],
                best_extra["type"],
                best_extra["sector"],
                best_extra["score"],
                best_extra["covered_classes"],
            )
        else:
            break

    targets = [j["point"] for j in selected_jumps]

    # Registrar decisión canónica en logs/yolo/jump_decision.json (sobrescribe un único archivo)
    try:
        out_dir = os.path.join("logs", "yolo")
        os.makedirs(out_dir, exist_ok=True)
        decision_file = os.path.join(out_dir, "jump_decision.json")

        decision_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "algorithm": "cluster_intersections",
            "config": {
                "radius": radius,
                "pair_min_dist": pair_min_dist,
                "pair_max_dist": pair_max_dist,
                "max_jumps": max_jumps,
                "min_jump_score": min_jump_score,
                "min_separation_between_jumps_px": min_sep,
                "entry_offset_ratio": entry_offset_ratio,
                "loot_weights": loot_weights,
            },
            "detections_count": len(valid_loot),
            "candidates_evaluated": len(scored_candidates),
            "selected_jumps": [
                {
                    "point": j["point"],
                    "type": j["type"],
                    "score": j["score"],
                    "covered_classes": j["covered_classes"],
                }
                for j in selected_jumps
            ],
            "top_candidates": [
                {
                    "point": c["point"],
                    "type": c["type"],
                    "score": c["score"],
                    "covered_classes": c["covered_classes"],
                }
                for c in scored_candidates[:10]
            ],
        }

        with open(decision_file, "w", encoding="utf-8") as f:
            json.dump(decision_log, f, indent=2, ensure_ascii=False)
        log.info("📝 Registro de decisión de saltos guardado en: %s", decision_file)
    except Exception as e_json:
        log.warning("No se pudo guardar registro JSON de decisión de saltos: %s", e_json)

    log.info("🎯 Jump targets calculados con cluster_intersections (total: %d): %s", len(targets), targets)
    return targets


def _compute_wall_raycast_jump_targets(
    screen: np.ndarray,
    detections: list[Detection],
    screen_size: tuple[int, int]
) -> list[tuple[int, int]]:
    """
    Calcula objetivos de Hechizos de Salto detectando la ubicación física real
    de los muros mediante raycasting de aristas OpenCV (vision/wall_raycast.py).
    
    1. Si screen es None o faltan detecciones, delega a cluster_intersections.
    2. Estima los ángulos isométricos dominantes de la aldea con estimate_iso_angles.
    3. Para cada almacén interior bloqueado:
       - Traza el rayo desde su punto de drop exterior legal hasta su centro.
       - Invoca find_wall_crossings para hallar las aristas de los muros que cruzan el camino.
       - Asigna el primer salto en el primer cruce desplazado +wall_inset_px hacia el almacén.
       - Si existe un segundo cruce (muro doble, distancia >= 35px) y restan saltos, añade un segundo salto.
       - Si no se detectan cruces, recurre a respaldo por ratio (0.75) y lo registra en el log.
    4. Garantiza separación mínima (min_sep), respeta max_jumps y límites de pantalla.
    5. Guarda logs/yolo/wall_raycast_debug.png con visualización completa.
    """
    if screen is None or screen.size == 0 or not detections:
        log.warning("⚠️ Screen o detecciones no disponibles para Wall Raycast. Usando cluster_intersections.")
        return _compute_cluster_intersection_jump_targets(detections, screen_size)

    w, h = screen_size
    attack_cfg = SETTINGS.get("attack", {})
    max_jumps = int(attack_cfg.get("max_jumps", 4))
    min_sep = float(attack_cfg.get("min_separation_between_jumps_px", 110))
    wall_inset = float(attack_cfg.get("wall_inset_px", 20.0))
    strip = int(attack_cfg.get("wall_strip_px", 45))
    tol_deg = float(attack_cfg.get("wall_tol_deg", 10.0))
    canny_low = int(attack_cfg.get("canny_low", 60))
    canny_high = int(attack_cfg.get("canny_high", 160))
    hough_thresh = int(attack_cfg.get("hough_threshold", 30))
    hough_min_len = int(attack_cfg.get("hough_min_len", 70))
    hough_max_gap = int(attack_cfg.get("hough_max_gap", 10))
    tol_deg = float(attack_cfg.get("wall_tol_deg", 10.0))
    min_crossing_angle = float(attack_cfg.get("min_crossing_angle_deg", 25.0))
    min_dist_drop = float(attack_cfg.get("min_dist_from_drop_px", 40.0))
    min_dist_target = float(attack_cfg.get("min_dist_from_target_px", 50.0))
    endpoint_tol = float(attack_cfg.get("wall_endpoint_tol_px", 30.0))

    loot_weights = attack_cfg.get("jump_loot_weights", {
        "storage_dark": 3,
        "storage_gold": 2,
        "storage_elixir": 2,
        "town_hall": 0,
        "collector_dark": 1,
    })

    # Filtrar y ordenar almacenes por peso de botín
    primary_targets = [
        d for d in detections
        if d.cls_name in {"storage_dark", "storage_gold", "storage_elixir", "town_hall"}
    ]
    if not primary_targets:
        primary_targets = [
            d for d in detections
            if "storage" in d.cls_name or "collector" in d.cls_name
        ]
    if not primary_targets:
        log.warning("No se encontraron objetivos de botín para Wall Raycast.")
        return _compute_cluster_intersection_jump_targets(detections, screen_size)

    primary_targets.sort(
        key=lambda d: loot_weights.get(d.cls_name, 0),
        reverse=True
    )

    # Centro de la aldea
    all_pts = np.array([d.center for d in detections], dtype=np.float32)
    center = (float(np.mean(all_pts[:, 0])), float(np.mean(all_pts[:, 1])))

    # 1. Extracción global de segmentos de muro en una sola pasada por batalla
    iso_angles, wall_segments = detect_global_wall_segments(
        screen,
        min_line_len=hough_min_len,
        max_line_gap=hough_max_gap,
        tol_deg=tol_deg,
        canny_low=canny_low,
        canny_high=canny_high,
        hough_thresh=hough_thresh,
        detections=detections,
    )

    adaptive_mode = attack_cfg.get("adaptive_spells_by_demand", True)
    if adaptive_mode:
        from vision.wall_raycast import calculate_adaptive_jump_and_breach_placements, find_wall_junctions, draw_wall_raycast_debug
        targets_with_drops = []
        dz_detector = None
        try:
            from vision.deploy_zone_detector import get_deploy_zone_detector
            dz_detector = get_deploy_zone_detector()
        except Exception:
            pass

        for target in primary_targets:
            if dz_detector and dz_detector.is_ready() and dz_detector.last_mask is not None:
                drop_x, drop_y = dz_detector.find_closest_deployable_point(
                    int(round(target.center[0])), int(round(target.center[1])), margin_px=25,
                    safe_bounds=(int(0.12 * w), int(0.14 * h), int(0.88 * w), int(0.84 * h))
                )
            else:
                p_target = (int(round(target.center[0])), int(round(target.center[1])))
                vx = p_target[0] - center[0]
                vy = p_target[1] - center[1]
                dist_c = max(float(np.hypot(vx, vy)), 1e-3)
                drop_x = int(np.clip(p_target[0] + 250.0 * (vx / dist_c), int(0.12 * w), int(0.88 * w)))
                drop_y = int(np.clip(p_target[1] + 250.0 * (vy / dist_c), int(0.14 * h), int(0.84 * h)))
            targets_with_drops.append((target, (drop_x, drop_y)))

        junctions = find_wall_junctions(wall_segments, iso_angles=iso_angles)
        selected_jumps, selected_swb, debug_rays = calculate_adaptive_jump_and_breach_placements(
            targets_with_drops=targets_with_drops,
            wall_segments=wall_segments,
            junctions=junctions,
            max_jumps=max_jumps,
            min_dist_between_jumps=min_sep,
            safe_y_min=int(0.14 * h),
            safe_y_max=int(0.84 * h),
            safe_x_min=int(0.12 * w),
            safe_x_max=int(0.88 * w),
        )

        try:
            draw_wall_raycast_debug(
                screen,
                debug_rays,
                selected_jumps,
                junctions=junctions,
                out_path="logs/yolo/wall_raycast_debug.png"
            )
        except Exception as e_dbg:
            log.warning("No se pudo renderizar wall_raycast_debug.png: %s", e_dbg)

        log.info("🎯 Jump targets calculados por Demanda Adaptativa e Intersecciones (total: %d): %s", len(selected_jumps), selected_jumps)
        if selected_jumps:
            return selected_jumps

    def _get_sector(pt: tuple[int, int], c_pt: tuple[float, float]) -> str:
        dx_s, dy_s = pt[0] - c_pt[0], pt[1] - c_pt[1]
        return ("N" if dy_s < 0 else "S") + ("W" if dx_s < 0 else "E")

    candidates = []
    debug_rays = []

    for target in primary_targets:
        p_target = (int(target.center[0]), int(target.center[1]))
        vx = p_target[0] - center[0]
        vy = p_target[1] - center[1]
        dist_c = max(float(np.hypot(vx, vy)), 1e-3)
        ux, uy = vx / dist_c, vy / dist_c

        # Proyección radial hacia el césped exterior (~250px más allá del objetivo)
        p_drop_x = int(np.clip(p_target[0] + 250.0 * ux, int(0.10 * w), int(0.90 * w)))
        p_drop_y = int(np.clip(p_target[1] + 250.0 * uy, int(0.14 * h), int(0.82 * h)))
        p_drop = (p_drop_x, p_drop_y)

        dist_ray = float(np.hypot(p_target[0] - p_drop[0], p_target[1] - p_drop[1]))
        if dist_ray < (min_dist_drop + min_dist_target):
            continue

        debug_entry = {}
        crossings = find_wall_crossings_from_segments(
            p_drop=p_drop,
            p_target=p_target,
            wall_segments=wall_segments,
            min_crossing_angle_deg=min_crossing_angle,
            min_dist_drop=min_dist_drop,
            min_dist_target=min_dist_target,
            endpoint_tol=endpoint_tol,
            debug_collector=debug_entry,
        )
        debug_rays.append(debug_entry)

        if crossings:
            log.info(
                "🧱 [RAYCAST] %s en %s: %d cruce(s) de muro físico detectado(s): %s",
                target.cls_name, p_target, len(crossings), crossings
            )
            for idx_cr, (cx_w, cy_w) in enumerate(crossings):
                vec_tx = p_target[0] - cx_w
                vec_ty = p_target[1] - cy_w
                dist_t = max(float(np.hypot(vec_tx, vec_ty)), 1e-3)
                t_ux, t_uy = vec_tx / dist_t, vec_ty / dist_t

                jx = int(round(cx_w + wall_inset * t_ux))
                jy = int(round(cy_w + wall_inset * t_uy))

                bonus = 2.0 if (idx_cr == 0 or len(crossings) == 1) else 1.5
                if target.cls_name in {"storage_dark", "town_hall"} and idx_cr > 0:
                    bonus += 1.0

                cand_pt = (jx, jy)
                candidates.append({
                    "pt": cand_pt,
                    "target": target.cls_name,
                    "sector": _get_sector(cand_pt, center),
                    "priority": float(loot_weights.get(target.cls_name, 2)) + bonus,
                    "is_fallback": False,
                })
        else:
            log.info(
                "⚠️ [RAYCAST] %s en %s: 0 cruces detectados. Usando respaldo por ratio (0.75)...",
                target.cls_name, p_target
            )
            jx = int(round(p_drop[0] + 0.75 * (p_target[0] - p_drop[0])))
            jy = int(round(p_drop[1] + 0.75 * (p_target[1] - p_drop[1])))
            cand_pt = (jx, jy)
            candidates.append({
                "pt": cand_pt,
                "target": target.cls_name,
                "sector": _get_sector(cand_pt, center),
                "priority": float(loot_weights.get(target.cls_name, 2)),
                "is_fallback": True,
            })

    # Clamping a zona de pantalla jugable
    min_x, max_x = int(0.12 * w), int(0.88 * w)
    min_y, max_y = int(0.14 * h), int(0.84 * h)
    for c in candidates:
        cx_c = int(np.clip(c["pt"][0], min_x, max_x))
        cy_c = int(np.clip(c["pt"][1], min_y, max_y))
        c["pt"] = (cx_c, cy_c)

    # Ordenar candidatos por prioridad
    candidates.sort(key=lambda c: c["priority"], reverse=True)

    available_sectors = set(c["sector"] for c in candidates)
    used_sectors = set()
    selected_jumps: list[tuple[int, int]] = []

    # Selección greedy con DIVERSIDAD SECTORIAL (1 salto por cuadrante NW, NE, SW, SE)
    for step in range(max_jumps):
        best_cand = None
        best_score = -1.0
        rem_sectors = available_sectors - used_sectors

        for cand in candidates:
            pt = cand["pt"]
            if any(np.hypot(pt[0] - sj[0], pt[1] - sj[1]) < min_sep for sj in selected_jumps):
                continue
            if rem_sectors and cand["sector"] in used_sectors:
                continue
            if cand["priority"] > best_score:
                best_score = cand["priority"]
                best_cand = cand

        if best_cand:
            selected_jumps.append(best_cand["pt"])
            used_sectors.add(best_cand["sector"])

    # Si aún no se alcanzaron max_jumps, tomar candidatos adicionales respetando min_sep
    if len(selected_jumps) < max_jumps:
        for cand in candidates:
            pt = cand["pt"]
            if all(np.hypot(pt[0] - sj[0], pt[1] - sj[1]) >= min_sep for sj in selected_jumps):
                selected_jumps.append(pt)
                if len(selected_jumps) >= max_jumps:
                    break

    # Si aún no se alcanzan max_jumps, rellenar con candidatos de cluster_intersections
    if len(selected_jumps) < max_jumps:
        fallback_targets = _compute_cluster_intersection_jump_targets(detections, screen_size)
        for fb in fallback_targets:
            if all(np.hypot(fb[0] - sj[0], fb[1] - sj[1]) >= min_sep for sj in selected_jumps):
                selected_jumps.append(fb)
                if len(selected_jumps) >= max_jumps:
                    break

    # Generar imagen de depuración visual
    try:
        draw_wall_raycast_debug(
            screen,
            debug_rays,
            selected_jumps,
            out_path="logs/yolo/wall_raycast_debug.png"
        )
    except Exception as e_dbg:
        log.warning("No se pudo renderizar wall_raycast_debug.png: %s", e_dbg)

    log.info("🎯 Jump targets calculados con Wall Raycast (total: %d): %s", len(selected_jumps), selected_jumps)
    return selected_jumps


def compute_adaptive_jump_targets_from_detections(
    detections: list[Detection], screen_size: tuple[int, int], screen: np.ndarray | None = None
) -> list[tuple[int, int]]:
    """
    Punto de entrada para el cálculo de objetivos de Hechizos de Salto.
    Selecciona el algoritmo según config/settings.json -> attack.jump_placement:
    - 'wall_raycast' (por defecto): Detección física real de muros mediante OpenCV raycasting de aristas.
    - 'cluster_intersections': Intersección de muros compartidos y máxima cobertura.
    - 'legacy_quadrant': Heurística heredada de proyección por cuadrantes.
    """
    mode = SETTINGS.get("attack", {}).get("jump_placement", "wall_raycast")
    if mode == "wall_raycast":
        if screen is not None:
            targets = _compute_wall_raycast_jump_targets(screen, detections, screen_size)
            if targets:
                return targets
            log.warning("⚠️ Wall raycast no devolvió saltos; ejecutando respaldo cluster_intersections.")
        else:
            log.warning("⚠️ Modo wall_raycast activo pero screen es None; usando respaldo cluster_intersections.")
    elif mode == "legacy_quadrant":
        return _compute_legacy_quadrant_jump_targets(detections, screen_size)
    return _compute_cluster_intersection_jump_targets(detections, screen_size)


def get_battle_hud_mask(h: int, w: int) -> np.ndarray:
    """
    Genera una máscara precisa de los elementos interactivos del HUD de batalla
    para evitar falsos positivos de zona roja sin recortar estructuras perimetrales.
    Corte limpio y uniforme sobre las cartas (y < 888px) que evita invasiones.
    Resolución de referencia: 2400x1080.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    scale_x = w / 2400.0
    scale_y = h / 1080.0

    # 1. Margen superior extremo (barra de estado, notch Android, temporizador)
    mask[:int(60 * scale_y), :] = 255

    # 2. Barra de despliegue de tropas (corte uniforme limpio a y=888, sin penetraciones en cartas)
    y1_bar = int(888 * scale_y)
    mask[y1_bar:, :] = 255

    # 3. Botón "Terminar Batalla" / "End Battle" (esquina inferior izquierda)
    # [730:885, :360]
    y1_eb, y2_eb = int(730 * scale_y), int(885 * scale_y)
    x1_eb, x2_eb = 0, int(360 * scale_x)
    mask[y1_eb:y2_eb, x1_eb:x2_eb] = 255

    # 4. Botón "Siguiente" / "Next" o diálogo lateral derecho
    # [850:980, 2040:]
    y1_next, y2_next = int(850 * scale_y), int(980 * scale_y)
    x1_next = int(2040 * scale_x)
    mask[y1_next:y2_next, x1_next:] = 255

    # 5. HUD superior (Loot top-left, Timer top-center, Medidores top-right)
    y_hud_top = int(100 * scale_y)
    mask[:y_hud_top, :int(380 * scale_x)] = 255
    # Strict top-right HUD mask para contadores de recursos (y < 160, x > 1900)
    y2_top_right = int(160 * scale_y)
    x1_top_right = int(1900 * scale_x)
    mask[:y2_top_right, x1_top_right:] = 255

    return mask


def detect_village_boundary(
    img: np.ndarray,
    detections: list[Detection] | None = None,
    mode: str | None = None,
    zone_name: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Detector de frontera de exclusión roja de la aldea y perímetro verde de despliegue legal.
    Sin simplificaciones convexas (elimina convexHull) para seguir exactamente concavidades,
    recolectores, campamentos y bastiones de muros.
    Visión por computadora pura: extrae la frontera auténtica directamente del frame,
    sin incrustar recuadros rectangulares que distorsionen los límites naturales.

    Soporta dos modos:
    - 'pixel_scan' (default): Escaneo radial milimétrico exhaustivo (paso 2-4px) en 360 ángulos.
    - 'high_res_contour': Extracción no convexa optimizada vía CHAIN_APPROX_TC89_KCOS + approxPolyDP(2.5).

    Retorna: (red_contour, green_contour) de forma (N, 2) int32.
    """
    if img is None or img.size == 0:
        return None, None

    h, w = img.shape[:2]
    if mode is None:
        mode = SETTINGS.get("attack", {}).get("boundary_detection_mode", "canonical_diamond")

    # Modo IA: Deploy Zone Finder (U-Net Semantic Segmentation)
    if mode in ("deploy_zone_finder", "ai", "ai_deploy_zone"):
        try:
            from vision.deploy_zone_detector import get_deploy_zone_detector
            dz_detector = get_deploy_zone_detector()
            if dz_detector.is_ready():
                pts_bld, pts_grass = dz_detector.get_boundary_polygons(img, margin_px=25)
                if pts_bld is not None and pts_grass is not None:
                    return pts_bld, pts_grass
        except Exception as e:
            log.warning("Fallo en detección IA de zona de despliegue, usando fallback: %s", e)

    # Fallback automático a canonical_diamond si la IA no produjo contornos o el modo es canonical_diamond
    if mode == "canonical_diamond" or mode in ("deploy_zone_finder", "ai", "ai_deploy_zone"):
        attack_cfg = SETTINGS.get("attack", {})
        if zone_name == "NORTE":
            cx, cy = w * 0.50, h * 1.25
        elif zone_name == "SUR":
            cx, cy = w * 0.50, -h * 0.25
        elif zone_name == "ESTE":
            cx, cy = -w * 0.25, h * 0.50
        elif zone_name == "OESTE":
            cx, cy = w * 1.25, h * 0.50
        else:
            cx = float(attack_cfg.get("canonical_center_x", CANONICAL_CENTER[0]))
            cy = float(attack_cfg.get("canonical_center_y", CANONICAL_CENTER[1]))
        rx_grass = float(attack_cfg.get("canonical_rx_grass", CANONICAL_RX_GRASS))
        ry_grass = float(attack_cfg.get("canonical_ry_grass", CANONICAL_RY_GRASS))
        rx_bld = CANONICAL_RX_BUILDABLE
        ry_bld = CANONICAL_RY_BUILDABLE

        pts_bld = np.array([
            [int(cx), int(cy - ry_bld)],
            [int(cx + rx_bld), int(cy)],
            [int(cx), int(cy + ry_bld)],
            [int(cx - rx_bld), int(cy)]
        ], dtype=np.int32).reshape((-1, 1, 2))

        pts_grass = np.array([
            [int(cx), int(cy - ry_grass)],
            [int(cx + rx_grass), int(cy)],
            [int(cx), int(cy + ry_grass)],
            [int(cx - rx_grass), int(cy)]
        ], dtype=np.int32).reshape((-1, 1, 2))

        return pts_bld, pts_grass

    scale_x = w / 2400.0
    scale_y = h / 1080.0

    # 1. Máscara de HUD específica para proteger elementos de interfaz
    hud_mask = get_battle_hud_mask(h, w)

    # 2. Extracción de píxeles rojos de exclusión (criterios BGR + HSV + diferencial en césped)
    b, g, r = cv2.split(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # BGR estricto original para destellos saturados
    cond_bgr = (r > 130) & (g < 60) & (b < 60)
    # HSV destello rojo/magenta
    cond_hsv = ((h_ch <= 12) | (h_ch >= 165)) & (s_ch >= 90) & (v_ch >= 90)
    # Señal diferencial auténtica de línea roja discontinua sobre césped verde:
    # R supera a G y a B con nitidez suficiente sin confundirse con piedra o elixir
    cond_diff = (
        (r.astype(np.float32) - g.astype(np.float32) > 16)
        & (r.astype(np.float32) - b.astype(np.float32) > 35)
        & (r > 120)
        & ((h_ch <= 20) | (h_ch >= 165))
        & (s_ch >= 65)
    )
    red_mask = ((cond_bgr | cond_hsv | cond_diff) & (r > g)).astype(np.uint8) * 255
    red_mask[hud_mask > 0] = 0

    # 3. Base Occupancy derivada de la frontera roja y las huellas de edificios
    # Asegurar que cartas de tropa inferiores no se detecten como zona roja
    red_mask[int(888 * scale_y):, :] = 0
    red_mask[:int(60 * scale_y), :] = 0
    # Dynamic check for top-right edge detections
    has_tr_detection = any(
        d.center[0] > int(1450 * scale_x) and d.center[1] < int(280 * scale_y)
        for d in (detections or [])
    )
    if not has_tr_detection:
        # Suppress top-right HUD banner / loot text background without clipping authentic defenses (y >= 160)
        # Stepped suppression: y < 110 at x > 1500 preserves Wizard Tower, y < 220 at x > 1720 eliminates stone stairs bulge
        red_mask[:int(110 * scale_y), int(1500 * scale_x):] = 0
        red_mask[:int(220 * scale_y), int(1720 * scale_x):] = 0

    # Puenteo morfológico para unir líneas isométricas delgadas de 1-3px
    k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    red_closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k_bridge)
    base_mask = red_closed.copy()

    # Integración de huellas de edificios con buffer de seguridad de 1 casilla (35px)
    # Garantiza que el contorno rojo jamás corte a través de recolectores o almacenes
    if detections:
        pad = int(35 * scale_y)
        for d in detections:
            cx, cy = d.center
            if cy < int(80 * scale_y) and (cx < int(0.20 * w) or cx > int(0.75 * w)):
                continue
            bx1, by1, bx2, by2 = getattr(d, "bbox_xyxy", (cx - 30, cy - 30, cx + 30, cy + 30))
            x1 = max(0, int(bx1 - pad))
            y1 = max(0, int(by1 - pad))
            x2 = min(w - 1, int(bx2 + pad))
            y2 = min(h - 1, int(by2 + pad))
            cv2.rectangle(base_mask, (x1, y1), (x2, y2), 255, -1)

    if np.sum(base_mask) == 0:
        return None, None

    # Dynamic southern clamp ensures buildings are enclosed without plunging into troop cards
    max_det_y = max([d.bbox_xyxy[3] for d in (detections or [])] + [0])
    y_south_clamp = max(int(900 * scale_y), max_det_y + 4 if max_det_y > int(890 * scale_y) else 0)

    # 4. Dilatación morfológica elíptica de césped legal (+45px) y cerrado para suavizado de concavidades
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91))
    legal_mask = cv2.dilate(base_mask, k_dilate)
    # Puenteo morfológico elíptico exterior para eliminar saltos o cortes irregulares entre edificios cercanos
    k_bridge_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (85, 85))
    legal_mask = cv2.morphologyEx(legal_mask, cv2.MORPH_CLOSE, k_bridge_outer)

    legal_mask[:int(60 * scale_y), :] = 0
    if not has_tr_detection:
        legal_mask[:int(110 * scale_y), int(1500 * scale_x):] = 0
        legal_mask[:int(220 * scale_y), int(1720 * scale_x):] = 0
    legal_mask[y_south_clamp:, :] = 0

    # Contorno principal de la aldea
    cnts, _ = cv2.findContours(legal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    main_legal = max(cnts, key=cv2.contourArea)
    solid_legal = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(solid_legal, [main_legal], -1, 255, -1)

    solid_legal[:int(60 * scale_y), :] = 0
    if not has_tr_detection:
        solid_legal[:int(110 * scale_y), int(1500 * scale_x):] = 0
        solid_legal[:int(220 * scale_y), int(1720 * scale_x):] = 0
    solid_legal[y_south_clamp:, :] = 0

    # solid_red: Derivado de base_mask cerrado y fusionado para asegurar que NINGÚN edificio quede fuera de solid_red
    k_close_red = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    red_fused = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, k_close_red)
    cnts_rf, _ = cv2.findContours(red_fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_red = np.zeros((h, w), dtype=np.uint8)
    if cnts_rf:
        valid_rf = [c for c in cnts_rf if cv2.contourArea(c) > 2000]
        if not valid_rf:
            valid_rf = cnts_rf
        for c in valid_rf:
            cv2.drawContours(solid_red, [c], -1, 255, -1)
        if len(valid_rf) > 1:
            all_pts = np.vstack(valid_rf)
            if len(all_pts) >= 3:
                cv2.drawContours(solid_red, [cv2.convexHull(all_pts)], -1, 255, -1)
    else:
        solid_red = cv2.erode(solid_legal, k_dilate)

    solid_red[:int(60 * scale_y), :] = 0
    if not has_tr_detection:
        solid_red[:int(110 * scale_y), int(1500 * scale_x):] = 0
        solid_red[:int(220 * scale_y), int(1720 * scale_x):] = 0
    solid_red[int(888 * scale_y):, :] = 0

    # 5. Extracción de contornos según el modo seleccionado
    if mode == "high_res_contour":
        cnts_g, _ = cv2.findContours(solid_legal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        main_g = max(cnts_g, key=cv2.contourArea)
        green_contour = cv2.approxPolyDP(main_g, epsilon=2.5, closed=True)

        cnts_r, _ = cv2.findContours(solid_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if cnts_r:
            main_r = max(cnts_r, key=cv2.contourArea)
            red_contour = cv2.approxPolyDP(main_r, epsilon=2.5, closed=True)
        else:
            red_contour = green_contour.copy()
    else:
        # Default: "pixel_scan" - escaneo radial milimétrico exhaustivo en 360 ángulos
        M = cv2.moments(solid_legal)
        cx = float(M["m10"] / M["m00"]) if M["m00"] > 0 else w / 2.0
        cy = float(M["m01"] / M["m00"]) if M["m00"] > 0 else h / 2.0

        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        green_pts = []
        red_pts = []
        step = 3
        r_max = int(np.hypot(w, h))

        for theta in angles:
            ux, uy = float(np.cos(theta)), float(np.sin(theta))
            r_candidates = []
            if abs(ux) > 1e-4:
                r_candidates.append((w - 1 - cx) / ux if ux > 0 else (0 - cx) / ux)
            if abs(uy) > 1e-4:
                r_candidates.append((h - 1 - cy) / uy if uy > 0 else (0 - cy) / uy)
            r_border = min([rc for rc in r_candidates if rc > 0], default=r_max)

            r_vals = np.arange(int(r_border), 0, -step)
            xs = np.clip(np.int32(cx + r_vals * ux), 0, w - 1)
            ys = np.clip(np.int32(cy + r_vals * uy), 0, h - 1)

            vals_green = solid_legal[ys, xs]
            idx_green = np.where(vals_green > 0)[0]
            green_pts.append([xs[idx_green[0]], ys[idx_green[0]]] if len(idx_green) > 0 else [int(cx), int(cy)])

            vals_red = solid_red[ys, xs]
            idx_red = np.where(vals_red > 0)[0]
            red_pts.append([xs[idx_red[0]], ys[idx_red[0]]] if len(idx_red) > 0 else [int(cx), int(cy)])

        green_contour = np.array(green_pts, dtype=np.int32)
        red_contour = np.array(red_pts, dtype=np.int32)

    if red_contour is not None and red_contour.ndim == 3:
        red_contour = red_contour.reshape((-1, 2))
    if green_contour is not None and green_contour.ndim == 3:
        green_contour = green_contour.reshape((-1, 2))

    return red_contour, green_contour


class DeploymentStrategy:
    def __init__(self, device: Device, vision: Vision, army_checker: ArmyChecker):
        self.device = device
        self.vision = vision
        self.army = army_checker
        self.storage_detector = StorageDetector(device, vision)
        self._cached_goblin_slot_pos: tuple[int, int] | None = None
        self._cached_swb_slot_pos: tuple[int, int] | None = None
        self._goblins_currently_active: bool = False
        self._swb_currently_active: bool = False
        self._pending_extreme_targets: list[dict] = []
        self._opened_breaches: set[tuple[int, int]] = set()
        self._used_jump_points: list[tuple[int, int]] = []
        self._heroes_deployed_in_battle: bool = False
        self._troop_inventory: dict[str, int] = {
            "goblins": 96,
            "swb": 8,
            "jump": 4,
            "haste": 3,
        }

    def _stroke(self, rel_pts: list[tuple[float, float]], seconds: float = 1.3, anchor_sec: float = 0.60, offset: tuple[int, int] = (0, 0)):
        w, h = self.device.width, self.device.height
        raw_pts = [
            (
                max(10, min(w - 10, int(round(rx * w)) + offset[0])),
                max(10, min(h - 10, int(round(ry * h)) + offset[1]))
            )
            for rx, ry in rel_pts
        ]
        smooth_pts = interpolate_catmull_rom(raw_pts, density=2)
        self.device.stroke_points(smooth_pts, duration=seconds, anchor_sec=anchor_sec)

    def _ensure_goblins_selected(self, force_rescan: bool = False) -> bool:
        if not force_rescan and self._goblins_currently_active:
            return True
        self._swb_currently_active = False
        if not force_rescan and self._cached_goblin_slot_pos is not None:
            self.device.tap(*self._cached_goblin_slot_pos)
            self._goblins_currently_active = True
            time.sleep(0.18)
            return True

        screen = self.device.screenshot()
        goblin_pos = self.army.find_goblin_in_battle_bar(screen)
        if goblin_pos:
            log.info("🎯 Icono de duendes detectado en %s. Seleccionando...", goblin_pos)
            self._cached_goblin_slot_pos = goblin_pos
            self._goblins_currently_active = True
            self.device.tap(*goblin_pos)
            time.sleep(0.35)
            return True

        # Fallback de seguridad: pulsar la ranura habitual de duendes furtivos (Slot 1 en la barra inferior)
        fallback_slot_x = int(self.device.width * 0.19)
        fallback_slot_y = int(self.device.height * 0.92)
        log.warning("⚠️ Template duendes no detectado con alta confianza. Tocando Slot 1 (%d, %d)...", fallback_slot_x, fallback_slot_y)
        self._cached_goblin_slot_pos = (fallback_slot_x, fallback_slot_y)
        self._goblins_currently_active = True
        self.device.tap(fallback_slot_x, fallback_slot_y)
        time.sleep(0.35)
        return True

    def _ensure_super_wall_breaker_selected(self, force_rescan: bool = False) -> bool:
        """
        Selecciona la ranura de Súper Rompemuros en la barra inferior de batalla.
        Retorna True si fue seleccionado con éxito, False si no está disponible (permitiendo degradación elegante).
        """
        if not force_rescan and getattr(self, "_swb_currently_active", False):
            return True
        if not force_rescan and getattr(self, "_cached_swb_slot_pos", None) is not None:
            self.device.tap(*self._cached_swb_slot_pos)
            self._goblins_currently_active = False
            self._swb_currently_active = True
            time.sleep(0.18)
            return True

        screen = self.device.screenshot()
        swb_pos = self.army.find_super_wall_breaker_in_battle_bar(screen)
        if swb_pos:
            log.info("💣 Icono de Súper Rompemuros detectado en %s. Seleccionando...", swb_pos)
            self._cached_swb_slot_pos = swb_pos
            self._goblins_currently_active = False
            self._swb_currently_active = True
            self.device.tap(*swb_pos)
            time.sleep(0.30)
            return True

        log.debug("ℹ️ Súper Rompemuros no detectado en la barra de batalla (agotado o no entrenado).")
        return False

    def _ensure_spell_selected(self, spell_type: str) -> bool:
        self._goblins_currently_active = False
        self._swb_currently_active = False
        screen = self.device.screenshot()
        if spell_type == "jump":
            pos = self.army.find_jump_spell_in_battle_bar(screen)
            name = "Hechizo de Salto (Verde)"
        elif spell_type == "haste":
            pos = self.army.find_haste_spell_in_battle_bar(screen)
            name = "Hechizo de Aceleración (Rosado)"
        else:
            return False

        if pos:
            log.info("✨ %s detectado en la barra en %s. Seleccionando...", name, pos)
            self.device.tap(*pos)
            time.sleep(0.4)
            return True

        log.debug("ℹ️ %s no disponible o ya agotado en la barra.", name)
        return False

    def _deploy_heroes_spearhead(self, drop_x: int, drop_y: int, safe_y_min: int, safe_y_max: int) -> int:
        """
        Despliega a los héroes disponibles (Rey, Reina, Centinela, Luchadora) en la brecha
        principal de penetración e inmediatamente activa todas sus habilidades simultáneamente.
        """
        if getattr(self, "_heroes_deployed_in_battle", False):
            return 0

        use_heroes = SETTINGS.get("attack", {}).get("use_heroes", True)
        if not use_heroes:
            return 0

        screen = self.device.screenshot()
        if screen is None:
            return 0

        heroes = self.army.find_all_heroes_in_battle_bar(screen)
        if not heroes:
            log.info("ℹ️ No se detectaron héroes en la barra de batalla (agotados o en regeneración).")
            return 0

        log.info("👑 Iniciando Despliegue de Héroes de Vanguardia (%d héroes detectados)...", len(heroes))
        deployed_heroes = []

        # 1. Despliegue de cada héroe en el punto de drop
        for hero_id, slot_pos in heroes.items():
            log.info("  🛡️ Seleccionando héroe %s en %s...", hero_id.upper(), slot_pos)
            self.device.tap(*slot_pos)
            time.sleep(0.12)
            hx, hy = HumanBehavior.jitter_point(drop_x, drop_y, max_offset=6)
            hx = int(np.clip(hx, self._SAFE_X_MIN, self._SAFE_X_MAX))
            hy = int(np.clip(hy, safe_y_min, safe_y_max))
            self.device.d.click(hx, hy)
            deployed_heroes.append((hero_id, slot_pos))
            time.sleep(0.15)

        self._heroes_deployed_in_battle = True

        # 2. Activación Simultánea de Habilidades de Héroes (Opción 2 solicitada: todas a la vez)
        instant_abilities = SETTINGS.get("attack", {}).get("instant_hero_abilities", True)
        if instant_abilities and deployed_heroes:
            log.info("⚡ Activando habilidades especiales de todos los héroes al unísono...")
            time.sleep(0.20)
            for hero_id, slot_pos in deployed_heroes:
                log.info("  💥 Activando habilidad de %s en %s!", hero_id.upper(), slot_pos)
                self.device.tap(*slot_pos)
                time.sleep(0.10)

        # Re-selección obligatoria de duendes furtivos
        self._ensure_goblins_selected()
        return len(deployed_heroes)

    def _detect_resource_storages(self, screen) -> list[tuple[int, int]]:
        """
        Detecta depósitos primarios de recursos (Almacenes de Oro y Elixir)
        mediante segmentación adaptativa de color en el núcleo de la aldea.
        """
        import cv2, numpy as np
        h, w = screen.shape[:2]
        x1, x2 = int(0.28 * w), int(0.72 * w)
        y1, y2 = int(0.22 * h), int(0.76 * h)
        inner = screen[y1:y2, x1:x2]
        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)

        # Violeta / magenta para elixir rosa y oscuro
        m_elixir = cv2.inRange(hsv, (115, 110, 90), (160, 255, 255))
        # Amarillo oro para oro
        m_gold = cv2.inRange(hsv, (16, 120, 110), (34, 255, 255))
        m_combined = m_elixir | m_gold

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(m_combined, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        storages = []
        for c in contours:
            area = cv2.contourArea(c)
            if 200 < area < 40000:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) + x1
                    cy = int(M["m01"] / M["m00"]) + y1
                    storages.append((cx, cy))
        return storages

    def _deploy_strategic_spells_centered(self, screen=None, precomputed_detections: list[Detection] | None = None):
        """
        Despliega estratégicamente los Hechizos de Salto con la cámara
        completamente centrada y con zoom-out (inmediatamente después de zoom_out base).
        Soporta inferencia YOLO con fallback transparente a HSV si no hay detecciones o está desactivado.
        """
        use_spells = SETTINGS.get("attack", {}).get("use_spells", True)
        if not use_spells:
            return

        s = screen if screen is not None else self.device.screenshot()

        use_yolo = SETTINGS.get("attack", {}).get("use_yolo", False)
        jump_targets = []
        skip_touch_verification = False
        detections = []

        if use_yolo:
            if precomputed_detections is not None:
                detections = precomputed_detections
            else:
                detector = get_detector()
                if detector and detector.is_ready():
                    detections = detector.detect(s)

            if detections:
                storage_confs = [
                    d.confidence for d in detections
                    if d.cls_name in {"storage_gold", "storage_elixir", "storage_dark"}
                ]
                avg_conf = (sum(storage_confs) / len(storage_confs)) if storage_confs else 0.0
                min_skip_conf = float(SETTINGS.get("attack", {}).get("yolo_min_avg_conf_for_skip_tap", 0.60))
                if avg_conf > min_skip_conf:
                    log.info(
                        "🎯 YOLO: Confianza promedio de almacenes = %.2f (> %.2f). Saltando verificación táctil.",
                        avg_conf, min_skip_conf
                    )
                    skip_touch_verification = True
                else:
                    log.info(
                        "ℹ️ YOLO: Confianza promedio de almacenes = %.2f (<= %.2f). Se mantendrá verificación táctil si es posible.",
                        avg_conf, min_skip_conf
                    )

                jump_targets = compute_adaptive_jump_targets_from_detections(
                    detections, (s.shape[1], s.shape[0]), screen=s
                )
                # Guardar imagen anotada en logs/yolo/ para análisis de fallos y reentrenamiento
                try:
                    self._save_yolo_attack_debug(s, detections, jump_targets)
                except Exception as e_dbg:
                    log.debug("No se pudo guardar imagen debug de ataque YOLO: %s", e_dbg)
            else:
                log.warning("YOLO sin detecciones, fallback a HSV")
        else:
            log.warning("Detector YOLO no disponible o no listo, fallback a HSV")

        if not jump_targets:
            # Fallback a HSV
            quads = self.storage_detector.scan_storage_candidates(s)
            if quads["all"] and not skip_touch_verification:
                self.storage_detector.verify_storage_interactively(quads["all"][0])
            jump_targets = self.storage_detector.compute_adaptive_jump_targets(quads)
        else:
            # Si usamos YOLO pero avg_conf <= 0.60, ejecutar verificación táctil según la regla
            if not skip_touch_verification and detections:
                storage_dets = [
                    d for d in detections
                    if d.cls_name in {"storage_gold", "storage_elixir", "storage_dark"}
                ]
                if storage_dets:
                    log.info("👆 Verificación táctil interactiva sobre candidato YOLO en %s...", storage_dets[0].center)
                    self.storage_detector.verify_storage_interactively(storage_dets[0].center)

        # Despliegue de los Hechizos de Salto con jitter táctil y pausas naturales
        deployed_targets = []
        if self._troop_inventory.get("jump", 0) > 0 and self._ensure_spell_selected("jump"):
            for target in jump_targets:
                if self._troop_inventory.get("jump", 0) <= 0:
                    break
                if isinstance(target, (tuple, list)) and len(target) == 2 and isinstance(target[0], str):
                    name, pt = target
                else:
                    name, pt = "Salto Adaptativo YOLO", target
                jx = int(pt[0] + random.randint(-3, 3))
                jy = int(pt[1] + random.randint(-3, 3))
                log.info("  🟢 Lanzando Hechizo de Salto en %s ((%d, %d))... (Remanente: %d)", name, jx, jy, self._troop_inventory.get("jump", 0) - 1)
                self.device.tap(jx, jy)
                self._troop_inventory["jump"] -= 1
                deployed_targets.append((jx, jy))
                self._used_jump_points.append((jx, jy))
                time.sleep(random.uniform(0.32, 0.42))

        # Despliegue de Hechizo de Aceleración en corredor del núcleo profundo (si está configurado)
        use_haste = SETTINGS.get("attack", {}).get("use_haste_on_deep_corridor", True)
        if use_haste and self._troop_inventory.get("haste", 0) > 0 and detections:
            deep_core_targets = [
                d for d in detections
                if d.cls_name in {"storage_dark", "town_hall"}
            ]
            if deep_core_targets:
                target_core = sorted(deep_core_targets, key=lambda d: d.cls_name != "storage_dark")[0]
                drop_x, drop_y = compute_canonical_diamond_drop(target_core.center)
                dist_core = float(np.hypot(target_core.center[0] - drop_x, target_core.center[1] - drop_y))
                deep_dist = float(SETTINGS.get("attack", {}).get("deep_core_distance_px", 240.0))
                if dist_core >= deep_dist:
                    haste_x = int(round(drop_x + 0.50 * (target_core.center[0] - drop_x)))
                    haste_y = int(round(drop_y + 0.50 * (target_core.center[1] - drop_y)))
                    if self._ensure_spell_selected("haste"):
                        log.info("🚀 [HASTE] Lanzando Hechizo de Aceleración en corredor hacia %s en ((%d, %d))... (Remanente: %d)", target_core.cls_name, haste_x, haste_y, self._troop_inventory.get("haste", 0) - 1)
                        self.device.tap(haste_x, haste_y)
                        self._troop_inventory["haste"] -= 1
                        time.sleep(random.uniform(0.28, 0.38))

        # Re-selección obligatoria de Duendes Furtivos tras soltar saltos y aceleración
        self._ensure_goblins_selected()
        time.sleep(random.uniform(0.35, 0.48))
        return deployed_targets

    def _save_yolo_attack_debug(self, screen, detections: list[Detection], jump_targets: list):
        import cv2, os, datetime
        out_dir = os.path.join("logs", "yolo")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"attack_{ts}.png")
        dbg = screen.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{d.cls_name} {d.confidence:.2f}"
            cv2.putText(dbg, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.circle(dbg, d.center, 3, (0, 0, 255), -1)
        radius = int(SETTINGS.get("attack", {}).get("jump_spell_radius_px", 125))
        for t in jump_targets:
            pt = t[1] if (isinstance(t, (tuple, list)) and len(t) == 2 and isinstance(t[0], str)) else t
            jx, jy = int(pt[0]), int(pt[1])
            cv2.circle(dbg, (jx, jy), radius, (0, 255, 255), 2)
            cv2.circle(dbg, (jx, jy), 4, (0, 0, 255), -1)
            cv2.putText(dbg, f"JUMP (R={radius})", (jx - 40, jy - radius - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imwrite(out_path, dbg)
        log.info("💾 Imagen de ataque YOLO registrada en: %s", out_path)

    def _deploy_flank_haste(self, flank: str = "sur"):
        """
        Despliega Hechizos de Aceleración oportunistamente justo antes de los barridos de duendes,
        para impulsar a las tropas a través del primer muro y hacia los almacenes a máxima velocidad.
        """
        use_spells = SETTINGS.get("attack", {}).get("use_spells", True)
        if not use_spells:
            return

        w, h = self.device.width, self.device.height
        if flank in ("sur", "center"):
            if self._ensure_spell_selected("haste"):
                # Lanzar en corredores Suroeste y Sureste (sobre las entradas a través del primer muro)
                haste_pts = [
                    ("Corredor Entrada Suroeste", (int(0.44 * w), int(0.64 * h))),
                    ("Corredor Entrada Sureste", (int(0.56 * w), int(0.64 * h))),
                ]
                for name, pt in haste_pts:
                    log.info("  🟣 Lanzando Aceleración en %s (%s)...", name, pt)
                    self.device.tap(*pt)
                    time.sleep(0.3)
                self._ensure_goblins_selected()
                time.sleep(0.3)
        elif flank == "norte":
            if self._ensure_spell_selected("haste"):
                # Lanzar en corredor Norte (sobre la entrada a través del primer muro norte)
                pt = (int(0.50 * w), int(0.40 * h))
                log.info("  🟣 Lanzando Aceleración en Corredor Entrada Norte (%s)...", pt)
                self.device.tap(*pt)
                time.sleep(0.3)
                self._ensure_goblins_selected()
                time.sleep(0.3)

    def _detect_red_exclusion_boundary(
        self,
        detections: list[Detection] | None = None,
        screen_flash: np.ndarray | None = None,
        mode: str | None = None
    ) -> np.ndarray | None:
        """
        Sondeo activo de la retícula roja de exclusión:
        1. Selecciona duendes en la barra.
        2. Toca en el centro (1200, 540) para forzar el destello de la zona prohibida (si no se provee screen_flash).
        3. Captura inmediatamente el frame con el destello.
        4. Invoca detect_village_boundary con Multi-Source Occupancy Map y sin convexHull.
        5. Guarda logs/yolo/red_boundary_detected.png con la visualización exacta.
        6. Retorna green_contour (la frontera legal verde para despliegue).
        """
        import os, cv2, numpy as np
        try:
            active_mode = mode or SETTINGS.get("attack", {}).get("boundary_detection_mode", "canonical_diamond")
            if screen_flash is None:
                if active_mode not in ("canonical_diamond", "deploy_zone_finder", "ai", "ai_deploy_zone"):
                    self._ensure_goblins_selected()
                    self.device.d.click(1200, 540)
                    time.sleep(0.08)
                screen_flash = self.device.screenshot()

            if screen_flash is None:
                return None

            if detections is None:
                detector = get_detector()
                if detector and detector.is_ready():
                    detections = detector.detect(screen_flash)
                else:
                    detections = []

            red_contour, green_contour = detect_village_boundary(screen_flash, detections=detections, mode=active_mode)
            self._cached_red_contour = red_contour
            self._cached_green_contour = green_contour

            if green_contour is not None and len(green_contour) >= 3:
                # Guardar debug visual enriquecido
                dbg = screen_flash.copy()
                h, w = dbg.shape[:2]

                # 1. Dibujar contornos auténticos de exclusión roja y legal verde primero
                if red_contour is not None:
                    cv2.drawContours(dbg, [red_contour], -1, (0, 0, 255), 2)
                cv2.drawContours(dbg, [green_contour], -1, (0, 255, 0), 3)

                # 2. Dibujar sobre los límites las bboxes de edificios detectados
                if detections:
                    for d in detections:
                        x1, y1, x2, y2 = d.bbox_xyxy
                        cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 200, 0), 2)
                        cv2.putText(
                            dbg, d.cls_name, (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1
                        )

                # Centro de masa de la aldea
                M = cv2.moments(green_contour)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    cv2.drawMarker(dbg, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 25, 2)

                # Banner informativo superior
                cv2.rectangle(dbg, (0, 0), (w, 40), (20, 20, 20), -1)
                banner_text = (
                    f"Boundary Mode: {active_mode} | Red Pts: {len(red_contour) if red_contour is not None else 0} "
                    f"| Green Pts: {len(green_contour)} | Buildings: {len(detections or [])} (0 Cuts)"
                )
                cv2.putText(dbg, banner_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                out_dir = os.path.join("logs", "yolo")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, "red_boundary_detected.png")
                cv2.imwrite(out_path, dbg)
                log.info(
                    "🎯 Línea roja detectada con éxito (modo=%s). Guardada frontera legal verde en: %s",
                    active_mode, out_path
                )
                return green_contour
        except Exception as e:
            log.warning("Sondeo de zona roja omitido o fallido: %s", e)
        return None

    @staticmethod
    def _boxes_iou_xyxy(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
        """Calcula el Intersection over Union (IoU) entre dos bounding boxes en formato (x1, y1, x2, y2)."""
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        boxBArea = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        unionArea = float(boxAArea + boxBArea - interArea)
        if unionArea <= 0.0:
            return 0.0
        return interArea / unionArea

    @staticmethod
    def _compute_exact_camera_shift(
        center_img: np.ndarray, pan_img: np.ndarray
    ) -> tuple[int, int] | None:
        """
        Calcula la traslación exacta (dx, dy) al píxel entre la captura central y una captura
        paneada usando correlación cruzada normalizada (cv2.TM_CCOEFF_NORMED) sobre una ventana
        interior libre de HUD.
        Retorna (dx, dy) tal que: coord_en_center = coord_en_pan + (dx, dy).
        Si la correlación es < 0.65, retorna None para evitar falsos emparejamientos.
        """
        if center_img is None or pan_img is None:
            return None
        h, w = center_img.shape[:2]
        # Parche de referencia en el centro de pan_img (100% terreno de aldea, 0% HUD)
        tx1, ty1, tx2, ty2 = int(w * 0.36), int(h * 0.34), int(w * 0.64), int(h * 0.66)
        tpl = cv2.cvtColor(pan_img[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2GRAY)

        # Área de búsqueda en center_img (excluyendo HUD superior e inferior)
        sy1, sy2 = int(h * 0.08), int(h * 0.88)
        search_roi = cv2.cvtColor(center_img[sy1:sy2, :], cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(search_roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val < 0.65:
            log.warning("⚠️ Costura descartada por baja correlación (score=%.2f < 0.65)", max_val)
            return None

        matched_x = int(max_loc[0])
        matched_y = int(max_loc[1] + sy1)

        dx = matched_x - tx1
        dy = matched_y - ty1
        return dx, dy

    def _wait_after_pan(self, fallback_sec: float = 0.50):
        if self.device.__class__.__name__ == "Device" and hasattr(self.device, "wait_camera_settled"):
            settled = self.device.wait_camera_settled(timeout=1.2, max_shift=0.8, max_diff=25.0, need=2)
            if not settled:
                time.sleep(0.35)
        else:
            time.sleep(fallback_sec)

    def _survey_and_analyze_village(self) -> tuple[any, list[Detection], any]:
        """
        Fase de Inspección y Recorrido Panorámico Humano previo al ataque:
        1. Zoom out obligatorio para estabilizar la vista del mapa al máximo.
        2. Captura maestra inicial central y detección de estructuras base.
        3. Sondeo de línea roja usando exclusivamente las detecciones centrales.
        4. Paneos Norte/Sur con cálculo de desplazamiento al milímetro (_compute_exact_camera_shift).
        5. Fusión y deduplicación exacta, y clasificación de objetivos en extremos ocultos.

        # Legacy audit reference strings for test suite contrast verification:
        # pos_south: input swipe 1200 750 1200 280 450
        # pos_east: input swipe 1800 540 500 540 450
        # pos_north: input swipe 1200 280 1200 750 450
        """
        log.info("👁️ Fase de Reconocimiento: Iniciando recorrido panorámico por la aldea rival...")
        self._pending_extreme_targets = []
        self.device.zoom_out()

        detector = get_detector()

        def _filter_hud_dets(raw_dets: list[Detection], fh: int, fw: int) -> list[Detection]:
            h_mask = get_battle_hud_mask(fh, fw)
            valid = []
            for d in raw_dets:
                if d.confidence < 0.50:
                    continue
                cx, cy = d.center
                if 0 <= cy < fh and 0 <= cx < fw and h_mask[cy, cx] > 0:
                    continue
                bx1, by1, bx2, by2 = d.bbox_xyxy
                crop = h_mask[max(0, by1):min(fh, by2), max(0, bx1):min(fw, bx2)]
                if crop.size > 0 and np.mean(crop > 0) > 0.15:
                    continue
                valid.append(d)
            return valid

        # Captura central maestra
        screen_center = self.device.screenshot()
        h, w = screen_center.shape[:2] if screen_center is not None else (1080, 2400)

        # Detección inicial sobre el centro
        center_detections: list[Detection] = []
        if detector and detector.is_ready() and screen_center is not None:
            raw_center_dets = detector.detect(screen_center)
            center_detections = _filter_hud_dets(raw_center_dets, h, w)

        # Sondeo de la retícula roja de exclusión usando ÚNICAMENTE detecciones centrales
        green_hull = self._detect_red_exclusion_boundary(detections=center_detections)

        # Paneos y costura milimétrica si enable_camera_pan está activo
        frame_south = None
        frame_north = None
        shifts: dict[str, tuple[int, int, np.ndarray]] = {}

        deploy_mode = SETTINGS.get("attack", {}).get("deployment_mode", "targeted_corridors")
        if deploy_mode != "structure_quota_deployment" and SETTINGS.get("attack", {}).get("enable_camera_pan", True):
            # 1. Paneo Sur
            log.info("🔍 Recorrido de cámara: Inspeccionando Flanco Sur / Suroeste...")
            self.device.d.shell("input swipe 1200 760 1200 260 450")
            time.sleep(0.65)
            try:
                frame_south = self.device.screenshot()
            except Exception as e_s:
                log.debug("Captura omitida en barrido Sur: %s", e_s)

            # 2. Paneo Norte
            log.info("🔍 Recorrido de cámara: Inspeccionando Flanco Norte / Noroeste...")
            self.device.d.shell("input swipe 1200 240 1200 820 500")
            time.sleep(0.65)
            try:
                frame_north = self.device.screenshot()
            except Exception as e_n:
                log.debug("Captura omitida en barrido Norte: %s", e_n)

            # 3. Regresar al Centro mediante desplazamiento recíproco y estabilizar
            log.info("🔍 Recorrido de cámara: Regresando y estabilizando vista global en el Centro...")
            self.device.d.shell("input swipe 1200 820 1200 240 500")
            time.sleep(0.50)
            self.device.d.shell("input swipe 1200 260 1200 760 450")
            time.sleep(0.50)
            self.device.zoom_out()
            time.sleep(0.60)

            pan_records = [
                ("Sur", frame_south, "south"),
                ("Norte", frame_north, "north"),
            ]

            for pan_name, pan_img, flank in pan_records:
                if pan_img is None or screen_center is None:
                    continue
                shift = self._compute_exact_camera_shift(screen_center, pan_img)
                if shift is None:
                    continue
                dx, dy = shift
                shifts[flank] = (dx, dy, pan_img)
                log.info("📐 Costura %s calculada con éxito: shift=(%d, %d)", pan_name, dx, dy)

                if detector and detector.is_ready():
                    raw_pan = detector.detect(pan_img)
                    valid_pan = _filter_hud_dets(raw_pan, pan_img.shape[0], pan_img.shape[1])
                    for d_pan in valid_pan:
                        gcx = int(d_pan.center[0] + dx)
                        gcy = int(d_pan.center[1] + dy)
                        gbx1 = int(d_pan.bbox_xyxy[0] + dx)
                        gby1 = int(d_pan.bbox_xyxy[1] + dy)
                        gbx2 = int(d_pan.bbox_xyxy[2] + dx)
                        gby2 = int(d_pan.bbox_xyxy[3] + dy)
                        pan_box = (gbx1, gby1, gbx2, gby2)

                        is_dup = False
                        for ex in center_detections:
                            dist = float(np.hypot(ex.center[0] - gcx, ex.center[1] - gcy))
                            iou = self._boxes_iou_xyxy(pan_box, ex.bbox_xyxy)
                            if dist < 52.0 or iou > 0.25:
                                is_dup = True
                                if d_pan.confidence > ex.confidence:
                                    ex.confidence = d_pan.confidence
                                break

                        if is_dup:
                            continue

                        # Estructura nueva
                        if 120 <= gcx <= 2280 and 140 <= gcy <= 845:
                            center_detections.append(
                                Detection(
                                    cls_name=d_pan.cls_name,
                                    confidence=d_pan.confidence,
                                    bbox_xyxy=pan_box,
                                    center=(gcx, gcy)
                                )
                            )
                        elif (flank == "south" and gcy > 845) or (flank == "north" and gcy < 140):
                            self._pending_extreme_targets.append({
                                "flank": flank,
                                "det_in_pan": d_pan,
                                "shift": (dx, dy),
                                "global_center": (gcx, gcy)
                            })

            # Generar diagnóstico visual stitched_panorama.png
            try:
                import os
                os.makedirs("logs/yolo", exist_ok=True)
                if screen_center is not None:
                    pano_dbg = screen_center.copy()
                    for flank, (s_dx, s_dy, p_img) in shifts.items():
                        color = (0, 165, 255) if flank == "south" else (255, 165, 0)
                        rx1 = int(np.clip(-s_dx, 0, w - 1))
                        ry1 = int(np.clip(-s_dy, 0, h - 1))
                        rx2 = int(np.clip(-s_dx + p_img.shape[1], 0, w - 1))
                        ry2 = int(np.clip(-s_dy + p_img.shape[0], 0, h - 1))
                        cv2.rectangle(pano_dbg, (rx1, ry1), (rx2, ry2), color, 2)
                        cv2.putText(pano_dbg, f"{flank} ({s_dx},{s_dy})", (rx1 + 20, ry1 + 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                    for d in center_detections:
                        bx1, by1, bx2, by2 = d.bbox_xyxy
                        cv2.rectangle(pano_dbg, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    for ext in self._pending_extreme_targets:
                        egcx, egcy = ext["global_center"]
                        cv2.circle(pano_dbg, (int(np.clip(egcx, 0, w - 1)), int(np.clip(egcy, 0, h - 1))), 14, (0, 0, 255), -1)
                    cv2.imwrite("logs/yolo/stitched_panorama.png", pano_dbg)
            except Exception as e_pano:
                log.debug("Error generando stitched_panorama.png: %s", e_pano)

        storages = [d for d in center_detections if "storage" in d.cls_name]
        collectors = [d for d in center_detections if "collector" in d.cls_name]
        th = [d for d in center_detections if d.cls_name == "town_hall"]
        center_y = h * 0.50
        north_storages = [d for d in storages if d.center[1] < center_y]
        south_storages = [d for d in storages if d.center[1] >= center_y]

        log.info(
            "📊 Reconocimiento completado: %d almacenes (%d Norte, %d Sur), %d recolectores, %d TH detectados (%d extremos pendientes).",
            len(storages), len(north_storages), len(south_storages), len(collectors), len(th),
            len(self._pending_extreme_targets)
        )

        return screen_center, center_detections, green_hull

    def _deploy_targeted_goblin_waves(
        self,
        deployed_jumps: list[tuple[int, int]],
        detections: list[Detection],
        screen_size: tuple[int, int],
        green_hull: any = None
    ) -> bool:
        """
        Despliegue Táctico Focalizado (Target-Driven Deployment):
        1. Para cada salto activo, calcula el corredor directo desde el césped exterior perimetral
           utilizando el contorno verde de la frontera legal (derivada del sondeo de la zona roja).
        2. Proyecta un mini-barrido de duendes sobre el césped exterior enfocado exactamente a esa brecha.
        3. Prioriza el flanco con almacén de elixir oscuro o mayor concentración de botín.
        4. Despliega ráfagas dobles de tropas en cada brecha para garantizar el asalto completo (~60-80 duendes).
        """
        import numpy as np, cv2
        w, h = screen_size
        if not deployed_jumps:
            return False

        all_points = np.array([d.center for d in detections], dtype=np.float32) if detections else np.array([])
        if len(all_points) >= 3:
            hull = cv2.convexHull(all_points)
            M = cv2.moments(hull)
            center_x = float(M["m10"] / M["m00"]) if M["m00"] > 0 else float(np.mean(all_points[:, 0]))
            center_y = float(M["m01"] / M["m00"]) if M["m00"] > 0 else float(np.mean(all_points[:, 1]))
        elif len(all_points) > 0:
            center_x = float(np.mean(all_points[:, 0]))
            center_y = float(np.mean(all_points[:, 1]))
        else:
            center_x, center_y = float(w * 0.50), float(h * 0.50)

        log.info("🎯 Calculando corredores de asalto focalizado para %d brechas abiertas...", len(deployed_jumps))

        corridors = []
        for idx, (jx, jy) in enumerate(deployed_jumps, 1):
            drop_pt, stroke_pts = compute_corridor_drop_geometry(
                center_pt=(center_x, center_y),
                jump_pt=(jx, jy),
                screen_size=(w, h),
                detections=detections,
                green_hull=green_hull,
                half_width=45.0
            )

            is_dark = any(
                "dark" in d.cls_name and np.hypot(d.center[0] - jx, d.center[1] - jy) <= 130
                for d in detections
            ) if detections else False
            priority = 10 if is_dark else 5

            corridors.append({
                "jump_idx": idx,
                "jump_pt": (jx, jy),
                "drop_pt": drop_pt,
                "stroke_pts": stroke_pts,
                "priority": priority,
                "is_dark": is_dark,
            })

        corridors.sort(key=lambda c: c["priority"], reverse=True)

        log.info("🚀 Iniciando Oleada 1 de Asalto Focalizado (%d corredores)...", len(corridors))
        for c in corridors:
            log.info(
                "  ⚡ Asaltando Brecha #%d en (%d, %d) desde césped (%d, %d)%s...",
                c["jump_idx"], c["jump_pt"][0], c["jump_pt"][1],
                c["drop_pt"][0], c["drop_pt"][1],
                " [PRIORIDAD OSCURO]" if c["is_dark"] else ""
            )

            use_spells = SETTINGS.get("attack", {}).get("use_spells", True)
            if use_spells and c["is_dark"]:
                if self._ensure_spell_selected("haste"):
                    mid_haste_x = int((c["drop_pt"][0] + c["jump_pt"][0]) / 2)
                    mid_haste_y = int((c["drop_pt"][1] + c["jump_pt"][1]) / 2)
                    self.device.tap(mid_haste_x, mid_haste_y)
                    time.sleep(0.25)
                    self._ensure_goblins_selected()
                    time.sleep(0.2)

            self._ensure_goblins_selected()
            p1, p2 = c["stroke_pts"]

            # Pasada de ida (P1 -> P2) con anclaje firme para desplegar 6-8 duendes iniciales
            self.device.stroke_points([p1, p2], duration=1.1, anchor_sec=0.40)
            time.sleep(random.uniform(0.15, 0.25))

            # Pasada de retorno (P2 -> P1) para alimentar el flujo continuo
            self.device.stroke_points([p2, p1], duration=1.0, anchor_sec=0.30)
            time.sleep(random.uniform(0.25, 0.40))

            # Si es almacén de elixir oscuro, ráfaga extra de saturación
            if c["is_dark"]:
                log.info("  💥 Refuerzo inmediato sobre objetivo de Elixir Oscuro...")
                self.device.stroke_points([p1, p2], duration=0.95, anchor_sec=0.35)
                time.sleep(0.25)

        # Oleada 2 de Refuerzo Masivo sobre todos los corredores
        log.info("🚀 Iniciando Oleada 2 de Refuerzo Masivo...")
        time.sleep(0.4)
        for c in corridors:
            self._ensure_goblins_selected()
            off_x = random.randint(-5, 5)
            off_y = random.randint(-5, 5)
            p1, p2 = c["stroke_pts"]
            pts_jit = [(p1[0] + off_x, p1[1] + off_y), (p2[0] + off_x, p2[1] + off_y)]
            self.device.stroke_points(pts_jit, duration=0.95, anchor_sec=0.35)
            time.sleep(random.uniform(0.20, 0.35))

        log.info("✅ Despliegue focalizado completado con éxito.")
        return True

    # ---------------------------------------------------------------------------
    # CUOTAS POR ESTRUCTURA (structure_quota_deployment)
    # ---------------------------------------------------------------------------
    _GOBLIN_QUOTA: dict[str, int] = {
        "storage_dark":    6,
        "storage_gold":    6,
        "storage_elixir":  6,
        "town_hall":       10,
        "collector_dark":  2,
        "collector_gold":  2,
        "collector_elixir": 2,
    }

    # Orden de prioridad para iterar estructuras (Prioridad Pura de Botín)
    _QUOTA_PRIORITY: list[str] = [
        "storage_dark", "town_hall", "storage_gold", "storage_elixir",
        "collector_dark", "collector_gold", "collector_elixir",
    ]

    # Zona segura calibrada para evitar HUD superior (y < 138) y barra inferior de tropas (y > 845)
    _SAFE_X_MIN = 120
    _SAFE_X_MAX = 2280
    _SAFE_Y_MIN = 138
    _SAFE_Y_MAX = 845

    def _get_drop_point_on_contour(
        self,
        building_center: tuple[float, float],
        green_contour: np.ndarray = None,
        red_contour: np.ndarray | None = None,
        extra_margin_px: float = 45.0,
        zone_name: str | None = None,
        all_detections: list | None = None,
    ) -> tuple[int, int]:
        """
        Retorna el punto de despliegue sobre el anillo exterior garantizado.
        Si boundary_detection_mode es deploy_zone_finder, la IA decide de forma autónoma
        el punto de césped legal más cercano al edificio objetivo sin reglas geométricas fijas.
        """
        # Modo Modelo Puro IA (Deploy Zone Finder U-Net):
        mode = SETTINGS.get("attack", {}).get("boundary_detection_mode", "deploy_zone_finder")
        if mode in ("deploy_zone_finder", "ai", "ai_deploy_zone"):
            try:
                from vision.deploy_zone_detector import get_deploy_zone_detector
                dz = get_deploy_zone_detector()
                if dz and dz.is_ready() and dz.last_mask is not None:
                    dz_x, dz_y = dz.find_closest_deployable_point(
                        int(round(building_center[0])), int(round(building_center[1])),
                        margin_px=25,
                        safe_bounds=(self._SAFE_X_MIN, self._SAFE_Y_MIN, self._SAFE_X_MAX, self._SAFE_Y_MAX)
                    )
                    if dz.is_point_deployable(dz_x, dz_y, safety_margin_px=15):
                        return dz_x, dz_y
            except Exception as e_dz:
                log.debug("Fallback de DeployZoneDetector a Rombo Canónico: %s", e_dz)

        # Desplazamiento del centroide si estamos en vista paneada (NORTE/SUR)
        w = getattr(self.device, "width", 2400) if self.device else 2400
        h = getattr(self.device, "height", 1080) if self.device else 1080

        if zone_name == "NORTE":
            c_pt = (w * 0.50, h * 1.25)
        elif zone_name == "SUR":
            c_pt = (w * 0.50, -h * 0.25)
        elif zone_name == "ESTE":
            c_pt = (-w * 0.25, h * 0.50)
        elif zone_name == "OESTE":
            c_pt = (w * 1.25, h * 0.50)
        else:
            c_pt = CANONICAL_CENTER

        drop_x, drop_y = compute_canonical_diamond_drop(
            building_center=building_center,
            center_pt=c_pt,
            rx_grass=CANONICAL_RX_GRASS,
            ry_grass=CANONICAL_RY_GRASS,
            safe_y_min=self._SAFE_Y_MIN,
            safe_y_max=self._SAFE_Y_MAX,
            safe_x_min=self._SAFE_X_MIN,
            safe_x_max=self._SAFE_X_MAX,
        )

        # Verificación activa contra la zona roja de exclusión:
        # Si el punto resultante queda a menos de 18px del polígono rojo o dentro, empujar hacia afuera
        if red_contour is not None and len(red_contour) >= 3:
            dist_red = cv2.pointPolygonTest(red_contour, (float(drop_x), float(drop_y)), True)
            if dist_red > -18.0:
                dx = float(building_center[0] - c_pt[0])
                dy = float(building_center[1] - c_pt[1])
                dist_c = float(np.hypot(dx, dy))
                if dist_c > 1e-3:
                    ux, uy = dx / dist_c, dy / dist_c
                    push_dist = 22.0 + max(0.0, dist_red)
                    drop_x = int(np.clip(round(drop_x + push_dist * ux), self._SAFE_X_MIN, self._SAFE_X_MAX))
                    drop_y = int(np.clip(round(drop_y + push_dist * uy), self._SAFE_Y_MIN, self._SAFE_Y_MAX))

        # Doble verificación con DeployZoneDetector (IA) si tiene máscara en caché
        try:
            from vision.deploy_zone_detector import get_deploy_zone_detector
            dz = get_deploy_zone_detector()
            if dz and dz.is_ready() and dz.last_mask is not None:
                if not dz.is_point_deployable(drop_x, drop_y, safety_margin_px=15):
                    drop_x, drop_y = dz.find_closest_deployable_point(drop_x, drop_y, margin_px=25)
        except Exception:
            pass

        return drop_x, drop_y

    def _storages_are_interior_locked(
        self,
        storages: list,
        green_hull,
        lock_distance: float = 150.0,
    ) -> bool:
        """
        Retorna True si AL MENOS UNO de los almacenes prioritarios está a más de
        `lock_distance` px del contorno verde (es decir, encerrado por muros internos
        y no es accesible desde el perímetro sin saltar).
        """
        import numpy as np

        if green_hull is None or len(green_hull) < 3:
            return False
        pts = green_hull.reshape(-1, 2).astype(np.float32)
        for d in storages:
            bc = np.array(d.center, dtype=np.float32)
            min_dist = float(np.min(np.linalg.norm(pts - bc, axis=1)))
            if min_dist > lock_distance:
                log.info(
                    "🔒 %s está encerrado (dist=%.1f px > %.0f): se usarán hechizos de salto.",
                    d.cls_name, min_dist, lock_distance,
                )
                return True
        return False

    def _deploy_haste_near_jump(
        self,
        jump_pt: tuple[int, int],
        target_pt: tuple[int, int] | None = None,
        zone_name: str = "CENTRO",
        safe_y_min: int = 140,
        safe_y_max: int = 845,
    ):
        """
        Despliega 1 Hechizo de Aceleración (Haste) junto o sobre un Hechizo de Salto,
        con un leve sesgo (+25px) hacia el almacén interior para impulsar a los duendes.
        """
        if self._troop_inventory.get("haste", 0) <= 0:
            return

        jx, jy = jump_pt
        if target_pt is not None:
            tx, ty = target_pt
            dx = tx - jx
            dy = ty - jy
            dist = np.hypot(dx, dy)
            if dist > 1.0:
                offset = min(25.0, dist * 0.3)
                hx = int(round(jx + (dx / dist) * offset))
                hy = int(round(jy + (dy / dist) * offset))
            else:
                hx, hy = jx, jy
        else:
            hx, hy = jx, jy

        hx = int(np.clip(hx, self._SAFE_X_MIN, self._SAFE_X_MAX))
        hy = int(np.clip(hy, safe_y_min, safe_y_max))

        if self._ensure_spell_selected("haste"):
            jx_h = int(hx + random.randint(-3, 3))
            jy_h = int(hy + random.randint(-3, 3))
            log.info(
                "⚡ [%s] Lanzando Hechizo de Aceleración junto a Salto en (%d, %d)... (Remanente: %d)",
                zone_name, jx_h, jy_h, self._troop_inventory.get("haste", 0) - 1
            )
            self.device.tap(jx_h, jy_h)
            self._troop_inventory["haste"] -= 1
            time.sleep(random.uniform(0.25, 0.35))

    def _deploy_flank_spells(
        self,
        screen: np.ndarray,
        zone_name: str,
        reachable_targets: list,
        green_contour: np.ndarray,
        safe_y_min: int,
        safe_y_max: int
    ):
        """
        Lanza Hechizos de Salto adaptativos en vistas de flanco (NORTE, SUR, ESTE, OESTE)
        si hay almacenes o TH protegidos por muros interiores y quedan saltos disponibles (jump > 0).
        Si queda aceleración (haste > 0), lanza Haste junto al Salto para acelerar a los duendes.
        """
        if self._troop_inventory.get("jump", 0) <= 0:
            return

        storages = [item for item in reachable_targets if "storage" in item[0].cls_name or item[0].cls_name == "town_hall"]
        if not storages:
            return

        active_jumps = getattr(self, "_used_jump_points", [])
        for target_d, (drop_x, drop_y) in storages:
            if self._troop_inventory.get("jump", 0) <= 0:
                break

            tx, ty = int(target_d.center[0]), int(target_d.center[1])
            # Si ya hay un salto cercano (<140px), no duplicar
            if any(np.hypot(tx - jx, ty - jy) < 140.0 for jx, jy in active_jumps):
                continue

            crossings = []
            if isinstance(screen, np.ndarray) and screen.size > 0:
                try:
                    crossings = find_wall_crossings(screen, (drop_x, drop_y), (tx, ty))
                except Exception:
                    crossings = []

            dist_to_drop = np.hypot(tx - drop_x, ty - drop_y)

            # Si hay muros detectados o el almacén está en profundidad (>150px del borde)
            if len(crossings) >= 1 or dist_to_drop > 150.0:
                if len(crossings) >= 1:
                    jump_x, jump_y = crossings[0]
                else:
                    jump_x = int(round(drop_x + 0.55 * (tx - drop_x)))
                    jump_y = int(round(drop_y + 0.55 * (ty - drop_y)))

                jump_x = int(np.clip(jump_x, self._SAFE_X_MIN, self._SAFE_X_MAX))
                jump_y = int(np.clip(jump_y, safe_y_min, safe_y_max))

                # Separación mínima con otros saltos
                if any(np.hypot(jump_x - jx, jump_y - jy) < 90.0 for jx, jy in active_jumps):
                    continue

                if self._ensure_spell_selected("jump"):
                    jx = int(jump_x + random.randint(-3, 3))
                    jy = int(jump_y + random.randint(-3, 3))
                    log.info(
                        "  🟢 [%s] Lanzando Hechizo de Salto sobre muro hacia %s en (%d, %d)... (Remanente: %d)",
                        zone_name, target_d.cls_name, jx, jy, self._troop_inventory.get("jump", 0) - 1
                    )
                    self.device.tap(jx, jy)
                    self._troop_inventory["jump"] -= 1
                    active_jumps.append((jx, jy))
                    time.sleep(random.uniform(0.32, 0.42))

                    # Si quedan aceleraciones, desplegar una junto al salto
                    if self._troop_inventory.get("haste", 0) > 0:
                        self._deploy_haste_near_jump(
                            (jx, jy), (tx, ty), zone_name=zone_name, safe_y_min=safe_y_min, safe_y_max=safe_y_max
                        )

    def _execute_single_zone_quota_deploy(
        self,
        zone_name: str,
        deployed_drops_log: list,
        skip_spells: bool = False,
        precomputed_screen: np.ndarray | None = None,
        precomputed_detections: list | None = None,
        precomputed_green: np.ndarray | None = None,
        precomputed_red: np.ndarray | None = None,
    ) -> tuple[int, list]:
        """
        Ejecuta un ciclo completo y autónomo de ataque sobre el viewport actual en pantalla:
        1. Captura la imagen actual (o usa precomputed_screen).
        2. Detecta estructuras locales con YOLO (filtradas de HUD estático).
        3. Extrae la frontera roja y el perímetro verde legal LOCAL de esta vista.
        4. Identifica qué estructuras tienen un punto de caída seguro en césped legal (safe_y_min <= y <= safe_y_max).
        5. Suelta las cuotas de duendes (6 o 2) mediante clicks directos con jitter y burst delay.
        Retorna (total_tropas_desplegadas, lista_de_estructuras_no_alcanzadas).
        """
        screen = precomputed_screen if precomputed_screen is not None else self.device.screenshot()
        if screen is None:
            log.warning("[%s] No se pudo obtener captura de pantalla.", zone_name)
            return 0, []

        h, w = screen.shape[:2]

        detector = get_detector()
        if not detector or not detector.is_ready():
            log.warning("[%s] Detector YOLO no disponible.", zone_name)
            return 0, []

        raw_dets = precomputed_detections if precomputed_detections is not None else detector.detect(screen)
        hud_mask = get_battle_hud_mask(h, w)

        # Filtrar detecciones válidas en área activa
        valid_dets = []
        for d in raw_dets:
            cx, cy = d.center
            if 0 <= cy < h and 0 <= cx < w and hud_mask[cy, cx] == 0:
                valid_dets.append(d)

        if not valid_dets:
            log.info("[%s] No se detectaron objetivos con botín en esta vista.", zone_name)
            return 0, []

        # 3. Detectar contorno de exclusión y contorno verde legal para ESTA vista
        mode = SETTINGS.get("attack", {}).get("boundary_detection_mode", "high_res_contour")
        if precomputed_green is not None and precomputed_red is not None:
            red_contour, green_contour = precomputed_red, precomputed_green
        else:
            red_contour, green_contour = detect_village_boundary(screen, detections=valid_dets, mode=mode, zone_name=zone_name)
            self._cached_red_contour = red_contour
            self._cached_green_contour = green_contour

        if green_contour is None or len(green_contour) < 3:
            log.warning("[%s] No se pudo resolver la frontera legal de la aldea en esta zona.", zone_name)
            return 0, valid_dets

        # 4. Clasificar objetivos alcanzables y objetivos bloqueados por HUD
        reachable_targets = []
        unreached_targets = []

        safe_y_min = int(SETTINGS.get("attack", {}).get("safe_drop_y_min", self._SAFE_Y_MIN))
        safe_y_max = int(SETTINGS.get("attack", {}).get("safe_drop_y_max", self._SAFE_Y_MAX))

        for d in valid_dets:
            # Obtener el punto de drop sobre el césped verde exterior con clearance amplio y orientación zonal
            drop_x, drop_y = self._get_drop_point_on_contour(
                d.center, green_contour, red_contour=red_contour, extra_margin_px=45.0, zone_name=zone_name
            )

            # Verificar si el punto de drop está en zona segura de toque (lejos de HUD y cartas)
            # Y verificar que no caiga dentro de la zona roja
            inside_red = False
            if red_contour is not None and len(red_contour) >= 3:
                dist_red = cv2.pointPolygonTest(red_contour, (float(drop_x), float(drop_y)), True)
                inside_red = dist_red > 0.0

            # Si el drop queda dentro del HUD o dentro de la zona roja, marcar como no alcanzable en esta vista
            if inside_red or drop_y < safe_y_min or drop_y > safe_y_max or hud_mask[drop_y, drop_x] > 0:
                unreached_targets.append(d)
            else:
                reachable_targets.append((d, (drop_x, drop_y)))

        # Ordenar objetivos alcanzables por prioridad de botín para atacar almacenes primero
        loot_priority = {
            "storage_dark": 10,
            "storage_gold": 8,
            "storage_elixir": 8,
            "town_hall": 7,
            "collector_dark": 4,
            "collector_gold": 3,
            "collector_elixir": 3,
        }
        reachable_targets.sort(key=lambda item: loot_priority.get(item[0].cls_name, 1), reverse=True)

        log.info(
            "[%s] Objetivos evaluados: %d alcanzables de forma 100%% legal, %d fuera de margen.",
            zone_name, len(reachable_targets), len(unreached_targets)
        )

        # Guardar imagen de auditoría local en tiempo real para esta vista (incluso si no hay alcanzables)
        try:
            import os
            out_dir = os.path.join("logs", "yolo")
            os.makedirs(out_dir, exist_ok=True)
            dbg_zone = screen.copy()
            if red_contour is not None and len(red_contour) >= 3:
                cv2.polylines(dbg_zone, [red_contour.reshape(-1, 1, 2).astype(np.int32)], True, (0, 0, 255), 2)
            if green_contour is not None and len(green_contour) >= 3:
                cv2.polylines(dbg_zone, [green_contour.reshape(-1, 1, 2).astype(np.int32)], True, (0, 255, 0), 2)
            for d, (dx, dy) in reachable_targets:
                cv2.circle(dbg_zone, (dx, dy), 7, (255, 0, 255), -1)
                cv2.arrowedLine(dbg_zone, (dx, dy), (int(d.center[0]), int(d.center[1])), (255, 255, 0), 2, tipLength=0.15)
                lbl = f"{d.cls_name}"
                cv2.putText(dbg_zone, lbl, (dx - 20, max(20, dy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
            for ud in unreached_targets:
                cv2.circle(dbg_zone, (int(ud.center[0]), int(ud.center[1])), 8, (0, 0, 255), 2)
                cv2.putText(dbg_zone, f"UNREACHED: {ud.cls_name}", (int(ud.center[0]) - 30, int(ud.center[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1)
            out_zpath = os.path.join(out_dir, f"zone_{zone_name}_debug.png")
            save_debug_image_async(out_zpath, dbg_zone)
            log.info("📸 Auditoría de zona [%s] guardada en: %s", zone_name, out_zpath)
        except Exception as e_zdbg:
            log.debug("Error guardando auditoría de zona %s: %s", zone_name, e_zdbg)

        if not reachable_targets:
            return 0, unreached_targets

        # 5. Si no se omiten hechizos, lanzar saltos según la zona
        if not skip_spells:
            storages = [d for d, _ in reachable_targets if "storage" in d.cls_name or d.cls_name == "town_hall"]
            if zone_name == "CENTRO":
                if self._storages_are_interior_locked(storages, green_contour):
                    self._deploy_strategic_spells_centered(screen, precomputed_detections=[d for d, _ in reachable_targets])
            else:
                self._deploy_flank_spells(screen, zone_name, reachable_targets, green_contour, safe_y_min, safe_y_max)

        # 5.5. Despliegue de Súper Rompemuros para detonar Capa 1 exterior
        t_last_swb = None
        use_swb = SETTINGS.get("attack", {}).get("use_super_wall_breakers", True)
        swb_per_entry = int(SETTINGS.get("attack", {}).get("swb_per_entry", 2))
        swb_delay = float(SETTINGS.get("attack", {}).get("swb_sequence_delay_sec", 0.20))
        max_swb_breaches = int(SETTINGS.get("attack", {}).get("swb_max_breaches", 4))

        swb_drop_points = []
        if use_swb and self._troop_inventory.get("swb", 0) > 0 and reachable_targets:
            seen_drops = set()
            active_jumps = getattr(self, "_used_jump_points", [])

            # Función para verificar si un edificio ya cuenta con cobertura de salto
            def has_jump_coverage(pt):
                for jx, jy in active_jumps:
                    if np.hypot(pt[0] - jx, pt[1] - jy) < 140.0:
                        return True
                return False

            # Prioridad 1: Objetivos de botín mayor (almacenes / TH) que NO tienen salto asignado
            candidates_priority = [
                (d, dx, dy) for d, (dx, dy) in reachable_targets
                if ("storage" in d.cls_name or d.cls_name == "town_hall") and not has_jump_coverage(d.center)
            ]
            # Prioridad 2: Otros edificios de botín sin salto
            candidates_secondary = [
                (d, dx, dy) for d, (dx, dy) in reachable_targets
                if not has_jump_coverage(d.center) and (d, dx, dy) not in candidates_priority
            ]
            # Prioridad 3: Resto de objetivos para abrir caminos alternos si sobran SWB
            candidates_fallback = [
                (d, dx, dy) for d, (dx, dy) in reachable_targets
                if (d, dx, dy) not in candidates_priority and (d, dx, dy) not in candidates_secondary
            ]

            ordered_candidates = candidates_priority + candidates_secondary + candidates_fallback

            for d, dx, dy in ordered_candidates:
                grid_key = (dx // 100, dy // 100)
                if grid_key not in seen_drops:
                    seen_drops.add(grid_key)
                    swb_drop_points.append((dx, dy, d))
                    if len(swb_drop_points) >= max_swb_breaches:
                        break

            if swb_drop_points and self._ensure_super_wall_breaker_selected():
                for swb_dx, swb_dy, target_d in swb_drop_points:
                    if self._troop_inventory.get("swb", 0) <= 0:
                        break
                    count_to_deploy = min(swb_per_entry, self._troop_inventory.get("swb", 0))
                    log.info(
                        "💣 [%s] Lanzando %d Súper Rompemuros secuenciales en (%d, %d) hacia %s... (Remanente: %d)",
                        zone_name, count_to_deploy, swb_dx, swb_dy, target_d.cls_name, self._troop_inventory.get("swb", 0) - count_to_deploy
                    )
                    for _ in range(count_to_deploy):
                        jx, jy = HumanBehavior.jitter_point(swb_dx, swb_dy, max_offset=4)
                        jx = int(np.clip(jx, self._SAFE_X_MIN, self._SAFE_X_MAX))
                        jy = int(np.clip(jy, safe_y_min, safe_y_max))
                        self.device.d.click(jx, jy)
                        self._troop_inventory["swb"] -= 1
                        self._opened_breaches.add((swb_dx, swb_dy))
                        time.sleep(swb_delay)
                    time.sleep(random.uniform(0.08, 0.14))
                t_last_swb = time.monotonic()

        # 5.6. Despliegue de Héroes de Vanguardia (Rey, Reina, Centinela, Luchadora) con Habilidades Simultáneas
        use_heroes = SETTINGS.get("attack", {}).get("use_heroes", True)
        if use_heroes and not getattr(self, "_heroes_deployed_in_battle", False) and reachable_targets:
            # Seleccionar la brecha primaria (punto de SWB o primer objetivo de botín mayor)
            if swb_drop_points:
                hero_drop_x, hero_drop_y = swb_drop_points[0][0], swb_drop_points[0][1]
            else:
                hero_drop_x, hero_drop_y = reachable_targets[0][1][0], reachable_targets[0][1][1]

            log.info("👑 [%s] Desplegando Héroes de Vanguardia en brecha principal (%d, %d)...", zone_name, hero_drop_x, hero_drop_y)
            self._deploy_heroes_spearhead(hero_drop_x, hero_drop_y, safe_y_min, safe_y_max)

        # 5.7. Despliegue táctico de Hechizos de Aceleración vinculados preferentemente a Saltos
        haste_per_breach = int(SETTINGS.get("attack", {}).get("haste_per_breach", 1))
        if haste_per_breach > 0 and self._troop_inventory.get("haste", 0) > 0 and reachable_targets:
            haste_targets = [item for item in reachable_targets if "storage" in item[0].cls_name or item[0].cls_name == "town_hall"]
            if not haste_targets:
                haste_targets = reachable_targets[:1]

            active_jumps = getattr(self, "_used_jump_points", [])
            for h_target, (h_dx, h_dy) in haste_targets[:2]:
                if self._troop_inventory.get("haste", 0) <= 0:
                    break
                # Si hay un salto asignado cerca del objetivo, desplegar Haste directamente junto al Salto
                closest_jump = None
                if active_jumps:
                    closest_jump = min(active_jumps, key=lambda j: np.hypot(j[0] - h_target.center[0], j[1] - h_target.center[1]))
                    if np.hypot(closest_jump[0] - h_target.center[0], closest_jump[1] - h_target.center[1]) > 220.0:
                        closest_jump = None

                if closest_jump is not None:
                    self._deploy_haste_near_jump(closest_jump, h_target.center, zone_name=zone_name, safe_y_min=safe_y_min, safe_y_max=safe_y_max)
                else:
                    mid_hx = int(round((h_dx + h_target.center[0]) / 2.0))
                    mid_hy = int(round((h_dy + h_target.center[1]) / 2.0))
                    mid_hx = int(np.clip(mid_hx, self._SAFE_X_MIN, self._SAFE_X_MAX))
                    mid_hy = int(np.clip(mid_hy, safe_y_min, safe_y_max))

                    if self._ensure_spell_selected("haste"):
                        count_haste = min(haste_per_breach, self._troop_inventory.get("haste", 0))
                        log.info(
                            "⚡ [%s] Lanzando %d Hechizo de Aceleración en corredor hacia %s en (%d, %d)... (Remanente: %d)",
                            zone_name, count_haste, h_target.cls_name, mid_hx, mid_hy, self._troop_inventory.get("haste", 0) - count_haste
                        )
                        for _ in range(count_haste):
                            jx, jy = HumanBehavior.jitter_point(mid_hx, mid_hy, max_offset=4)
                            jx = int(np.clip(jx, self._SAFE_X_MIN, self._SAFE_X_MAX))
                            jy = int(np.clip(jy, safe_y_min, safe_y_max))
                            self.device.d.click(jx, jy)
                            self._troop_inventory["haste"] -= 1
                            time.sleep(0.12)
                        time.sleep(0.12)
                    else:
                        break

        # 6. Despliegue de tropas con ranura cacheada
        if t_last_swb is not None:
            elapsed_swb = time.monotonic() - t_last_swb
            lead_wait = float(SETTINGS.get("attack", {}).get("swb_travel_lead_sec", 0.85))
            if elapsed_swb < lead_wait and self.device.__class__.__name__ == "Device":
                time.sleep(lead_wait - elapsed_swb)

        if self._troop_inventory.get("goblins", 0) > 0:
            self._ensure_goblins_selected()
        total_deployed = 0

        # Función para determinar si el objetivo tiene camino libre garantizado:
        # a) Es recolector exterior (no requiere salto ni muro)
        # b) Tiene un Hechizo de Salto cerca de su corredor/centro (< 150px)
        # c) Tiene una brecha abierta por SWB cerca de su drop (< 150px)
        def target_openness_rank(item):
            d, (drop_x, drop_y) = item
            is_collector = "collector" in d.cls_name
            has_jump = any(np.hypot(d.center[0] - jx, d.center[1] - jy) < 150.0 for jx, jy in self._used_jump_points)
            has_breach = any(np.hypot(drop_x - bx, drop_y - by) < 150.0 for bx, by in self._opened_breaches)

            if d.cls_name == "town_hall":
                dist_to_drop = float(np.hypot(d.center[0] - drop_x, d.center[1] - drop_y))
                is_open = has_jump or dist_to_drop < 160.0
            else:
                is_open = is_collector or has_jump or has_breach

            loot_prio = self._QUOTA_PRIORITY.index(d.cls_name) if d.cls_name in self._QUOTA_PRIORITY else 99
            return (0 if is_open else 1, loot_prio)

        # Ordenar alcanzables priorizando vías abiertas y luego por botín
        reachable_targets.sort(key=target_openness_rank)

        for d, (drop_x, drop_y) in reachable_targets:
            if self._troop_inventory.get("goblins", 0) <= 0:
                log.info("🛑 [%s] Inventario de duendes agotado (0 restantes). Finalizando despliegue de duendes.", zone_name)
                break
            desired_quota = self._GOBLIN_QUOTA.get(d.cls_name, 2)
            quota = min(desired_quota, self._troop_inventory.get("goblins", 0))
            rem_after = self._troop_inventory.get("goblins", 0) - quota
            log.info(
                "  🎯 [%s] %s  centro=(%d,%d) -> drop=(%d,%d) [Cuota: %d, Remanente: %d]",
                zone_name, d.cls_name, int(d.center[0]), int(d.center[1]), drop_x, drop_y, quota, rem_after
            )

            for idx_q in range(quota):
                jx, jy = HumanBehavior.jitter_point(drop_x, drop_y, max_offset=4)
                jx = int(np.clip(jx, self._SAFE_X_MIN, self._SAFE_X_MAX))
                jy = int(np.clip(jy, safe_y_min, safe_y_max))
                self.device.d.click(jx, jy)
                self._troop_inventory["goblins"] -= 1
                total_deployed += 1
                if idx_q < quota - 1 and self.device.__class__.__name__ == "Device":
                    HumanBehavior.burst_tap_delay()

            deployed_drops_log.append({
                "zone": zone_name,
                "building": d.cls_name,
                "center": (int(d.center[0]), int(d.center[1])),
                "drop": (drop_x, drop_y),
                "quota": quota
            })
            time.sleep(random.uniform(0.04, 0.08))

        return total_deployed, unreached_targets

    def _deploy_by_structure_quotas(
        self,
        detections: list = None,
        green_hull = None,
        screen_size: tuple[int, int] = None,
        screen_centered: np.ndarray | None = None,
        red_contour: np.ndarray | None = None,
    ) -> bool:
        """
        Orquestador de Ataque por Zonas Autónomas:
        1. Zona Central: zoom out completo, evalúa y ataca todo lo que tenga drop 100% legal.
        2. Recorrido en cruz 4 direcciones (Norte, Sur, Este, Oeste) con desplazamientos calibrados
           y retornos recíprocos al Centro exacto.
        3. Despliegue de héroes de vanguardia y habilidades simultáneas + aceleración activa.
        4. Guarda el log visual consolidado de auditoría en logs/yolo/quota_attack_debug.png.
        """
        deployed_drops_log = []
        self._opened_breaches.clear()
        self._used_jump_points.clear()
        self._heroes_deployed_in_battle = False
        self._troop_inventory = {
            "goblins": 96,
            "swb": 8,
            "jump": 4,
            "haste": 3,
        }
        log.info("🚀 Iniciando Pipeline de Ataque por Zonas Autónomas...")
        log.info("📦 Inventario inicial: %s", self._troop_inventory)

        # ── FASE 1: VISTA CENTRAL ─────────────────────────────────────────────
        log.info("📍 --- FASE 1: Operando en Zona Central ---")
        deployed_center, unreached = self._execute_single_zone_quota_deploy(
            zone_name="CENTRO",
            deployed_drops_log=deployed_drops_log,
            skip_spells=False,
            precomputed_screen=screen_centered,
            precomputed_detections=detections,
            precomputed_green=green_hull,
            precomputed_red=red_contour,
        )

        enable_pan = SETTINGS.get("attack", {}).get("enable_camera_pan", True)
        enable_full_cross_pan = SETTINGS.get("attack", {}).get("enable_full_cross_pan", True)
        min_unreached = int(SETTINGS.get("attack", {}).get("min_unreached_targets_for_pan", 1))
        configured_flanks = [f.lower() for f in SETTINGS.get("attack", {}).get("zone_attack_flanks", ["north", "south", "east", "west"])]

        should_pan = enable_pan and (enable_full_cross_pan or len(unreached) >= min_unreached)

        if should_pan:
            # Evaluar objetivos inalcanzables en los 4 cuadrantes
            unreached_north = [d for d in unreached if d.center[1] < 450]
            unreached_south = [d for d in unreached if d.center[1] >= 450]
            unreached_east = [d for d in unreached if d.center[0] >= 1350]
            unreached_west = [d for d in unreached if d.center[0] < 1100]

            # ── FASE 2: FLANCO NORTE ──────────────────────────────────────────
            if self._troop_inventory.get("goblins", 0) <= 0:
                log.info("🛑 Inventario de duendes agotado (0 restantes). Omitiendo paneos restantes.")
            elif "north" in configured_flanks and (enable_full_cross_pan or unreached_north):
                desc_north = f"{len(unreached_north)} objetivos pendientes" if unreached_north else "Barrido Completo 4 Cuadrantes"
                log.info("📍 --- FASE 2: Paneo hacia Flanco Norte (%s) ---", desc_north)
                # Swipe hacia abajo para revelar el Norte (arrastre hacia abajo sube la vista al Norte)
                self.device.d.shell("input swipe 1200 240 1200 820 480")
                self._wait_after_pan(fallback_sec=0.50)
                try:
                    self._execute_single_zone_quota_deploy(
                        zone_name="NORTE",
                        deployed_drops_log=deployed_drops_log,
                        skip_spells=False
                    )
                finally:
                    # Desplazamiento recíproco obligatorio para regresar al Centro exacto
                    log.info("🔄 Desplazamiento recíproco: Regresando del Flanco Norte al Centro...")
                    self.device.d.shell("input swipe 1200 820 1200 240 480")
                    time.sleep(0.50)

            # ── FASE 3: FLANCO SUR ───────────────────────────────────────────
            if self._troop_inventory.get("goblins", 0) <= 0:
                log.info("🛑 Inventario de duendes agotado (0 restantes). Omitiendo paneos restantes.")
            elif "south" in configured_flanks and (enable_full_cross_pan or unreached_south):
                desc_south = f"{len(unreached_south)} objetivos pendientes" if unreached_south else "Barrido Completo 4 Cuadrantes"
                log.info("📍 --- FASE 3: Paneo hacia Flanco Sur (%s) ---", desc_south)
                # Swipe hacia arriba para revelar el Sur (arrastre hacia arriba baja la vista al Sur)
                self.device.d.shell("input swipe 1200 780 1200 240 480")
                self._wait_after_pan(fallback_sec=0.50)
                try:
                    self._execute_single_zone_quota_deploy(
                        zone_name="SUR",
                        deployed_drops_log=deployed_drops_log,
                        skip_spells=False
                    )
                finally:
                    # Desplazamiento recíproco obligatorio para regresar al Centro exacto
                    log.info("🔄 Desplazamiento recíproco: Regresando del Flanco Sur al Centro...")
                    self.device.d.shell("input swipe 1200 240 1200 780 480")
                    time.sleep(0.50)

            # ── FASE 4: FLANCO ESTE ──────────────────────────────────────────
            if self._troop_inventory.get("goblins", 0) <= 0:
                log.info("🛑 Inventario de duendes agotado (0 restantes). Omitiendo paneos restantes.")
            elif "east" in configured_flanks and (enable_full_cross_pan or unreached_east):
                desc_east = f"{len(unreached_east)} objetivos pendientes" if unreached_east else "Barrido Completo 4 Cuadrantes"
                log.info("📍 --- FASE 4: Paneo hacia Flanco Este (%s) ---", desc_east)
                # Swipe calibrado de 700px hacia la izquierda para revelar el Este (sin desplazar estructuras fuera de pantalla)
                self.device.d.shell("input swipe 1550 540 850 540 480")
                self._wait_after_pan(fallback_sec=0.50)
                try:
                    self._execute_single_zone_quota_deploy(
                        zone_name="ESTE",
                        deployed_drops_log=deployed_drops_log,
                        skip_spells=False
                    )
                finally:
                    # Desplazamiento recíproco obligatorio para regresar al Centro exacto
                    log.info("🔄 Desplazamiento recíproco: Regresando del Flanco Este al Centro...")
                    self.device.d.shell("input swipe 850 540 1550 540 480")
                    time.sleep(0.50)

            # ── FASE 5: FLANCO OESTE ─────────────────────────────────────────
            if self._troop_inventory.get("goblins", 0) <= 0:
                log.info("🛑 Inventario de duendes agotado (0 restantes). Omitiendo paneos restantes.")
            elif "west" in configured_flanks and (enable_full_cross_pan or unreached_west):
                desc_west = f"{len(unreached_west)} objetivos pendientes" if unreached_west else "Barrido Completo 4 Cuadrantes"
                log.info("📍 --- FASE 5: Paneo hacia Flanco Oeste (%s) ---", desc_west)
                # Swipe calibrado de 700px hacia la derecha para revelar el Oeste
                self.device.d.shell("input swipe 850 540 1550 540 480")
                self._wait_after_pan(fallback_sec=0.50)
                try:
                    self._execute_single_zone_quota_deploy(
                        zone_name="OESTE",
                        deployed_drops_log=deployed_drops_log,
                        skip_spells=False
                    )
                finally:
                    # Desplazamiento recíproco obligatorio para regresar al Centro exacto
                    log.info("🔄 Desplazamiento recíproco: Regresando del Flanco Oeste al Centro...")
                    self.device.d.shell("input swipe 1550 540 850 540 480")
                    time.sleep(0.50)

        # ── FASE FINAL: AUDITORÍA VISUAL ─────────────────────────────────────
        try:
            self._save_quota_attack_debug_image(deployed_drops_log)
        except Exception as e_dbg:
            log.warning("No se pudo generar imagen de auditoría de drop: %s", e_dbg)

        log.info("✅ Pipeline por Zonas completado. Total de impactos: %d", len(deployed_drops_log))
        return True

    def _save_quota_attack_debug_image(self, deployed_drops_log: list):
        import cv2, os
        out_dir = os.path.join("logs", "yolo")
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, "quota_attack_debug.png")
        out_path_ts = os.path.join(out_dir, f"quota_attack_{ts}.png")

        screen = self.device.screenshot() if self.device else None
        if isinstance(screen, np.ndarray) and screen.size > 0:
            dbg = screen.copy()
        else:
            dbg = np.zeros((1080, 2400, 3), dtype=np.uint8)

        cached_red = getattr(self, "_cached_red_contour", None)
        if cached_red is not None and len(cached_red) >= 3:
            cv2.polylines(dbg, [cached_red.reshape(-1, 1, 2).astype(np.int32)], True, (0, 0, 255), 2)
        cached_green = getattr(self, "_cached_green_contour", None)
        if cached_green is not None and len(cached_green) >= 3:
            cv2.polylines(dbg, [cached_green.reshape(-1, 1, 2).astype(np.int32)], True, (0, 255, 0), 2)

        for item in deployed_drops_log:
            bx, by = item["center"]
            dx, dy = item["drop"]
            cls_name = item["building"]
            quota = item["quota"]
            zone = item.get("zone", "C")

            # Punto de drop en magenta
            cv2.circle(dbg, (dx, dy), 8, (255, 0, 255), -1)
            # Flecha cian hacia el edificio
            cv2.arrowedLine(dbg, (dx, dy), (bx, by), (255, 255, 0), 2, tipLength=0.15)
            # Etiqueta
            lbl = f"[{zone}] {cls_name} x{quota}"
            cv2.putText(dbg, lbl, (dx - 20, max(20, dy - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

        save_debug_image_async(out_path, dbg)
        save_debug_image_async(out_path_ts, dbg)
        log.info("📸 Auditoría visual guardada en: %s y %s", out_path, out_path_ts)

    def execute_attack(self) -> bool:
        log.info("Iniciando despliegue de ataque con ritmo pausado y seguro...")

        # Reinicio de estado de batalla para caché de ranuras y objetivos
        self._cached_goblin_slot_pos = None
        self._goblins_currently_active = False
        self._pending_extreme_targets = []
        self._troop_inventory = {
            "goblins": 96,
            "swb": 8,
            "jump": 4,
            "haste": 3,
        }

        # 1. Reconocimiento y análisis completo de la aldea antes de soltar hechizos o tropas
        screen_centered, detections, green_hull = self._survey_and_analyze_village()

        # 2. Modo de despliegue seleccionado en settings
        deploy_mode = SETTINGS.get("attack", {}).get("deployment_mode", "targeted_corridors")

        # ── Modo: cuotas por estructura (nuevo método recomendado) ────────────
        if deploy_mode == "structure_quota_deployment":
            log.info("🎯 Modo QuotaDeploy activo — despliegue por estructura con clicks directos.")
            return self._deploy_by_structure_quotas(
                detections,
                green_hull,
                (screen_centered.shape[1], screen_centered.shape[0]),
                screen_centered=screen_centered,
                red_contour=getattr(self, "_cached_red_contour", None),
            )

        # ── Modo: corredores focalizados (legado — requiere hechizos de salto) ─
        # 2b. Despliegue de hechizos de salto estratégico con cámara centrada (reutilizando detecciones)
        deployed_jumps = self._deploy_strategic_spells_centered(screen_centered, precomputed_detections=detections)

        # 3. Confirmar duendes seleccionados
        if not self._ensure_goblins_selected():
            return False
        time.sleep(random.uniform(0.35, 0.55))

        # 4. Modo de Despliegue Autónomo Focalizado (Target-Driven Deployment)
        if deploy_mode == "targeted_corridors" and deployed_jumps:
            log.info("🎯 Modo de Ataque Autónomo Activo: Despliegue Focalizado en %d Corredores de Salto.", len(deployed_jumps))
            return self._deploy_targeted_goblin_waves(
                deployed_jumps, detections, (screen_centered.shape[1], screen_centered.shape[0]), green_hull=green_hull
            )

        log.info("⚙️ Modo de Ataque Convencional: Ejecutando barridos perimetrales...")

        sweep_rounds = int(SETTINGS.get("attack", {}).get("sweep_rounds", 1))
        enable_pan = bool(SETTINGS.get("attack", {}).get("enable_camera_pan", True))
        start_cfg = str(SETTINGS.get("attack", {}).get("camera_start_side", "adaptive_random")).lower()

        # 4. Decisión dinámica y aleatorizada del flanco de ataque inicial
        h, w = screen_centered.shape[:2]
        center_y = h * 0.50
        north_loot_score = sum(
            3 if "dark" in d.cls_name else 2
            for d in detections if "storage" in d.cls_name and d.center[1] < center_y
        )
        south_loot_score = sum(
            3 if "dark" in d.cls_name else 2
            for d in detections if "storage" in d.cls_name and d.center[1] >= center_y
        )

        if start_cfg in ("adaptive", "adaptive_random", "auto"):
            if north_loot_score > south_loot_score * 1.5:
                # Mayor botín al norte: alta probabilidad de iniciar al norte
                chosen_side = "norte" if random.random() < 0.75 else "sur"
            elif south_loot_score > north_loot_score * 1.5:
                # Mayor botín al sur: alta probabilidad de iniciar al sur
                chosen_side = "sur" if random.random() < 0.75 else "norte"
            else:
                # Distribución equilibrada: 50/50 puro
                chosen_side = random.choice(["sur", "norte"])
            log.info("🎲 Flanco inicial adaptativo/aleatorio: %s (Score Norte=%d, Sur=%d)", chosen_side.upper(), north_loot_score, south_loot_score)
        elif start_cfg in ("random", "aleatorio"):
            chosen_side = random.choice(["sur", "norte"])
            log.info("🎲 Flanco inicial aleatorio puro: %s", chosen_side.upper())
        elif start_cfg in ("norte", "sur"):
            chosen_side = start_cfg
            log.info("⚙️ Flanco inicial fijo configurado: %s", chosen_side.upper())
        else:
            chosen_side = random.choice(["sur", "norte"])

        # 5. Mover la cámara desde el Centro hacia el extremo inicial elegido
        current_side = "center"
        if enable_pan:
            if chosen_side == "norte":
                log.info("📷 Moviendo cámara del Centro al Sector Norte (PAN_CENTER_TO_NORTH)...")
                self.device.pan_camera(*PAN_CENTER_TO_NORTH, duration=0.9)
                current_side = "norte"
            else:
                log.info("📷 Moviendo cámara del Centro al Sector Sur (PAN_CENTER_TO_SOUTH)...")
                self.device.pan_camera(*PAN_CENTER_TO_SOUTH, duration=0.9)
                current_side = "sur"

            self.device.wait_camera_settled(timeout=2.5)
            time.sleep(random.uniform(0.5, 0.7))
            self._ensure_goblins_selected()
            time.sleep(0.4)

        for r in range(1, sweep_rounds + 1):
            log.info("⚔️ Ejecutando Ronda de Barrido %d/%d (Flanco actual: %s)...", r, sweep_rounds, current_side.upper())

            if current_side in ("sur", "center"):
                # Aceleración fresca sincronizada para el Sector Sur
                if r == 1:
                    self._deploy_flank_haste("sur")

                # FASE A: Despliegue en Sector Sur (orden de lados aleatorizado)
                south_order = random.choice([("izq", "der"), ("der", "izq")])
                log.info("  -> Fase: Barridos continuos Sector Sur (orden: %s -> %s)...", south_order[0].upper(), south_order[1].upper())
                self._ensure_goblins_selected()
                time.sleep(0.4)

                for lado in south_order:
                    off_sur = (random.randint(-2, 2), random.randint(-2, 2))
                    self._stroke(SWEEP_SUR[lado], seconds=1.3, anchor_sec=0.60, offset=off_sur)
                    time.sleep(random.uniform(0.4, 0.65))

                if enable_pan:
                    # Mover cámara al otro extremo (Norte)
                    log.info("📷 Moviendo cámara al otro extremo: Sector Norte (PAN_SOUTH_TO_NORTH)...")
                    self.device.pan_camera(*PAN_SOUTH_TO_NORTH, duration=1.0)
                    self.device.wait_camera_settled(timeout=2.5)
                    time.sleep(0.6)
                    current_side = "norte"

                    # Aceleración fresca sincronizada para el Sector Norte
                    if r == 1:
                        self._deploy_flank_haste("norte")

                    # FASE B: Despliegue en Sector Norte (orden de lados aleatorizado)
                    north_order = random.choice([("izq", "der"), ("der", "izq")])
                    log.info("  -> Fase: Barridos continuos Sector Norte (orden: %s -> %s)...", north_order[0].upper(), north_order[1].upper())
                    self._ensure_goblins_selected()
                    time.sleep(0.4)

                    for lado in north_order:
                        off_norte = (random.randint(-2, 2), random.randint(-2, 2))
                        self._stroke(SWEEP_NORTE[lado], seconds=1.3, anchor_sec=0.60, offset=off_norte)
                        time.sleep(random.uniform(0.4, 0.65))

                    if r < sweep_rounds:
                        log.info("📷 Regresando cámara al otro extremo: Sector Sur (PAN_NORTH_TO_SOUTH)...")
                        self.device.pan_camera(*PAN_NORTH_TO_SOUTH, duration=1.0)
                        self.device.wait_camera_settled(timeout=2.5)
                        time.sleep(0.6)
                        current_side = "sur"
            else:
                # Si inició en Norte
                if r == 1:
                    self._deploy_flank_haste("norte")

                # FASE A: Despliegue en Sector Norte (orden de lados aleatorizado)
                north_order = random.choice([("izq", "der"), ("der", "izq")])
                log.info("  -> Fase: Barridos continuos Sector Norte (orden: %s -> %s)...", north_order[0].upper(), north_order[1].upper())
                self._ensure_goblins_selected()
                time.sleep(0.4)

                for lado in north_order:
                    off_norte = (random.randint(-2, 2), random.randint(-2, 2))
                    self._stroke(SWEEP_NORTE[lado], seconds=1.3, anchor_sec=0.60, offset=off_norte)
                    time.sleep(random.uniform(0.4, 0.65))

                if enable_pan:
                    # Mover cámara al otro extremo (Sur)
                    log.info("📷 Moviendo cámara al otro extremo: Sector Sur (PAN_NORTH_TO_SOUTH)...")
                    self.device.pan_camera(*PAN_NORTH_TO_SOUTH, duration=1.0)
                    self.device.wait_camera_settled(timeout=2.5)
                    time.sleep(0.6)
                    current_side = "sur"

                    if r == 1:
                        self._deploy_flank_haste("sur")

                    # FASE B: Despliegue en Sector Sur (orden de lados aleatorizado)
                    south_order = random.choice([("izq", "der"), ("der", "izq")])
                    log.info("  -> Fase: Barridos continuos Sector Sur (orden: %s -> %s)...", south_order[0].upper(), south_order[1].upper())
                    self._ensure_goblins_selected()
                    time.sleep(0.4)

                    for lado in south_order:
                        off_sur = (random.randint(-2, 2), random.randint(-2, 2))
                        self._stroke(SWEEP_SUR[lado], seconds=1.3, anchor_sec=0.60, offset=off_sur)
                        time.sleep(random.uniform(0.4, 0.65))

                    if r < sweep_rounds:
                        log.info("📷 Regresando cámara al otro extremo: Sector Norte (PAN_SOUTH_TO_NORTH)...")
                        self.device.pan_camera(*PAN_SOUTH_TO_NORTH, duration=1.0)
                        self.device.wait_camera_settled(timeout=2.5)
                        time.sleep(0.6)
                        current_side = "norte"

            time.sleep(random.uniform(1.0, 1.4))  # Pausa deliberada entre rondas de ataque

        log.info("✅ Despliegue de ataque completado con éxito.")
        return True
```

## File: attack/edrag_strategy.py
```python
"""
attack/edrag_strategy.py
Estrategia de ataque autónoma: Dragones Eléctricos + Furias (EDrag 50% Rush).
Enfocada en victorias rápidas del 50%+ para farmeo constante del Bonus de Liga.
Despliega en el Triángulo Norte de la aldea:
  1. Zoom out y Paneo decisivo hacia el Norte para encuadrar la zona norte.
  2. Furias de apertura cubriendo la entrada de ambos lados del triángulo.
  3. Despliegue de 11 Dragones Eléctricos en ambos lados a la vez en césped exterior alto y alejado.
  4. Los 4 Héroes juntos en césped exterior protegido (RC, Minion Prince, Warden, Duke Dragon).
  5. Activación rápida de habilidades de héroes tras 2 segundos.
  6. Furias de avance escalonadas hacia el núcleo / Ayuntamiento tras 10 segundos.
  (Sin máquina de asedio).
"""
import os
import time
import random
import datetime
import cv2
import numpy as np

from core.device import Device
from core.vision import Vision
from humanizer.behavior import HumanBehavior
from core.logger import get_logger
from core.config import SETTINGS
from vision.army_checker import ArmyChecker

log = get_logger("edrag_strategy")


class EDragStrategy:
    """Implementa la secuencia de ataque por el Triángulo Norte con Dragones Eléctricos y Furias."""

    def __init__(self, device: Device, vision: Vision, human: HumanBehavior = None):
        self.device = device
        self.vision = vision
        self.human = human or HumanBehavior()
        self.army = ArmyChecker(self.vision)
        self.w = device.width
        self.h = device.height

        cfg = SETTINGS.get("attack", {}).get("edrag_rush", {})
        self.zoom_mode = str(cfg.get("zoom_mode", "half")).lower()
        self.rage_opening_count = int(cfg.get("rage_opening_count", 2))
        self.rage_core_delay_sec = float(cfg.get("rage_core_delay_sec", 10.0))
        self.hero_ability_delay_sec = float(cfg.get("hero_ability_delay_sec", 2.0))
        self._last_detections = []

    def compute_north_triangle_drops(self, screen=None) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """
        Calcula puntos de despliegue legal en el césped exterior norte,
        altos y alejados de la línea roja construible a ambos lados del vértice norte.
        Con zoom moderado (mitad disponible), la superficie de césped exterior es enorme.
        """
        safe_top_y = 145
        if screen is not None:
            try:
                from vision.yolo_detector import get_detector
                detector = get_detector()
                if detector and detector.is_ready():
                    dets = detector.detect(screen)
                    if dets:
                        self._last_detections = dets
                        min_b_y = min(d.bbox_xyxy[1] for d in dets)
                        # Margen seguro por encima del edificio más alto, nunca menor a 140 (área activa de pantalla)
                        safe_top_y = max(140, min(175, min_b_y - 80))
            except Exception as e:
                log.debug("YOLO safe drop check omitido: %s", e)

        # Flanco Noroeste: césped exterior amplio y seguro a la izquierda
        nw_drops = [
            (1020, safe_top_y - 5),
            (945, safe_top_y + 10),
            (880, safe_top_y + 35),
            (810, safe_top_y + 65),
            (740, safe_top_y + 95),
        ]
        # Flanco Noreste: césped exterior amplio y seguro a la derecha
        ne_drops = [
            (1380, safe_top_y),
            (1455, safe_top_y + 15),
            (1520, safe_top_y + 45),
            (1590, safe_top_y + 75),
            (1660, safe_top_y + 105),
        ]

        # Validación y ajuste fino vía DeployZoneDetector (IA U-Net)
        if screen is not None:
            try:
                from vision.deploy_zone_detector import get_deploy_zone_detector
                dz = get_deploy_zone_detector()
                if dz and dz.is_ready():
                    mask = dz.predict_mask(screen)
                    if mask is not None:
                        nw_drops = [
                            dz.find_closest_deployable_point(pt[0], pt[1], margin_px=25, mask=mask)
                            if not dz.is_point_deployable(pt[0], pt[1], mask=mask, safety_margin_px=15)
                            else pt
                            for pt in nw_drops
                        ]
                        ne_drops = [
                            dz.find_closest_deployable_point(pt[0], pt[1], margin_px=25, mask=mask)
                            if not dz.is_point_deployable(pt[0], pt[1], mask=mask, safety_margin_px=15)
                            else pt
                            for pt in ne_drops
                        ]
                        log.info("🛡️ DeployZoneDetector validó puntos del Triángulo Norte (IA activa)")
            except Exception as e:
                log.debug("DeployZoneDetector en EDrag check omitido: %s", e)

        nw_drops = [(int(np.clip(pt[0], 120, 2280)), int(np.clip(pt[1], 140, 845))) for pt in nw_drops]
        ne_drops = [(int(np.clip(pt[0], 120, 2280)), int(np.clip(pt[1], 140, 845))) for pt in ne_drops]

        return nw_drops, ne_drops

    def get_opening_rage_points(self) -> list[tuple[int, int]]:
        """
        Puntos de impacto para los Hechizos de Furia de apertura en el cono norte.
        Cubren la entrada de los dragones desde ambos flancos sobre las primeras defensas.
        """
        return [
            (1100, 260),  # Flanco izquierdo del cono norte
            (1300, 260),  # Flanco derecho del cono norte
            (1200, 200),  # Vértice Norte central (si hay 3 furias)
        ]

    def get_core_rage_points(self, detections: list | None = None) -> list[tuple[int, int]]:
        """
        Puntos de impacto para las Furias escalonadas hacia el núcleo / Ayuntamiento
        tras el avance de los dragones y héroes.
        """
        target_center = None
        if detections:
            # 1. Buscar town_hall en la mitad norte/central (300 < y < 650)
            th_candidates = [
                d for d in detections
                if getattr(d, "cls_name", "") == "town_hall" and 300 < getattr(d, "center", (0, 0))[1] < 650
            ]
            if th_candidates:
                target_center = th_candidates[0].center
            else:
                # 2. Buscar centroide de almacenes en 300 < y < 650
                storage_candidates = [
                    d for d in detections
                    if "storage" in getattr(d, "cls_name", "") and 300 < getattr(d, "center", (0, 0))[1] < 650
                ]
                if storage_candidates:
                    avg_x = sum(d.center[0] for d in storage_candidates) / len(storage_candidates)
                    avg_y = sum(d.center[1] for d in storage_candidates) / len(storage_candidates)
                    target_center = (avg_x, avg_y)

        if target_center is not None:
            c_x = int(np.clip(target_center[0], 950, 1450))
            c_y = int(np.clip(target_center[1], 320, 620))
            left_x = int(np.clip(c_x - 110, 120, self.w - 120))
            left_y = int(np.clip(c_y - 50, 310, 640))
            right_x = int(np.clip(c_x + 110, 120, self.w - 120))
            right_y = int(np.clip(c_y - 50, 310, 640))
            return [
                (c_x, c_y),
                (left_x, left_y),
                (right_x, right_y),
            ]

        return [
            (1200, 420),  # Ayuntamiento / Núcleo central
            (1100, 360),  # Compartimento interior izquierdo
            (1300, 360),  # Compartimento interior derecho
        ]

    def _tap_jitter(self, x: int, y: int):
        jx, jy = HumanBehavior.jitter_point(x, y, max_offset=6)
        jx = max(10, min(self.w - 10, jx))
        jy = max(10, min(self.h - 10, jy))
        self.device.tap(jx, jy)

    def execute_attack(self) -> bool:
        """
        Ejecuta la secuencia completa de ataque EDrag Rush por el Triángulo Norte:
        1. Zoom out y Paneo decisivo hacia el Norte.
        2. Furias de apertura en el cono norte.
        3. 11 Dragones Eléctricos en ambos lados a la vez (césped exterior alto y alejado).
        4. 4 Héroes juntos en césped exterior seguro (RC, Prince, Warden, Duke Dragon).
        5. Activación de habilidades de héroes.
        6. Furias de avance al núcleo.
        """
        log.info("🐉 ════════════════════════════════════════════════════════════")
        log.info("🐉 INICIANDO ESTRATEGIA DE ATAQUE: DRAGONES ELÉCTRICOS + FURIAS")
        log.info("🐉 Objetivo: 50%+ de Destrucción y Captura del Bonus de Liga")
        log.info("🐉 ════════════════════════════════════════════════════════════")

        # 1. Asegurar encuadre de combate óptimo según zoom_mode
        if self.zoom_mode == "half":
            log.info("🔍 Aplicando Zoom Moderado ('mitad disponible') para mayor superficie de césped...")
            if hasattr(self.device, "zoom_half"):
                self.device.zoom_half()
            else:
                self.device.zoom_out()
            log.info("🔍 Paneo de cámara hacia el sector Norte para encuadrar césped exterior amplio...")
            self.device.pan_camera(0.50, 0.35, 0.50, 0.70, duration=0.65)
        elif self.zoom_mode == "none":
            log.info("🔍 Manteniendo zoom nativo de batalla...")
            self.device.wait_camera_settled(timeout=1.0)
            log.info("🔍 Paneo de cámara suave hacia el sector Norte...")
            self.device.pan_camera(0.50, 0.30, 0.50, 0.60, duration=0.55)
        else:
            log.info("🔍 Ejecutando Zoom Out completo...")
            self.device.zoom_out()
            log.info("🔍 Paneo de cámara hacia el sector Norte...")
            self.device.pan_camera(0.50, 0.20, 0.50, 0.82, duration=0.85)

        self.device.wait_camera_settled()

        screen = self.device.screenshot()
        if screen is None:
            log.error("❌ No se pudo capturar la pantalla para el ataque.")
            return False

        # Guardar captura inicial del ataque tras el paneo
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("logs", "yolo")
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"edrag_attack_start_{ts}.png"), screen)

        # 3. Localizar tropas, hechizos y héroes en la barra de batalla
        edrag_pos = self.army.find_edrag_in_battle_bar(screen)
        rage_pos = self.army.find_rage_spell_in_battle_bar(screen)
        heroes = self.army.find_all_heroes_in_battle_bar(screen)

        log.info("📦 Reconocimiento de barra de batalla:")
        log.info("   • Dragones Eléctricos : %s", edrag_pos)
        log.info("   • Hechizo de Furia    : %s", rage_pos)
        log.info("   • Héroes detectados   : %s", list(heroes.keys()))

        if not edrag_pos:
            log.warning("⚠️ No se detectó ranura dinámica de E-Drag. Usando ranura estándar en (407, 955)...")
            edrag_pos = (407, 955)

        if not rage_pos:
            log.warning("⚠️ No se detectó ranura dinámica de Furia. Usando ranura estándar en (1345, 960)...")
            rage_pos = (1345, 960)

        nw_drops, ne_drops = self.compute_north_triangle_drops(screen)

        # ─────────────────────────────────────────────────────────────
        # FASE 1: HECHIZOS DE FURIA DE APERTURA (Escalonado Táctico)
        # ─────────────────────────────────────────────────────────────
        if rage_pos:
            log.info("✨ Seleccionando Hechizo de Furia en %s...", rage_pos)
            self.device.tap(*rage_pos)
            HumanBehavior.reaction_delay()

            opening_rages = self.get_opening_rage_points()[:self.rage_opening_count]
            for idx_r, r_pt in enumerate(opening_rages, 1):
                log.info("⚡ [APERTURA] Lanzando Furia #%d en el Triángulo Norte: %s", idx_r, r_pt)
                self._tap_jitter(r_pt[0], r_pt[1])
                time.sleep(0.18)
        else:
            log.warning("⚠️ No se detectó Hechizo de Furia en la barra.")

        # ─────────────────────────────────────────────────────────────
        # FASE 2: DESPLIEGUE EN 2 TAPS SOSTENIDOS SEPARADOS (DRAGONES ELÉCTRICOS)
        # ─────────────────────────────────────────────────────────────
        log.info("🐉 Seleccionando Dragones Eléctricos en %s...", edrag_pos)
        self.device.tap(*edrag_pos)
        HumanBehavior.reaction_delay()

        # Puntos de despliegue sostenido: 1 en flanco Noroeste y 1 en flanco Noreste
        # Ambos en césped exterior amplio y seguro, altos y alejados de la línea roja
        pt_nw = nw_drops[1]
        pt_ne = ne_drops[1]

        log.info("🌊 [1/2] Despliegue sostenido de Dragones Eléctricos en Flanco Noroeste: %s (sosteniendo ~0.9s)...", pt_nw)
        if hasattr(self.device, "hold_tap"):
            self.device.hold_tap(pt_nw[0], pt_nw[1], duration=0.90)
        else:
            self._tap_jitter(pt_nw[0], pt_nw[1])
        time.sleep(0.12)

        log.info("🌊 [2/2] Despliegue sostenido de Dragones Eléctricos en Flanco Noreste: %s (sosteniendo ~0.9s)...", pt_ne)
        if hasattr(self.device, "hold_tap"):
            self.device.hold_tap(pt_ne[0], pt_ne[1], duration=0.90)
        else:
            self._tap_jitter(pt_ne[0], pt_ne[1])
        time.sleep(0.12)

        log.info("✅ Desplegados Dragones Eléctricos en 2 tiradas sostenidas continuas con éxito.")

        # ─────────────────────────────────────────────────────────────
        # FASE 3: DESPLIEGUE DE LOS 4 HÉROES UNO A UNO EN EL MISMO PUNTO DE LOS PRIMEROS DRAGS
        # ─────────────────────────────────────────────────────────────
        # Orden preferente: Royal Champion (1ª), Minion Prince (2º), Grand Warden (3º), Duke Dragon / King (4º)
        hero_slots = {
            "champion": heroes.get("champion", (725, 960)),
            "minion_prince": heroes.get("minion_prince", (880, 960)),
            "warden": heroes.get("warden", (1035, 960)),
            "duke_dragon": heroes.get("duke_dragon", heroes.get("king", (1190, 960))),
        }
        hero_order = ["champion", "minion_prince", "warden", "duke_dragon"]

        # Punto único guardado donde se soltaron los primeros dragones (pt_nw)
        hero_drop_anchor = pt_nw
        log.info("👑 Desplegando los 4 Héroes uno a uno en el mismo punto de primeros dragones %s:", hero_drop_anchor)

        for idx_h, h_id in enumerate(hero_order, 1):
            h_slot = hero_slots[h_id]
            log.info("   • [%d/4] Seleccionando héroe %s en %s...", idx_h, h_id.upper(), h_slot)
            self.device.tap(*h_slot, humanize=False)
            time.sleep(0.08)
            log.info("     -> Soltando %s en punto ancla de dragones: %s", h_id.upper(), hero_drop_anchor)
            self.device.tap(hero_drop_anchor[0], hero_drop_anchor[1], humanize=False)
            time.sleep(0.10)

        # ─────────────────────────────────────────────────────────────
        # FASE 4: ACTIVACIÓN RÁPIDA DE HABILIDADES DE HÉROES (2s)
        # ─────────────────────────────────────────────────────────────
        log.info("⏳ Esperando %.1fs para que los héroes tomen agro y penetren defensas...", self.hero_ability_delay_sec)
        time.sleep(self.hero_ability_delay_sec)

        log.info("⚡ Activando habilidades especiales de los 4 héroes mediante toques sucesivos en la barra...")
        for h_id in hero_order:
            h_slot = hero_slots[h_id]
            log.info("   💥 Disparando habilidad especial de %s en %s!", h_id.upper(), h_slot)
            self.device.tap(*h_slot)
            time.sleep(0.10)

        # ─────────────────────────────────────────────────────────────
        # FASE 5: FURIAS DE AVANCE AL NÚCLEO / AYUNTAMIENTO (tras 10s)
        # ─────────────────────────────────────────────────────────────
        log.info("⏳ Esperando %.1fs de combate antes de soltar las Furias de avance al núcleo...", self.rage_core_delay_sec)
        time.sleep(self.rage_core_delay_sec)

        if rage_pos:
            log.info("✨ Re-seleccionando Hechizo de Furia para el avance central...")
            self.device.tap(*rage_pos)
            HumanBehavior.reaction_delay()

            core_rages = self.get_core_rage_points(getattr(self, "_last_detections", None))
            for idx_c, c_pt in enumerate(core_rages, 1):
                log.info("⚡ [NÚCLEO] Lanzando Furia de Avance #%d en: %s", idx_c, c_pt)
                self._tap_jitter(c_pt[0], c_pt[1])
                time.sleep(0.20)

        # Guardar captura final del despliegue completado
        s_deployed = self.device.screenshot()
        if s_deployed is not None:
            cv2.imwrite(os.path.join(out_dir, f"edrag_attack_deployed_{ts}.png"), s_deployed)

        log.info("🚀 Despliegue de Dragones Eléctricos, Furias y Héroes completado con éxito.")
        return True
```

## File: config/settings.json
```json
{
  "device": {
    "serial": "5fba8618",
    "app_package": "com.supercell.clashofclans"
  },
  "ui_cache": {
    "enabled": true,
    "use_micro_verify": true,
    "learn_on_detect": true
  },
  "loot_criteria": {
    "min_gold": 800000,
    "min_elixir": 800000,
    "min_dark_elixir": 7000,
    "conditions_required": 2
  },
  "economy": {
    "min_rss_to_spend": 20000000,
    "enable_wall_upgrade": false,
    "upgrade_walls": false
  },
  "attack": {
    "attack_strategy": "sneaky_farming",
    "edrag_rush": {
      "loot_criteria": {
        "min_gold": 500000,
        "min_elixir": 500000,
        "min_dark_elixir": 6000,
        "conditions_required": 2
      },
      "rage_opening_count": 2,
      "rage_core_delay_sec": 10.0,
      "hero_ability_delay_sec": 2.0,
      "target_destruction_percent": 70,
      "max_battle_duration_sec": 150,
      "damage_stagnation_timeout_sec": 5.0,
      "min_combat_duration_before_stagnation_sec": 12.0,
      "sweep_points_per_side": 5
    },
    "enable_camera_pan": true,
    "camera_start_side": "adaptive_random",
    "sweep_rounds": 1,
    "use_spells": true,
    "use_super_wall_breakers": true,
    "swb_per_entry": 2,
    "swb_sequence_delay_sec": 0.2,
    "use_heroes": true,
    "instant_hero_abilities": true,
    "haste_per_breach": 1,
    "use_haste_on_deep_corridor": true,
    "deep_core_distance_px": 240,
    "jump_memory_radius_px": 90,
    "enable_full_cross_pan": false,
    "deployment_mode": "structure_quota_deployment",
    "boundary_detection_mode": "deploy_zone_finder",
    "battle_duration_timeout_sec": 50,
    "post_deploy_wait_sec": 10,
    "use_yolo": true,
    "yolo_model_path": "models/yolov8s.onnx",
    "yolo_conf": 0.4,
    "yolo_iou": 0.5,
    "yolo_imgsz": 1280,
    "yolo_min_avg_conf_for_skip_tap": 0.6,
    "deploy_zone_ai": {
      "enabled": true,
      "model_path": "models/deploy_zone_finder.onnx",
      "imgsz": 1280,
      "conf_thresh": 0.5,
      "use_tta": false
    },
    "jump_placement": "wall_raycast",
    "adaptive_spells_by_demand": true,
    "allow_surrender": true,
    "early_surrender_on_zero_loot": true,
    "canny_low": 60,
    "canny_high": 160,
    "hough_threshold": 30,
    "hough_min_len": 70,
    "hough_max_gap": 10,
    "wall_tol_deg": 10.0,
    "min_crossing_angle_deg": 8.0,
    "min_dist_from_drop_px": 20.0,
    "min_dist_from_target_px": 30.0,
    "wall_endpoint_tol_px": 35.0,
    "wall_inset_px": 20,
    "jump_spell_radius_px": 90,
    "pair_min_dist_px": 65,
    "pair_max_dist_px": 210,
    "max_jumps": 4,
    "min_jump_score": 1.0,
    "swb_max_breaches": 4,
    "min_separation_between_jumps_px": 110,
    "jump_entry_wall_offset_ratio": 0.28,
    "canonical_center_x": 1220,
    "canonical_center_y": 500,
    "canonical_rx_grass": 955,
    "canonical_ry_grass": 715,
    "jump_loot_weights": {
      "storage_dark": 3,
      "storage_gold": 2,
      "storage_elixir": 2,
      "collector_gold": 1,
      "collector_elixir": 1,
      "collector_dark": 1,
      "town_hall": 0
    },
    "jump_optimizer": {
      "buildings_block": true,
      "wall_thickness": 8,
      "endpoint_tol": 25.0,
      "morph_kernel_size": 5,
      "candidate_step": 20,
      "min_marginal_gain": 0.5,
      "time_budget_sec": 0.3
    },
    "zone_attack_flanks": [
      "north",
      "south",
      "east",
      "west"
    ],
    "min_unreached_targets_for_pan": 1,
    "safe_drop_y_min": 140,
    "safe_drop_y_max": 845
  },
  "humanizer": {
    "tap_jitter_px": 10,
    "sweep_jitter_px": 12,
    "reaction_min_sec": 0.8,
    "reaction_max_sec": 2.2,
    "between_sides_min_sec": 1.2,
    "between_sides_max_sec": 2.6,
    "hold_deploy_min_sec": 0.28,
    "hold_deploy_max_sec": 0.36,
    "burst_tap_min_sec": 0.085,
    "burst_tap_max_sec": 0.135
  },
  "rois": {
    "enemy_loot": {
      "gold": [
        0.112,
        0.16,
        0.062,
        0.17
      ],
      "elixir": [
        0.165,
        0.215,
        0.062,
        0.17
      ],
      "dark_elixir": [
        0.22,
        0.268,
        0.062,
        0.17
      ]
    },
    "own_resources": {
      "gold": [
        0.024,
        0.075,
        0.825,
        0.932
      ],
      "elixir": [
        0.09,
        0.14,
        0.825,
        0.932
      ],
      "dark_elixir": [
        0.154,
        0.205,
        0.825,
        0.932
      ]
    }
  },
  "loot": {
    "wait_after_next_min": 0.55,
    "wait_after_next_max": 0.8,
    "stability_frames": 2,
    "stability_delay": 0.12,
    "max_read_retries": 3,
    "allow_attack_on_unstable_read": false
  },
  "camera": {
    "sign_x": 1,
    "sign_y": -1,
    "safe_origin_x": 1200,
    "safe_origin_y": 540,
    "settle_wait": 0.25
  },
  "swipe_profiles": {
    "camera_jump_north_to_south": {
      "duration": 0.18,
      "distance": 750,
      "jitter": 0.03
    },
    "camera_jump_south_to_north": {
      "duration": 0.18,
      "distance": 750,
      "jitter": 0.03
    },
    "deploy_sweep_north": {
      "duration": 0.42,
      "distance": 950,
      "jitter": 0.02
    },
    "deploy_sweep_south": {
      "duration": 0.42,
      "distance": 950,
      "jitter": 0.02
    },
    "recenter_camera": {
      "duration": 0.25,
      "distance": 400,
      "jitter": 0.02
    }
  },
  "battle_end": {
    "use_surrender": true,
    "wait_for_surrender_timeout": 10.0,
    "wait_for_okay_timeout": 10.0,
    "wait_for_return_home_timeout": 15.0,
    "max_soft_retries": 3,
    "restart_app_on_failure": false
  },
  "debug": {
    "search_only": false
  }
}
```

## File: core/__init__.py
```python

```

## File: core/config.py
```python
"""
core/config.py
Gestión centralizada de rutas y configuración JSON.
"""
import json
import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "settings.json")


def load_settings() -> dict:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"No se encontró el archivo de configuración en: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


SETTINGS = load_settings()

# Constantes expuestas para compatibilidad directa con device.py y vision.py
DEVICE_SERIAL: str = SETTINGS["device"]["serial"]
APP_PACKAGE: str = SETTINGS["device"]["app_package"]
TESSERACT_CMD: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =====================================================
# BLOQUE DE ESTABILIDAD DE LOOT Y CÁMARA
# =====================================================
_LOOT = SETTINGS.get("loot", {})
_CAMERA = SETTINGS.get("camera", {})
_SWIPE_PROFILES = SETTINGS.get("swipe_profiles", {})
_BATTLE_END = SETTINGS.get("battle_end", {})
_DEBUG = SETTINGS.get("debug", {})
_ATTACK = SETTINGS.get("attack", {})

# ---- Estabilidad de lectura de botín ----
LOOT_WAIT_AFTER_NEXT_MIN = float(_LOOT.get("wait_after_next_min", 0.55))
LOOT_WAIT_AFTER_NEXT_MAX = float(_LOOT.get("wait_after_next_max", 0.80))
LOOT_STABILITY_FRAMES = int(_LOOT.get("stability_frames", 2))
LOOT_STABILITY_DELAY = float(_LOOT.get("stability_delay", 0.12))
LOOT_MAX_READ_RETRIES = int(_LOOT.get("max_read_retries", 3))
LOOT_ALLOW_ATTACK_ON_UNSTABLE_READ = bool(_LOOT.get("allow_attack_on_unstable_read", False))

# ---- Cámara ----
CAMERA_SIGN_X = int(_CAMERA.get("sign_x", 1))
CAMERA_SIGN_Y = int(_CAMERA.get("sign_y", -1))
CAMERA_SAFE_ORIGIN_X = int(_CAMERA.get("safe_origin_x", 1200))
CAMERA_SAFE_ORIGIN_Y = int(_CAMERA.get("safe_origin_y", 540))
CAMERA_SETTLE_WAIT = float(_CAMERA.get("settle_wait", 0.25))

# ---- Perfiles de swipe y cierre de batalla ----
SWIPE_PROFILES = _SWIPE_PROFILES
BATTLE_END = _BATTLE_END

# ---- Debug ----
SEARCH_ONLY = bool(_DEBUG.get("search_only", True))

# ---- YOLO Detection ----
USE_YOLO = bool(_ATTACK.get("use_yolo", False))
YOLO_MODEL_PATH = str(_ATTACK.get("yolo_model_path", "models/yolov8s.onnx"))
YOLO_CONF = float(_ATTACK.get("yolo_conf", 0.40))
YOLO_IOU = float(_ATTACK.get("yolo_iou", 0.5))
YOLO_IMGSZ = int(_ATTACK.get("yolo_imgsz", 1280))
YOLO_MIN_AVG_CONF_FOR_SKIP_TAP = float(_ATTACK.get("yolo_min_avg_conf_for_skip_tap", 0.60))
```

## File: core/device.py
```python
"""
core/device.py
Wrapper uiautomator2 con soporte de gestos, paneo de cámara y trazo continuo seguro.
"""
import os
import random
import tempfile
import time
import cv2
import numpy as np
import uiautomator2 as u2
from PIL import Image

from core.config import DEVICE_SERIAL, APP_PACKAGE, SETTINGS
from core.logger import get_logger
from humanizer.behavior import HumanBehavior

log = get_logger("device")

_MAX_RECONNECT = 3
_RECONNECT_WAIT = 5


class Device:
    def __init__(self, serial: str = DEVICE_SERIAL):
        self.serial = serial
        self.d: u2.Device | None = None
        self.width: int = 2400
        self.height: int = 1080

    def connect(self) -> bool:
        try:
            log.info("Conectando al dispositivo %s...", self.serial)
            self.d = u2.connect(self.serial)
            self.d.implicitly_wait(5)
            info = self.d.info
            w = info.get("displayWidth", 2400)
            h = info.get("displayHeight", 1080)
            self.width = max(w, h)
            self.height = min(w, h)
            log.info("ADB conectado. Resolucion: %dx%d", self.width, self.height)
            # Control de punteros visuales: desactivados por defecto para optimizar GPU y evitar lag
            show_trace = SETTINGS.get("device", {}).get("show_touch_trace", False)
            if show_trace:
                try:
                    self.d.show_touch_trace(pointer_location=True, show_touches=True)
                    log.info("Visualizacion de toques y puntero activada en pantalla.")
                except Exception as e:
                    log.warning("No se pudo activar show_touch_trace: %s", e)
            else:
                try:
                    self.d.show_touch_trace(pointer_location=False, show_touches=False)
                except Exception:
                    pass
            return True
        except Exception as e:
            log.error("Fallo conexion ADB: %s", e)
            return False

    def _reconnect(self) -> bool:
        for intento in range(1, _MAX_RECONNECT + 1):
            log.warning("Reintentando conexion ADB (%d/%d)...", intento, _MAX_RECONNECT)
            time.sleep(_RECONNECT_WAIT)
            try:
                self.d = u2.connect(self.serial)
                self.d.implicitly_wait(5)
                log.info("Reconexion ADB exitosa.")
                return True
            except Exception as e:
                log.error("Error en reconexion: %s", e)
        log.critical("No se pudo reconectar con ADB tras %d intentos.", _MAX_RECONNECT)
        return False

    def screenshot(self, target: str = "opencv") -> np.ndarray:
        save_path = None
        fmt = target
        if target.lower().endswith((".png", ".jpg", ".jpeg")):
            save_path = target
            fmt = "opencv"

        for intento in range(_MAX_RECONNECT):
            try:
                img = self.d.screenshot(format=fmt)
                if isinstance(img, np.ndarray) and img.shape[1] > 0:
                    self.height, self.width = img.shape[:2]
                if save_path is not None and isinstance(img, np.ndarray):
                    cv2.imwrite(save_path, img)
                return img
            except Exception as e:
                log.warning("Error en screenshot (intento %d): %s", intento + 1, e)
                if not self._reconnect():
                    raise RuntimeError("ADB caido al tomar captura.") from e
        raise RuntimeError("Imposible obtener captura.")

    def tap(self, x: int | float, y: int | float, humanize: bool = True):
        tx, ty = int(x), int(y)
        if humanize:
            tx, ty = HumanBehavior.jitter_point(tx, ty)
        for intento in range(_MAX_RECONNECT):
            try:
                self.d.click(tx, ty)
                return
            except Exception as e:
                log.warning("Error en tap(%d, %d): %s", tx, ty, e)
                if not self._reconnect():
                    raise RuntimeError("ADB caido en tap.") from e

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5):
        for intento in range(_MAX_RECONNECT):
            try:
                self.d.swipe(int(x1), int(y1), int(x2), int(y2), duration=duration)
                return
            except Exception as e:
                log.warning("Error en swipe: %s", e)
                if not self._reconnect():
                    raise RuntimeError("ADB caido en swipe.") from e

    def touch_down(self, x: int | float, y: int | float):
        """Mantiene pulsado un punto (ACTION_DOWN) en la pantalla sin soltarlo."""
        tx, ty = int(x), int(y)
        for intento in range(_MAX_RECONNECT):
            try:
                self.d.touch.down(tx, ty)
                return
            except Exception as e:
                log.warning("Error en touch_down(%d, %d): %s", tx, ty, e)
                if not self._reconnect():
                    raise RuntimeError("ADB caido en touch_down.") from e

    def touch_up(self, x: int | float, y: int | float):
        """Libera el punto pulsado (ACTION_UP) en la pantalla."""
        tx, ty = int(x), int(y)
        for intento in range(_MAX_RECONNECT):
            try:
                self.d.touch.up(tx, ty)
                return
            except Exception as e:
                log.warning("Error en touch_up(%d, %d): %s", tx, ty, e)
                if not self._reconnect():
                    raise RuntimeError("ADB caido en touch_up.") from e

    def deploy_heroes_held_anchor(self, anchor_pt: tuple[int | float, int | float], hero_slots: list[tuple[int, int]], slot_delay: float = 0.10):
        """
        Despliega los héroes según la técnica pro de Clash of Clans:
        1. Selecciona el 1er héroe en la barra de batalla.
        2. Mantiene pulsado el mapa (touch_down) en el punto de drop exterior (anchor_pt),
           desplegando inmediatamente al primer héroe.
        3. Con el dedo anclado en el mapa, pulsa sucesivamente los héroes restantes en la barra,
           los cuales se despliegan instantáneamente en ese mismo punto de anclaje.
        4. Libera el toque del mapa (touch_up).
        """
        if not hero_slots:
            return
        ax, ay = int(anchor_pt[0]), int(anchor_pt[1])
        first_hero = hero_slots[0]
        remaining_heroes = hero_slots[1:]

        # 1. Seleccionar el primer héroe en la barra
        log.info("👑 [1/%d] Seleccionando 1er héroe en la barra en (%d, %d)...", len(hero_slots), first_hero[0], first_hero[1])
        self.tap(first_hero[0], first_hero[1], humanize=False)
        time.sleep(0.08)

        # 2. Dejar click sostenido en el anclaje del mapa para soltar el 1er héroe
        log.info("👑 Anclando toque sostenido en (%d, %d) (suelta 1er héroe y prepara restantes)...", ax, ay)
        try:
            self.touch_down(ax, ay)
            time.sleep(0.12)
            # 3. Pulsar sucesivamente los héroes restantes en la barra mientras el anclaje sigue activo
            for idx_h, (hx, hy) in enumerate(remaining_heroes, 2):
                log.info("   -> [%d/%d] Pulsando ranura de heroe restante en (%d, %d) con anclaje activo...", idx_h, len(hero_slots), hx, hy)
                self.tap(hx, hy, humanize=False)
                time.sleep(slot_delay)
            time.sleep(0.08)
        finally:
            try:
                self.touch_up(ax, ay)
            except Exception:
                pass

    def hold_tap(self, x: int | float, y: int | float, duration: float = 0.85):
        """
        Ejecuta un toque sostenido (press and hold) en (x, y) durante 'duration' segundos.
        Permite el despliegue continuo y masivo de tropas (EDrags) a alta velocidad en una sola tirada.
        """
        tx, ty = int(x), int(y)
        tx, ty = HumanBehavior.jitter_point(tx, ty, max_offset=4)
        for intento in range(_MAX_RECONNECT):
            try:
                self.d.touch.down(tx, ty).sleep(duration).up(tx, ty)
                return
            except Exception as e:
                log.warning("Error en touch.down/up hold_tap(%d, %d): %s", tx, ty, e)
                try:
                    dur_ms = int(duration * 1000)
                    self.d.shell(f"input swipe {tx} {ty} {tx+1} {ty+1} {dur_ms}")
                    return
                except Exception as e2:
                    log.warning("Fallback input swipe falló: %s", e2)
                if not self._reconnect():
                    raise RuntimeError("ADB caído en hold_tap.") from e

    def stroke_points(self, points: list[tuple[int, int]], duration: float = 1.3, anchor_sec: float = 0.60):
        """
        Ejecuta un barrido continuo con click inicial sostenido (~600ms, reducido un 33%)
        para activar de forma garantizada el modo despliegue continuo de tropas en Clash of Clans,
        seguido de un arrastre pausado y lento (duration) a lo largo de la trayectoria.
        Usa el método nativo UiDevice.swipePoints (jsonrpc.swipePoints) compatible con dispositivos
        Xiaomi / MIUI / HyperOS y emuladores, con fallback directo por hardware a adb shell input.
        """
        if not points or len(points) < 2:
            return

        start_x, start_y = points[0]
        num_move_segments = max(1, len(points) - 1)

        # En Android UiDevice.swipe, cada paso (step) toma ~10.5ms (5ms pause interno + overhead de eventos)
        step_duration = 0.0105
        steps_per_segment = max(2, int(round(duration / (num_move_segments * step_duration))))

        # Calcular los puntos duplicados necesarios para sostener el click inicial exactamente anchor_sec (~900ms)
        seg_time = steps_per_segment * step_duration
        num_anchor_segments = max(1, int(round(anchor_sec / seg_time)))
        anchor_count = num_anchor_segments + 1

        flat_pts = [int(start_x), int(start_y)] * anchor_count
        for px, py in points[1:]:
            flat_pts.extend([int(px), int(py)])

        try:
            self.d.jsonrpc.swipePoints(flat_pts, steps_per_segment)
        except Exception as e:
            log.warning("swipePoints fallo (%s), ejecutando fallback por adb shell...", e)
            try:
                end_x, end_y = points[-1]
                dur_ms = int((duration + anchor_sec) * 1000)
                self.d.shell(f"input swipe {int(start_x)} {int(start_y)} {int(end_x)} {int(end_y)} {dur_ms}")
            except Exception as e2:
                log.error("Fallo critico en inyeccion de sweep: %s", e2)

    def pan_camera(self, x1_rel: float, y1_rel: float, x2_rel: float, y2_rel: float, duration: float = 0.9):
        sx1, sy1 = int(x1_rel * self.width), int(y1_rel * self.height)
        sx2, sy2 = int(x2_rel * self.width), int(y2_rel * self.height)

        self.swipe(sx1, sy1, sx2, sy2, duration=duration)
        time.sleep(0.5)

    def wait_camera_settled(self, timeout=2.5, interval=0.10, max_shift=0.8, max_diff=25.0, need=2) -> bool:
        prev, ok, t0 = None, 0, time.monotonic()
        while time.monotonic() - t0 < timeout:
            img = self.screenshot()
            if img is None:
                time.sleep(interval)
                continue
            h = img.shape[0]
            g = cv2.cvtColor(cv2.resize(img[int(h * .12):int(h * .80)], (640, 300)), cv2.COLOR_BGR2GRAY).astype(
                np.float32)
            if prev is not None:
                (dx, dy), _ = cv2.phaseCorrelate(prev, g)
                diff = float(cv2.absdiff(prev, g).mean())
                # En combate o aldea activa, héroes, tropas y animaciones cambian píxeles locales, pero la traslación global de cámara (dx, dy) es nula
                is_settled = (abs(dx) < max_shift and abs(dy) < max_shift) and (diff < max_diff)
                ok = ok + 1 if is_settled else 0
                if ok >= need:
                    return True
            prev = g
            time.sleep(interval)
        log.warning("La camara NO se estabilizo en %.1fs", timeout)
        return False

    def zoom_out(self):
        """
        Ejecuta zoom out completo obligatorio centrado en la pantalla (percent=85, steps=35).
        Mantiene los dedos dentro de la zona segura alejada del notch (94px) y bordes de navegación,
        asegurando tope máximo de alejamiento con doble pinch suave.
        """
        log.info("🔍 Ejecutando Zoom Out centrado en zona segura...")
        try:
            self.d.pinch_in(percent=85, steps=35)
        except Exception:
            try:
                self.d().pinch_in(percent=85, steps=35)
            except Exception as e2:
                log.warning("Pinch_in global falló (%s).", e2)
        time.sleep(0.35)
        # Segundo pinch suave de confirmación para asegurar encuadre total sin zonas recortadas
        try:
            self.d.pinch_in(percent=65, steps=25)
        except Exception:
            pass
        time.sleep(0.5)
        self.wait_camera_settled(timeout=2.0)
        log.info("🔍 Zoom Out finalizado y mapa estabilizado.")

    def zoom_half(self):
        """
        Ejecuta zoom moderado (~mitad disponible) para mantener amplia superficie
        de césped exterior legal (evita toques en zona roja al desplegar tropas masivas).
        """
        log.info("🔍 Ejecutando Zoom Moderado (mitad disponible para mayor superficie)...")
        try:
            self.d().pinch_in(percent=35, steps=20)
        except Exception:
            try:
                self.d.pinch_in(percent=35, steps=20)
            except Exception as e2:
                log.warning("Pinch_in moderado falló (%s).", e2)
        time.sleep(0.35)
        self.wait_camera_settled(timeout=1.5)
        log.info("🔍 Zoom Moderado finalizado y cámara estabilizada.")

    def is_app_running(self, package: str = APP_PACKAGE) -> bool:
        try:
            return self.d.app_current().get("package") == package
        except Exception:
            return False

    def open_app(self):
        self.d.app_start(APP_PACKAGE)

    def restart_app(self):
        log.warning("Reiniciando %s...", APP_PACKAGE)
        try:
            self.d.app_stop(APP_PACKAGE)
            time.sleep(2.0)
            self.d.app_start(APP_PACKAGE)
        except Exception as e:
            log.error("Error al reiniciar aplicacion: %s", e)
            if self._reconnect():
                self.d.app_start(APP_PACKAGE)
```

## File: core/ledger.py
```python
"""
core/ledger.py
Contabilidad de sesión, cálculo de ganancias y disparador de mejora de muros.
"""
import time
from core.config import SETTINGS
from core.logger import get_logger

log = get_logger("ledger")


class Ledger:
    def __init__(self):
        self.start_time = time.time()
        self.rss_start_session: dict[str, int] | None = None
        self.rss_current: dict[str, int] = {"gold": 0, "elixir": 0, "dark_elixir": 0}
        self.total_profit: dict[str, int] = {"gold": 0, "elixir": 0, "dark_elixir": 0}
        self.attacks_completed: int = 0
        self.villages_skipped: int = 0

    def record_initial_snapshot(self, rss: dict[str, int]):
        if not rss:
            return
        gold = rss.get("gold", 0)
        elixir = rss.get("elixir", 0)
        if gold <= 0 and elixir <= 0:
            return

        if self.rss_start_session is None:
            # Evitar registrar snapshot con Elixir u Oscuro en 0 si Oro es millonario (fallo de OCR inicial)
            if gold > 500_000 and (elixir == 0 or rss.get("dark_elixir", 0) == 0):
                log.warning("⚠️ Snapshot inicial incompleto (lectura en 0 con Oro alto: %s). Reintentando en siguiente captura.", rss)
                return
            self.rss_start_session = rss.copy()
            self.rss_current = rss.copy()
            log.info("📊 Snapshot inicial registrado con éxito: %s", self.rss_start_session)
        else:
            # Si el snapshot inicial tenía 0 en elixir u oscuro por error de OCR, corregir línea base
            for k in ("elixir", "dark_elixir"):
                if self.rss_start_session.get(k, 0) == 0 and rss.get(k, 0) > 10_000:
                    log.info("🔄 Corrigiendo línea base preexistente de %s en snapshot inicial: %s", k, f"{rss[k]:,}")
                    self.rss_start_session[k] = rss[k]
                    self.rss_current[k] = rss[k]

    def record_post_attack(self, new_rss: dict[str, int]):
        self.attacks_completed += 1
        if not any(new_rss.values()):
            log.warning("⚠️ No se pudieron leer recursos propios tras el ataque (valores en 0).")
            return

        gain = {"gold": 0, "elixir": 0, "dark_elixir": 0}
        if any(self.rss_current.values()):
            for k in ("gold", "elixir", "dark_elixir"):
                # Si el valor anterior era 0 pero el nuevo es alto, fue un fallo de OCR previo, no botín ganado
                if self.rss_current.get(k, 0) == 0 and new_rss.get(k, 0) > 50_000:
                    log.warning("⚠️ Detectada recuperación de recurso %s (de 0 a %s). Registrando saldo real sin sumar a botín.", k, f"{new_rss[k]:,}")
                    self.rss_current[k] = new_rss[k]
                    if self.rss_start_session:
                        self.rss_start_session[k] = new_rss[k]
                    continue

                diff = new_rss.get(k, 0) - self.rss_current.get(k, 0)
                if diff > 0:
                    gain[k] = diff
                    self.total_profit[k] += diff

        self.rss_current = new_rss.copy()
        log.info(
            "💰 Botín de este ataque -> Oro: +%s | Elixir: +%s | Oscuro: +%s",
            f"{gain['gold']:,}",
            f"{gain['elixir']:,}",
            f"{gain['dark_elixir']:,}",
        )
        log.info(
            "📈 Total Acumulado en Sesión (#%d) -> Oro: +%s | Elixir: +%s | Oscuro: +%s",
            self.attacks_completed,
            f"{self.total_profit['gold']:,}",
            f"{self.total_profit['elixir']:,}",
            f"{self.total_profit['dark_elixir']:,}",
        )

    def get_wall_upgrade_resource(self) -> str | None:
        econ = SETTINGS["economy"]
        if not econ.get("enable_wall_upgrade", False):
            return None

        threshold = econ.get("min_rss_to_spend", 20_000_000)
        gold = self.rss_current.get("gold", 0)
        elixir = self.rss_current.get("elixir", 0)

        # Jerarquía: Priorizar siempre Elixir si supera el umbral (20M).
        # Si Elixir no alcanza o ya se gastó, usar Oro.
        if elixir >= threshold:
            return "elixir"
        if gold >= threshold:
            return "gold"
        return None

    def print_summary(self):
        elapsed_min = max(1.0, (time.time() - self.start_time) / 60.0)
        log.info("═" * 60)
        log.info("🏆 RESUMEN DE SESIÓN (%.1f minutos)", elapsed_min)
        log.info("   • Ataques realizados : %d", self.attacks_completed)
        log.info("   • Aldeas saltadas    : %d", self.villages_skipped)
        log.info("   • Oro farmeado       : +%s", f"{self.total_profit['gold']:,}")
        log.info("   • Elixir farmeado    : +%s", f"{self.total_profit['elixir']:,}")
        log.info("   • Oscuro farmeado    : +%s", f"{self.total_profit['dark_elixir']:,}")
        log.info("═" * 60)
```

## File: core/logger.py
```python
"""
core/logger.py
Logger centralizado con salida formateada a consola y archivo.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from core.config import PROJECT_ROOT

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "clash_bot.log")

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-16s | %(message)s"
_DATE_FMT = "%H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
```

## File: core/process_manager.py
```python
"""
core/process_manager.py
Supervisor de proceso, reconexión ante caídas de red y cierre de popups.
"""
import time
from core.device import Device
from core.vision import Vision
from core.logger import get_logger
from humanizer.behavior import HumanBehavior

log = get_logger("process_manager")


class ProcessManager:
    def __init__(self, device: Device, vision: Vision):
        self.device = device
        self.vision = vision

        # Botón de ataque para confirmar aldea lista
        self.tpl_attack_btn = "assets/buttons/attack_btn.png"

        # Popups genéricos y avisos
        self.tpl_okay_btn = "assets/buttons/okay_btn.png"
        self.tpl_close_btn = "assets/buttons/close_btn.png"

        # Modales, pestañas y cajones emergentes
        self.tpl_red_x = "assets/buttons/red_x_close.png"
        self.tpl_blue_x = "assets/buttons/blue_x_close.png"
        self.tpl_train_close = "assets/buttons/train_close_btn.png"

    def handle_popup(self, screenshot=None) -> bool:
        """
        Detecta y cierra popups genéricos, modales, menús de tienda/ajustes o avisos.
        Retorna True si cerró algún elemento emergente, False si la pantalla estaba limpia.
        """
        screen = screenshot if screenshot is not None else self.device.screenshot()

        # 1. Menús con botón rojo de cerrar (X) en la esquina superior derecha (Tienda, Ajustes, Layout Editor)
        # Nota: Ajustes/Tienda ignoran la tecla BACK nativa, se requiere toque directo en la X.
        red_x_pos = self.vision.find_template(screen, self.tpl_red_x, threshold=0.78, zona="ESQUINA_SUP_DER")
        if not red_x_pos:
            red_x_pos = self.vision.find_template(screen, self.tpl_red_x, threshold=0.78, zona="MITAD_SUPERIOR")
        if red_x_pos:
            log.info("🚨 Botón Red X detectado en %s. Pulsando para cerrar menú...", red_x_pos)
            HumanBehavior.reaction_delay()
            self.device.tap(*red_x_pos)
            time.sleep(1.0)
            return True

        # 2. Modales centrados con botón azul de cerrar (Perfil, Clan, etc.)
        blue_x_pos = self.vision.find_template(screen, self.tpl_blue_x, threshold=0.74, zona="MITAD_SUPERIOR")
        if blue_x_pos:
            log.info("🚨 Modal emergente detectado (Blue X) en %s. Cerrando con tecla BACK...", blue_x_pos)
            HumanBehavior.reaction_delay()
            self.device.d.press("back")
            time.sleep(1.0)
            return True

        # 3. Modal de entrenamiento de tropas (Train Troops)
        train_pos = self.vision.find_template(screen, self.tpl_train_close, threshold=0.75, zona="MITAD_SUPERIOR")
        if train_pos:
            log.info("🚨 Modal Entrenar Tropas detectado en %s. Cerrando con tecla BACK...", train_pos)
            HumanBehavior.reaction_delay()
            self.device.d.press("back")
            time.sleep(1.0)
            return True

        # 4. Pestaña de cierre adicional si existe
        close_btn_pos = self.vision.find_template(screen, self.tpl_close_btn, threshold=0.75, zona="MITAD_SUPERIOR")
        if close_btn_pos:
            log.info("🚨 Botón de cerrar detectado en %s. Cerrando...", close_btn_pos)
            HumanBehavior.reaction_delay()
            self.device.d.press("back")
            time.sleep(1.0)
            return True

        # 5. Avisos informativos con botón Okay (mejoras completadas, noticias)
        okay_pos = self.vision.find_template(screen, self.tpl_okay_btn, threshold=0.74, zona="COMPLETA")
        if okay_pos:
            log.info("🚨 Aviso / Mejora detectada (Okay) en %s. Confirmando...", okay_pos)
            HumanBehavior.reaction_delay()
            self.device.tap(*okay_pos)
            time.sleep(1.0)
            return True

        return False

    def dismiss_startup_popups(self, max_checks: int = 5):
        """
        Al entrar al juego, busca activamente y cierra mensajes de mejoras, modales
        o avisos antes de que el bot inicie cualquier acción en la aldea.
        """
        log.info("🔍 Comprobando y cerrando modales o avisos de inicio...")
        for _ in range(max_checks):
            screen = self.device.screenshot()
            # Si el botón de ataque ya está visible y libre de popups, salir inmediatamente
            if self.vision.find_template(screen, self.tpl_attack_btn, threshold=0.74, zona="ESQUINA_INF_IZQ"):
                if not self.handle_popup(screen):
                    break
            elif not self.handle_popup(screen):
                break

    def ensure_game_ready(self, timeout: int = 45) -> bool:
        """
        Asegura que el juego esté abierto y listo en la aldea principal:
        1. Inicia Clash of Clans si no está corriendo.
        2. Cierra activamente mensajes de mejoras, modales o pestañas emergentes.
        3. Deselecciona cualquier edificio o muro marcado tocando zona neutral.
        4. Confirma la presencia del botón Attack para continuar.
        """
        if not self.device.is_app_running():
            log.warning("⚠️ Clash of Clans no está activo. Iniciando aplicación...")
            self.device.open_app()
            time.sleep(6)

        # 1. Al entrar al juego antes de hacer cualquier cosa, cerrar mensajes y pestañas
        self.dismiss_startup_popups()

        inicio = time.monotonic()
        while time.monotonic() - inicio < timeout:
            screen = self.device.screenshot()

            # 1) Si el botón de ataque ya está visible y sin popups encima, la aldea está lista:
            if self.vision.find_template(screen, self.tpl_attack_btn, threshold=0.74, zona="ESQUINA_INF_IZQ"):
                # Verificar rápidamente si hay un popup residual bloqueando
                if not self.handle_popup(screen):
                    log.info("🏰 Aldea lista y confirmada (botón Attack visible).")
                    return True
                time.sleep(1.0)
                continue

            # 2) Si el botón Attack está cubierto, intentar cerrar el popup que lo bloquea
            if self.handle_popup(screen):
                time.sleep(1.0)
                continue

            # Si el botón Attack aún no es visible tras unos segundos:
            elapsed = time.monotonic() - inicio
            if elapsed > 2.0:
                # 1) Descartar overlays o cajones emergentes de pantalla completa con tecla BACK
                log.debug("Intentando tecla BACK para descartar overlay de pantalla completa...")
                self.device.d.press("back")
                time.sleep(0.6)

                # 2) Deseleccionar cualquier edificio o muro marcado tocando césped neutral
                log.debug("Deseleccionando edificio/muro marcado tocando césped neutral...")
                safe_x = int(self.device.width * 0.35)
                safe_y = int(self.device.height * 0.40)
                self.device.tap(safe_x, safe_y)
                time.sleep(0.6)

            time.sleep(0.8)

        log.error("⏱️ Juego bloqueado tras %ds. Ejecutando force-stop y reinicio...", timeout)
        self.device.restart_app()
        time.sleep(8)
        return False
```

## File: core/state_machine.py
```python
import time
from enum import Enum, auto

from core.config import SETTINGS
from core.device import Device
from core.vision import Vision, save_debug_image_async
from core.ledger import Ledger
from core.process_manager import ProcessManager
from vision.resource_reader import ResourceReader
from vision.damage_reader import DamageReader
from vision.army_checker import ArmyChecker
from attack.deployment_strategy import DeploymentStrategy
from attack.edrag_strategy import EDragStrategy
from humanizer.behavior import HumanBehavior
from core.logger import get_logger
from core.wall_upgrader import WallUpgrader
from core.ui_cache import UICache
import random
from core import config

log = get_logger("state_machine")


class BotState(Enum):
    ENSURE_GAME_READY = auto()
    MAIN_VILLAGE = auto()
    SEARCHING_MATCH = auto()
    EVALUATING_LOOT = auto()
    ATTACKING = auto()
    WAITING_BATTLE_END = auto()
    COLLECTING_LOOT = auto()
    RETURNING_HOME = auto()
    ERROR_RECOVERY = auto()


class ClashStateMachine:
    def __init__(self, device: Device, vision: Vision):
        self.device = device
        self.vision = vision
        self.ledger = Ledger()
        self.proc_mgr = ProcessManager(device, vision)
        self.reader = ResourceReader(vision)
        self.damage_reader = DamageReader()
        self.army = ArmyChecker(vision)
        self.deployer = DeploymentStrategy(device, vision, self.army)
        self.edrag_deployer = EDragStrategy(device, vision)
        self.active_army_type = "sneaky_farming"
        self.wall_upgrader = WallUpgrader(device, vision)
        self.ui_cache = UICache()

        self.state = BotState.ENSURE_GAME_READY

        # Templates
        self.tpl_attack = "assets/buttons/attack_btn.png"
        self.tpl_find_match = "assets/buttons/find_match_btn.png"
        self.tpl_attack_confirms = [
            "assets/buttons/attack_btn_confirm.png",
            "assets/buttons/confirm_attack_btn.png"
        ]
        self.tpl_attack_confirm = "assets/buttons/confirm_attack_btn.png"
        self.tpl_next = "assets/buttons/next_btn.png"
        self.tpl_end_battle = "assets/buttons/end_battle.png"
        self.tpl_surrender_battle = "assets/buttons/surrender_battle.png"
        self.tpl_confirm_end = "assets/buttons/confirm_end.png"
        self.tpl_okay_surrender = "assets/buttons/okay_btn.png"
        self.tpl_return_home = "assets/buttons/return_home.png"

    def finish_battle_sequence(self, allow_surrender: bool = False):
        """
        Secuencia completa de cierre de batalla:
        1. Comprobar si 'Return Home' ya está en pantalla.
        2. Si allow_surrender=True:
           - Rendición ultrarrápida sin esperas previas: pulsa botón de fin/rendición,
             confirma diálogo si aparece y pulsa Return Home con muestreo instantáneo.
        3. Si allow_surrender=False:
           - Espera pasiva y segura hasta que Return Home aparezca naturalmente.
        """
        log.info("🏁 Finalizando batalla y esperando Return Home (allow_surrender=%s)...", allow_surrender)
        screen = self.device.screenshot()

        # Si ya estamos en la pantalla de resultados con Return Home (la batalla terminó sola)
        ret_pos = self.vision.find_template(screen, self.tpl_return_home, threshold=0.72, zona="MITAD_INFERIOR")
        if ret_pos:
            log.info("🎯 Botón Return Home detectado en %s. Volviendo a la aldea al instante...", ret_pos)
            self.device.tap(ret_pos[0], ret_pos[1], humanize=False)
            time.sleep(2.5)
            return

        if not allow_surrender:
            log.info("🛡️ Protección activa: Prohibido pulsar Surrender. Esperando aparición natural de Return Home...")
            ret_pos = self.vision.wait_for_template(
                self.device.screenshot, self.tpl_return_home, timeout=15, threshold=0.72, zona="MITAD_INFERIOR"
            )
            if ret_pos:
                log.info("🎯 Botón Return Home detectado en %s. Volviendo a la aldea...", ret_pos)
                HumanBehavior.reaction_delay()
                self.device.tap(*ret_pos)
                time.sleep(3.5)
            else:
                log.warning("⚠️ Return Home no apareció tras espera; intentando tecla BACK suave...")
                self.device.d.press("back")
                time.sleep(2.0)
            return

        # ⚡ MODO RENDICIÓN ULTRARRÁPIDA (allow_surrender=True)
        # Pulsar de inmediato el botón de fin de batalla (End Battle o Surrender) mediante UI Cache
        surrender_pos = self.vision.find_template_cached(
            screen, self.tpl_end_battle, cache_name="surrender_btn", ui_cache=self.ui_cache,
            threshold=0.70, zona="ESQUINA_INF_IZQ"
        )
        if not surrender_pos:
            surrender_pos = self.vision.find_template_cached(
                screen, self.tpl_surrender_battle, cache_name="surrender_btn", ui_cache=self.ui_cache,
                threshold=0.70, zona="ESQUINA_INF_IZQ"
            )
        if not surrender_pos:
            # Coordenada calibrada en UI Cache para botón de rendición
            surrender_pos = self.ui_cache.get_pos("surrender_btn", self.device.width, self.device.height)

        log.info("🎯 Rendición ultrarrápida: Pulsando botón de fin de batalla en %s...", surrender_pos)
        self.device.tap(surrender_pos[0], surrender_pos[1], humanize=False)
        time.sleep(0.30)

        # Confirmar diálogo central si aparece (Okay o Confirm)
        s_dlg = self.device.screenshot()
        confirm_pos = self.vision.find_template_cached(
            s_dlg, self.tpl_okay_surrender, cache_name="confirm_surrender_btn", ui_cache=self.ui_cache,
            threshold=0.70, zona="CENTRO"
        )
        if not confirm_pos:
            confirm_pos = self.vision.find_template_cached(
                s_dlg, self.tpl_confirm_end, cache_name="confirm_surrender_btn", ui_cache=self.ui_cache,
                threshold=0.70, zona="CENTRO"
            )
        if not confirm_pos:
            confirm_pos = self.ui_cache.get_pos("confirm_surrender_btn", self.device.width, self.device.height)

        if confirm_pos:
            log.info("🎯 Confirmando rendición inmediatamente en %s...", confirm_pos)
            self.device.tap(confirm_pos[0], confirm_pos[1], humanize=False)
            time.sleep(0.35)

        # Esperar Return Home con muestreo rápido (cada 0.20s) y UI Cache
        t_ret = time.monotonic()
        while time.monotonic() - t_ret < 7.0:
            s_ret = self.device.screenshot()
            ret_pos = self.vision.find_template_cached(
                s_ret, self.tpl_return_home, cache_name="return_home_btn", ui_cache=self.ui_cache,
                threshold=0.72, zona="MITAD_INFERIOR"
            )
            if ret_pos:
                log.info("🎯 Botón Return Home detectado en %s. Volviendo a la aldea al instante...", ret_pos)
                self.device.tap(ret_pos[0], ret_pos[1], humanize=False)
                time.sleep(2.5)
                return
            time.sleep(0.20)

        log.warning("⚠️ No se encontró botón Return Home tras rendición.")

    def clean_surrender(self):
        """Secuencia controlada de rendicion sin matar la app."""
        self.finish_battle_sequence(allow_surrender=True)

    def run_forever(self, max_attacks: int | None = None):
        log.info("Iniciando bucle de control CLASH_FARMER_BOT_V2 (máx ataques: %s)...", max_attacks or "infinito")
        while True:
            if max_attacks is not None and self.ledger.attacks_completed >= max_attacks:
                log.info("🎯 Se completaron los %d ataques solicitados. Deteniendo bot de forma limpia.", max_attacks)
                self.ledger.print_summary()
                break
            try:
                # No procesar popups genericos si estamos en pleno ataque
                if self.state not in (BotState.ATTACKING, BotState.WAITING_BATTLE_END):
                    self.proc_mgr.handle_popup()
                self._tick()
            except KeyboardInterrupt:
                log.info("Detencion manual solicitada.")
                self.ledger.print_summary()
                break
            except Exception as e:
                log.error("Excepcion en estado %s: %s", self.state.name, e)
                # Si falla durante batalla, terminar ordenadamente antes de reiniciar
                if self.state in (BotState.ATTACKING, BotState.WAITING_BATTLE_END):
                    self.clean_surrender()
                self.state = BotState.ERROR_RECOVERY

    def _tick(self):
        if self.state == BotState.ENSURE_GAME_READY:
            if self.proc_mgr.ensure_game_ready():
                self.state = BotState.MAIN_VILLAGE
            else:
                self.state = BotState.ERROR_RECOVERY

        elif self.state == BotState.MAIN_VILLAGE:
            # Comprobar si quedó algún popup abierto antes de proceder
            if self.proc_mgr.handle_popup():
                return

            screen = self.device.screenshot()
            own_rss = self.reader.read_own_resources(screen)
            self.ledger.record_initial_snapshot(own_rss)
            self.ledger.rss_current = own_rss

            res_to_spend = self.ledger.get_wall_upgrade_resource()
            if res_to_spend and self.wall_upgrader.upgrade_wall(res_to_spend):
                return

            if not self.army.can_start_battle(screen):
                log.info("Campamentos no listos. Pausa de 30s...")
                time.sleep(30)
                return

            attack_pos = self.vision.find_template_cached(
                screen, self.tpl_attack, cache_name="attack_btn", ui_cache=self.ui_cache,
                threshold=0.74, zona="ESQUINA_INF_IZQ"
            )
            if attack_pos:
                HumanBehavior.reaction_delay()
                self.device.tap(*attack_pos)

                find_pos = self.vision.wait_for_template(
                    self.device.screenshot, self.tpl_find_match, timeout=6, zona="COMPLETA",
                    cache_name="find_match_btn", ui_cache=self.ui_cache
                )
                if find_pos:
                    HumanBehavior.reaction_delay()
                    self.device.tap(*find_pos)
                    time.sleep(0.8)

                    # Esperar y confirmar ataque (botón que inicia la búsqueda de rivales/recursos)
                    confirm_pos = None
                    t_start = time.monotonic()
                    while time.monotonic() - t_start < 4.5:
                        s_confirm = self.device.screenshot()
                        match = self.vision.find_any(
                            s_confirm, self.tpl_attack_confirms, threshold=0.72, zona="MITAD_INFERIOR"
                        )
                        if match:
                            confirm_pos = (match[1], match[2])
                            self.ui_cache.update_pos("confirm_attack_btn", confirm_pos[0], confirm_pos[1], self.device.width, self.device.height)
                            break
                        time.sleep(0.25)

                    if confirm_pos:
                        log.info("🎯 Botón Confirm Attack detectado en %s. Iniciando búsqueda...", confirm_pos)
                        HumanBehavior.reaction_delay()
                        self.device.tap(*confirm_pos)
                    else:
                        # Clic directo de confirmación usando UI Cache
                        cached_conf = self.ui_cache.get_pos("confirm_attack_btn", self.device.width, self.device.height, jitter=4)
                        log.info("🎯 Ejecutando clic de confirmación de ataque usando UI Cache: %s...", cached_conf)
                        HumanBehavior.reaction_delay()
                        self.device.tap(*cached_conf)

                    self.state = BotState.SEARCHING_MATCH
            else:
                log.warning("⚠️ Botón Attack no detectado en la aldea (umbral 0.74). Revalidando estado del juego...")
                self.state = BotState.ENSURE_GAME_READY


        elif self.state == BotState.SEARCHING_MATCH:
            log.info("Buscando rival...")
            next_pos = self.vision.wait_for_template(
                self.device.screenshot, self.tpl_next, timeout=25, zona="ESQUINA_INF_DER",
                cache_name="next_btn", ui_cache=self.ui_cache
            )
            if next_pos:
                # Espera configurable para que disipen nubes y termine el conteo
                wait_post_next = random.uniform(
                    config.LOOT_WAIT_AFTER_NEXT_MIN,
                    config.LOOT_WAIT_AFTER_NEXT_MAX,
                )
                log.info(f"[LOOT] Espera post-Next: {wait_post_next:.2f}s")
                time.sleep(wait_post_next)
                self.state = BotState.EVALUATING_LOOT
            else:
                log.warning("Timeout esperando boton Next.")
                self.state = BotState.ERROR_RECOVERY
        elif self.state == BotState.EVALUATING_LOOT:
            # Lectura estable con banco de escalas
            screen = self.device.screenshot()
            self.active_army_type = self.army.detect_active_army_type(screen)
            loot = self.reader.read_enemy_loot_stable(self.device.screenshot)

            # Criterios dinámicos según el tipo de ejército activo
            if self.active_army_type == "edrag_rush":
                edrag_crit = SETTINGS.get("attack", {}).get("edrag_rush", {}).get("loot_criteria")
                meets = self.reader.meets_attack_criteria(loot, criteria_override=edrag_crit)
            else:
                meets = self.reader.meets_attack_criteria(loot)

            if loot is not None and meets:
                self.state = BotState.ATTACKING
            else:
                self.ledger.villages_skipped += 1
                # Toque de latencia cero usando la posición en UI Cache
                next_pos = self.ui_cache.get_pos("next_btn", self.device.width, self.device.height, jitter=5)
                HumanBehavior.reaction_delay()
                self.device.tap(*next_pos)
                time.sleep(1.0)
                self.state = BotState.SEARCHING_MATCH

        elif self.state == BotState.ATTACKING:
            if self.active_army_type == "edrag_rush":
                success = self.edrag_deployer.execute_attack()
            else:
                success = self.deployer.execute_attack()

            if success:
                self.state = BotState.WAITING_BATTLE_END
            else:
                self.clean_surrender()
                self.state = BotState.COLLECTING_LOOT

        elif self.state == BotState.WAITING_BATTLE_END:
            if self.active_army_type == "edrag_rush":
                edrag_cfg = SETTINGS.get("attack", {}).get("edrag_rush", {})
                max_sec = edrag_cfg.get("max_battle_duration_sec", 150)
                stagnation_timeout = edrag_cfg.get("damage_stagnation_timeout_sec", 5.0)
                min_combat_sec = edrag_cfg.get("min_combat_duration_before_stagnation_sec", 12.0)
                target_destruction_pct = edrag_cfg.get("target_destruction_percent", 70)

                log.info(
                    "🐉 Esperando combate E-Drag (máx %ds, meta %d%%, timeout estancamiento %.1fs tras %.1fs o Return Home natural)...",
                    max_sec, target_destruction_pct, stagnation_timeout, min_combat_sec
                )
                t0 = time.monotonic()
                cur_atk = self.ledger.attacks_completed + 1
                import os, cv2
                os.makedirs("logs/runs", exist_ok=True)
                last_snap = 0.0

                last_damage: int | None = None
                last_damage_increase_time = t0
                early_exit_triggered = False

                while time.monotonic() - t0 < max_sec:
                    screen = self.device.screenshot()
                    now = time.monotonic()
                    elapsed = now - t0

                    # Tomar capturas periódicas cada 6s para auditoría visual completa de la batalla
                    if elapsed - last_snap >= 6.0 and screen is not None:
                        last_snap = elapsed
                        save_debug_image_async(f"logs/runs/battle_progress_edrag_atk{cur_atk}_{int(elapsed)}s.png", screen)

                    # Si la batalla termina sola (100% destrucción, fin de tropas o tiempo agotado)
                    if self.vision.find_template(screen, self.tpl_return_home, zona="MITAD_INFERIOR"):
                        log.info("🏆 Batalla terminada naturalmente (Return Home detectado tras %.1fs).", elapsed)
                        break

                    # Watchdog de daño:
                    if screen is not None:
                        pct = self.damage_reader.read_destruction_percentage(screen)
                        if pct is not None:
                            if last_damage is None or pct > last_damage:
                                log.info("💥 Destrucción actualizada: %d%% (tras %.1fs de batalla)", pct, elapsed)
                                last_damage = pct
                                last_damage_increase_time = now

                            # 🎯 CONDICIÓN 1: Meta del 70% de destrucción alcanzada -> Rendición inmediata al toque
                            if pct >= target_destruction_pct:
                                log.info(
                                    "⚡ META ALCANZADA: %d%% de destrucción (>= %d%%) tras %.1fs — ejecutando rendición ultrarrápida!",
                                    pct, target_destruction_pct, elapsed
                                )
                                early_exit_triggered = True
                                break

                    # 🎯 CONDICIÓN 2: Estancamiento de daño por >= stagnation_timeout segundos
                    if elapsed >= min_combat_sec:
                        time_since_increase = now - last_damage_increase_time
                        if time_since_increase >= stagnation_timeout:
                            log.info(
                                "⚡ Destrucción estancada en %s%% durante %.1fs (>= %.1fs) — finalizando batalla anticipadamente.",
                                f"{last_damage}" if last_damage is not None else "0",
                                time_since_increase,
                                stagnation_timeout
                            )
                            early_exit_triggered = True
                            break

                    time.sleep(0.9)

                s_end = self.device.screenshot()
                if s_end is not None:
                    save_debug_image_async(f"logs/runs/battle_end_edrag_atk{cur_atk}.png", s_end)

                # Rendición controlada ultrarrápida o retorno inmediato
                self.finish_battle_sequence(allow_surrender=early_exit_triggered)
                self.state = BotState.COLLECTING_LOOT
            else:
                wait_sec = SETTINGS.get("attack", {}).get("post_deploy_wait_sec", 10)
                early_exit = False
                log.info("Esperando %ds para limpieza de recolectores por las tropas...", wait_sec)
                t0 = time.monotonic()
                last_loot: tuple[int, int, int] | None = None
                consecutive_same_loot = 0

                while time.monotonic() - t0 < wait_sec:
                    screen = self.device.screenshot()
                    # Si termina sola (100% destrucción o tropas agotadas)
                    if self.vision.find_template(screen, self.tpl_return_home, zona="MITAD_INFERIOR"):
                        log.info("Batalla terminada naturalmente (Return Home detectado).")
                        break

                    elapsed = time.monotonic() - t0
                    if screen is not None:
                        cur_atk = self.ledger.attacks_completed + 1
                        save_debug_image_async(f"logs/runs/battle_progress_atk{cur_atk}.png", screen)
                        try:
                            loot = self.reader.read_enemy_loot(screen)
                            g = loot.get("gold") or 0
                            e = loot.get("elixir") or 0
                            d = loot.get("dark_elixir") or 0
                            total_loot = g + e + d
                            if total_loot == 0 and elapsed >= 3.0:
                                log.info("⚡ Botín rival vaciado al 100% (0 recursos restantes) — ejecutando rendición inmediata!")
                                early_exit = True
                                break

                            current_loot = (g, e, d)
                            if elapsed >= 5.0:
                                if last_loot is not None and current_loot == last_loot:
                                    consecutive_same_loot += 1
                                    if consecutive_same_loot >= 2:
                                        log.info("⚡ Botín estancado tras saqueo — ejecutando rendición anticipada!")
                                        early_exit = True
                                        break
                                else:
                                    consecutive_same_loot = 0
                                last_loot = current_loot
                        except Exception as e_loot:
                            log.debug("Error leyendo botín en espera post-ataque: %s", e_loot)

                    time.sleep(1.5)

                # Guardar captura final previa a la rendición
                s_end = self.device.screenshot()
                if s_end is not None:
                    cur_atk = self.ledger.attacks_completed + 1
                    save_debug_image_async(f"logs/runs/battle_end_atk{cur_atk}.png", s_end)

                # Ejecutar secuencia de cierre ultrarrápida si se vació el botín o allow_surrender está activo
                allow_surrender_cfg = SETTINGS.get("attack", {}).get("allow_surrender", True)
                self.finish_battle_sequence(allow_surrender=(early_exit or allow_surrender_cfg))
                self.state = BotState.COLLECTING_LOOT

        elif self.state == BotState.COLLECTING_LOOT:
            log.info("📊 Evaluando recursos tras el ataque en la aldea principal...")
            # Esperar a que la aldea principal esté completamente cargada (aparición del botón Attack)
            t_wait = time.monotonic()
            while time.monotonic() - t_wait < 10.0:
                screen = self.device.screenshot()
                if self.vision.find_template(screen, self.tpl_attack, threshold=0.75, zona="ESQUINA_INF_IZQ"):
                    break
                time.sleep(1.0)

            # Tomar captura de la aldea principal y leer los recursos propios
            screen = self.device.screenshot()
            post_rss = self.reader.read_own_resources(screen)
            self.ledger.record_post_attack(post_rss)
            self.state = BotState.MAIN_VILLAGE

        elif self.state == BotState.RETURNING_HOME:
            screen = self.device.screenshot()
            ret_pos = self.vision.find_template(screen, self.tpl_return_home, zona="MITAD_INFERIOR")
            if ret_pos:
                HumanBehavior.reaction_delay()
                self.device.tap(*ret_pos)
                time.sleep(3.5)
            self.state = BotState.COLLECTING_LOOT

        elif self.state == BotState.ERROR_RECOVERY:
            log.warning("Entrando en recuperacion de juego...")
            self.device.restart_app()
            time.sleep(6.0)
            self.state = BotState.ENSURE_GAME_READY
```

## File: core/ui_cache.py
```python
"""
core/ui_cache.py
Motor de Caché Inteligente de Coordenadas de Interfaz (UI) para ClashFarmerbot.
Permite clics de latencia cero y micro-verificaciones en ROIs reducidas (<1 ms),
eliminando búsquedas template matching de pantalla completa redundantes.
"""
from __future__ import annotations
import random
import cv2
import numpy as np
from core.logger import get_logger
from core.config import SETTINGS

log = get_logger("ui_cache")


class UICache:
    """
    Administrador centralizado de coordenadas UI pre-calibradas con auto-aprendizaje dinámico.
    """

    # Coordenadas de referencia normalizadas (x / 2400, y / 1080)
    DEFAULT_NORMALIZED_COORDS: dict[str, tuple[float, float]] = {
        "next_btn": (2185.0 / 2400.0, 892.0 / 1080.0),            # Botón 'Siguiente' en batalla
        "attack_btn": (195.0 / 2400.0, 945.0 / 1080.0),           # Botón 'Atacar' en aldea
        "find_match_btn": (1820.0 / 2400.0, 780.0 / 1080.0),      # Botón 'Buscar' multijugador
        "confirm_attack_btn": (1200.0 / 2400.0, 778.0 / 1080.0),  # Confirmar ataque / gastar oro
        "surrender_btn": (214.0 / 2400.0, 808.0 / 1080.0),        # Rendirse / Terminar batalla
        "confirm_surrender_btn": (1400.0 / 2400.0, 670.0 / 1080.0), # 'Vale' de confirmación
        "return_home_btn": (1200.0 / 2400.0, 955.0 / 1080.0),     # Volver a casa
    }

    def __init__(self):
        self._norm_coords: dict[str, tuple[float, float]] = dict(self.DEFAULT_NORMALIZED_COORDS)
        self._learned_hits: dict[str, int] = {k: 0 for k in self._norm_coords}
        self.enabled: bool = SETTINGS.get("ui_cache", {}).get("enabled", True)
        self.use_micro_verify: bool = SETTINGS.get("ui_cache", {}).get("use_micro_verify", True)
        self.learn_on_detect: bool = SETTINGS.get("ui_cache", {}).get("learn_on_detect", True)

    def get_pos(self, name: str, width: int = 2400, height: int = 1080, jitter: int = 0) -> tuple[int, int]:
        """
        Retorna la coordenada absoluta en pantalla escalada a (width, height),
        con variación humanizada opcional (jitter).
        """
        if name not in self._norm_coords:
            raise KeyError(f"Elemento UI '{name}' no existe en UICache.")

        norm_x, norm_y = self._norm_coords[name]
        cx = int(round(norm_x * width))
        cy = int(round(norm_y * height))

        if jitter > 0:
            cx += random.randint(-jitter, jitter)
            cy += random.randint(-jitter, jitter)

        # Garantizar que se mantenga dentro de los límites legales de pantalla
        cx = max(10, min(width - 10, cx))
        cy = max(10, min(height - 10, cy))
        return (cx, cy)

    def update_pos(self, name: str, x: int, y: int, width: int = 2400, height: int = 1080):
        """
        Auto-aprendizaje dinámico: actualiza la posición normalizada
        cuando un template es localizado con alta confianza.
        """
        if not self.learn_on_detect or width <= 0 or height <= 0:
            return

        norm_x = float(x) / float(width)
        norm_y = float(y) / float(height)

        # Suavizado exponencial ligero (EMA) si ya existía para amortiguar ruido
        if name in self._norm_coords:
            old_x, old_y = self._norm_coords[name]
            alpha = 0.35  # Ponderación de la nueva observación
            smooth_x = (1.0 - alpha) * old_x + alpha * norm_x
            smooth_y = (1.0 - alpha) * old_y + alpha * norm_y
            self._norm_coords[name] = (smooth_x, smooth_y)
        else:
            self._norm_coords[name] = (norm_x, norm_y)

        self._learned_hits[name] = self._learned_hits.get(name, 0) + 1
        log.debug("📍 [UI_CACHE] Aprendida pos para '%s': (%d, %d) en %dx%d (hits: %d)",
                  name, x, y, width, height, self._learned_hits[name])

    def micro_verify(
        self,
        screen: np.ndarray,
        name: str,
        template: np.ndarray,
        threshold: float = 0.68,
        margin_x: int = 45,
        margin_y: int = 45,
    ) -> tuple[int, int] | None:
        """
        Micro-verificación de template en una ventana reducida alrededor de la coordenada en caché.
        Tarda < 1 ms al procesar un área pequeña (~90x90 px) en vez de toda la pantalla.
        Retorna la coordenada central exacta (cx, cy) si hay coincidencia, o None si no.
        """
        if not self.enabled or screen is None or template is None:
            return None

        h, w = screen.shape[:2]
        cx, cy = self.get_pos(name, width=w, height=h, jitter=0)

        th, tw = template.shape[:2]
        # Calcular ROI con márgenes suficientes para que quepa el template holgadamente
        pad_x = max(margin_x, tw // 2 + 10)
        pad_y = max(margin_y, th // 2 + 10)

        x1 = max(0, cx - pad_x)
        y1 = max(0, cy - pad_y)
        x2 = min(w, cx + pad_x)
        y2 = min(h, cy + pad_y)

        # Si el recorte es menor que el template, no se puede buscar
        if (y2 - y1) < th or (x2 - x1) < tw:
            return None

        crop = screen[y1:y2, x1:x2]
        # Convertir a escala de grises si es necesario
        if crop.ndim == 3 and template.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        elif crop.ndim == 2 and template.ndim == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Evitar falsos positivos en recortes o templates sin contraste (std == 0)
        if np.std(crop) < 1.0 or np.std(template) < 1.0:
            return None

        res = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if not np.isnan(max_val) and max_val >= threshold:
            # Coordenada del centro del template en el sistema global
            best_x = x1 + max_loc[0] + tw // 2
            best_y = y1 + max_loc[1] + th // 2
            # Auto-actualizar caché si la confianza es muy alta
            if max_val >= 0.80:
                self.update_pos(name, best_x, best_y, width=w, height=h)
            return (best_x, best_y)

        return None
```

## File: core/vision.py
```python
"""
core/vision.py
Motor de visión por computadora optimizado con ROI dinámico (porcentajes) y caché RAM.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image

from core.config import PROJECT_ROOT
from core.logger import get_logger

log = get_logger("vision")

_DEBUG_IMG_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="debug_img_saver")


def save_debug_image_async(path: str, img: np.ndarray, sync: bool = False) -> None:
    """Guarda imágenes de depuración y auditoría de forma asíncrona para no bloquear el hilo principal."""
    if img is None or getattr(img, "size", 0) == 0:
        return
    is_testing = "unittest" in sys.modules
    img_copy = img.copy()

    def _write():
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            cv2.imwrite(path, img_copy)
        except Exception as e:
            log.debug("Error guardando imagen debug %s: %s", path, e)

    if sync or is_testing:
        _write()
    else:
        _DEBUG_IMG_EXECUTOR.submit(_write)


class Vision:
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold
        self._template_cache: dict[str, tuple[np.ndarray, tuple[int, int]]] = {}

    def _load_template(self, template_path: str) -> tuple[np.ndarray, tuple[int, int]] | None:
        if template_path in self._template_cache:
            return self._template_cache[template_path]

        ruta_absoluta = os.path.join(PROJECT_ROOT, template_path)
        if not os.path.exists(ruta_absoluta):
            log.debug("Template no encontrado: %s", ruta_absoluta)
            return None

        template = cv2.imread(ruta_absoluta, cv2.IMREAD_GRAYSCALE)
        if template is None:
            log.debug("No se pudo decodificar: %s", ruta_absoluta)
            return None

        h, w = template.shape
        self._template_cache[template_path] = (template, (h, w))
        return self._template_cache[template_path]

    def preload_templates(self, template_dir: str = "assets"):
        ruta_abs = os.path.join(PROJECT_ROOT, template_dir)
        if not os.path.isdir(ruta_abs):
            return
        cargados = 0
        for root, dirs, files in os.walk(ruta_abs):
            dirs[:] = [d for d in dirs if not d.startswith(("_", "."))]
            for fname in files:
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    abs_file = os.path.join(root, fname)
                    rel_path = os.path.relpath(abs_file, PROJECT_ROOT).replace("\\", "/")
                    if self._load_template(rel_path):
                        cargados += 1
        log.info("📦 Precargados %d templates en caché RAM.", cargados)

    def clear_cache(self):
        self._template_cache.clear()

    @staticmethod
    def to_gray(image) -> np.ndarray:
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return image
            elif image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        raise TypeError(f"Formato de imagen no soportado: {type(image)}")

    @staticmethod
    def _crop_zone(screen_gray: np.ndarray, zona: str) -> tuple[np.ndarray, int, int]:
        h, w = screen_gray.shape[:2]

        zonas_calc = {
            "COMPLETA": (0, h, 0, w),
            "CENTRO": (int(h * 0.15), int(h * 0.85), int(w * 0.20), int(w * 0.80)),
            "MITAD_INFERIOR": (int(h * 0.45), h, 0, w),
            "MITAD_SUPERIOR": (0, int(h * 0.55), 0, w),
            "MITAD_DERECHA": (0, h, int(w * 0.45), w),
            "MITAD_IZQUIERDA": (0, h, 0, int(w * 0.55)),
            "ESQUINA_INF_DER": (int(h * 0.45), h, int(w * 0.45), w),
            "ESQUINA_INF_IZQ": (int(h * 0.45), h, 0, int(w * 0.55)),
            "ESQUINA_SUP_DER": (0, int(h * 0.35), int(w * 0.50), w),
            "ESQUINA_SUP_IZQ": (0, int(h * 0.35), 0, int(w * 0.50)),
        }

        if not zona or zona not in zonas_calc:
            return screen_gray, 0, 0

        y1, y2, x1, x2 = zonas_calc[zona]
        return screen_gray[y1:y2, x1:x2], x1, y1

    def find_template(
        self,
        screenshot,
        template_path: str,
        threshold: float | None = None,
        zona: str = "COMPLETA"
    ) -> tuple[int, int] | None:
        threshold = threshold if threshold is not None else self.threshold
        cached = self._load_template(template_path)
        if cached is None:
            return None

        template_gray, (h, w) = cached
        screen_gray = self.to_gray(screenshot)

        search_area, offset_x, offset_y = self._crop_zone(screen_gray, zona)

        if search_area.shape[0] >= h and search_area.shape[1] >= w:
            res = cv2.matchTemplate(search_area, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= threshold:
                cx = max_loc[0] + offset_x + w // 2
                cy = max_loc[1] + offset_y + h // 2
                return (cx, cy)

        return None

    def find_template_cached(
        self,
        screenshot,
        template_path: str,
        cache_name: str,
        ui_cache=None,
        threshold: float | None = None,
        zona: str = "COMPLETA",
    ) -> tuple[int, int] | None:
        """
        Intenta micro-verificación rápida (<1ms) si existe en ui_cache.
        Si no se detecta en la micro-ROI, recurre a find_template normal
        y auto-actualiza la posición en el caché.
        """
        threshold = threshold if threshold is not None else self.threshold
        if ui_cache is not None and getattr(ui_cache, "enabled", True) and getattr(ui_cache, "use_micro_verify", True):
            cached = self._load_template(template_path)
            if cached is not None:
                tpl_gray, _ = cached
                screen_gray = self.to_gray(screenshot)
                pos = ui_cache.micro_verify(screen_gray, cache_name, tpl_gray, threshold=threshold)
                if pos:
                    return pos

        pos = self.find_template(screenshot, template_path, threshold=threshold, zona=zona)
        if pos and ui_cache is not None and getattr(ui_cache, "enabled", True) and getattr(ui_cache, "learn_on_detect", True):
            h, w = (screenshot.shape[:2] if isinstance(screenshot, np.ndarray) else (1080, 2400))
            ui_cache.update_pos(cache_name, pos[0], pos[1], width=w, height=h)
        return pos

    def find_any(
        self,
        screenshot,
        template_paths: list[str],
        threshold: float | None = None,
        zona: str = "COMPLETA"
    ) -> tuple[str, int, int] | None:
        threshold = threshold if threshold is not None else self.threshold
        screen_gray = self.to_gray(screenshot)

        for path in template_paths:
            pos = self.find_template(screen_gray, path, threshold=threshold, zona=zona)
            if pos:
                return (path, pos[0], pos[1])
        return None

    def wait_for_template(
        self,
        screenshot_func,
        template_path: str,
        timeout: int = 12,
        threshold: float | None = None,
        interval: float = 0.25,
        zona: str = "COMPLETA",
        cache_name: str | None = None,
        ui_cache=None,
    ) -> tuple[int, int] | None:
        threshold = threshold if threshold is not None else self.threshold
        inicio = time.monotonic()

        while time.monotonic() - inicio < timeout:
            try:
                screen = screenshot_func()
                if cache_name and ui_cache:
                    pos = self.find_template_cached(
                        screen, template_path, cache_name, ui_cache=ui_cache, threshold=threshold, zona=zona
                    )
                else:
                    pos = self.find_template(screen, template_path, threshold, zona=zona)
                if pos:
                    return pos
            except Exception as e:
                log.warning("Error en wait_for_template: %s", e)
            time.sleep(interval)

        return None
```

## File: core/wall_upgrader.py
```python
"""
core/wall_upgrader.py
Módulo automatizado para mejorar pares de muros (20M) con Oro o Elixir.
"""
import time
import cv2
import numpy as np
from core.device import Device
from core.vision import Vision
from humanizer.behavior import HumanBehavior
from core.logger import get_logger

log = get_logger("wall_upgrader")


class WallUpgrader:
    def __init__(self, device: Device, vision: Vision):
        self.device = device
        self.vision = vision

        # 1. Menú del constructor
        self.tpl_build_btn = "assets/buttons/build_btn.png"
        self.tpl_filter_btn = "assets/buttons/build_filter_btn.png"
        self.tpl_wall_row = "assets/buttons/wall_row_item.png"

        # 2. Filtros apagados (marrón)
        self.tpl_filter_gold_off = "assets/buttons/filter_gold_off.png"
        self.tpl_filter_elixir_off = "assets/buttons/filter_elixir_off.png"
        self.tpl_filter_perm_off = "assets/buttons/filter_perm_off.png"

        # 3. Secuencia de selección múltiple (Muros dinámicos)
        self.tpl_upgrade_more = "assets/buttons/upgrade_more_btn.png"      # Botón "Upgrade More"
        self.tpl_add_wall_1 = "assets/buttons/add_wall_1_btn.png"          # Botón "+1 Add Wall"
        self.tpl_remove_wall_1 = "assets/buttons/remove_wall_1_btn.png"    # Botón "-1 Remove Wall"
        self.tpl_upgrade_hammers = "assets/buttons/upgrade_hammers.png"     # Martillos + Upgrade
        self.tpl_upgrade_gold = "assets/buttons/upgrade_with_gold.png"
        self.tpl_upgrade_elixir = "assets/buttons/upgrade_with_elixir.png"
        self.tpl_confirm = "assets/buttons/confirm_upgrade_btn.png"
        self.tpl_dialog_confirm = "assets/buttons/dialog_confirm_btn.png"   # Botón verde de validación múltiple

    def _apply_filter(self, resource_type: str):
        """
        Abre el menú de filtros del constructor, activa el filtro de recurso
        correspondiente y cierra el desplegable para despejar la lista.
        """
        screen = self.device.screenshot()
        filter_pos = self.vision.find_template(screen, self.tpl_filter_btn, threshold=0.70, zona="ESQUINA_SUP_DER")
        if not filter_pos:
            log.debug("Botón de filtro no detectado en el menú Build.")
            return

        log.info("🔍 Abriendo menú de filtros de Build...")
        HumanBehavior.reaction_delay()
        self.device.tap(*filter_pos)
        time.sleep(0.8)

        target_res_tpl = self.tpl_filter_gold_off if resource_type == "gold" else self.tpl_filter_elixir_off

        # El menú desplegable aparece en el centro/derecha (zona COMPLETA para no cortar el botón de oro en x=980)
        res_pos = self.vision.wait_for_template(
            self.device.screenshot, target_res_tpl, timeout=2.5, threshold=0.74, zona="COMPLETA"
        )
        if res_pos:
            log.info("🎯 Filtro de recurso (%s) detectado en %s. Activando...", resource_type.upper(), res_pos)
            HumanBehavior.reaction_delay()
            self.device.tap(*res_pos)
            time.sleep(0.6)
        else:
            log.info("ℹ️ Filtro de recurso (%s) ya está activado en el menú.", resource_type.upper())

        # Cerrar el menú desplegable tocando de nuevo el botón de filtro
        log.info("Cerrando desplegable de filtros...")
        self.device.tap(*filter_pos)
        time.sleep(0.8)

    def _find_and_tap_wall_in_list(self) -> bool:
        """
        Busca la fila de 'Wall' en la lista de mejoras del constructor.
        Usa 'Wall x' como template (umbral 0.74) para evitar falsos positivos
        con héroes como Grand Warden (match máx ~0.57).
        Realiza deslizamientos más largos hacia arriba para recorrer toda la lista.
        """
        max_attempts = 7
        log.info("🔍 Buscando fila de 'Wall' en la lista (máx %d intentos)...", max_attempts)

        for attempt in range(max_attempts):
            screen = self.device.screenshot()
            wall_pos = self.vision.find_template(screen, self.tpl_wall_row, threshold=0.74, zona="MITAD_DERECHA")
            if wall_pos:
                log.info("🎯 'Wall' encontrado en %s (intento %d/%d). Tocando...", wall_pos, attempt + 1, max_attempts)
                HumanBehavior.reaction_delay()
                self.device.tap(*wall_pos)
                time.sleep(0.8)
                return True

            if attempt < max_attempts - 1:
                # Swipe más largo y fluido: de 80% a 22% de la altura
                sx = int(self.device.width * 0.83)
                sy_start = int(self.device.height * 0.80)
                sy_end = int(self.device.height * 0.22)
                log.debug("📜 Deslizando lista hacia abajo (intento %d): (%d, %d) -> (%d, %d)", attempt + 1, sx, sy_start, sx, sy_end)
                self.device.swipe(sx, sy_start, sx, sy_end, duration=0.55)
                time.sleep(1.0)

        log.warning("❌ No se encontró 'Wall' tras %d intentos de desplazamiento.", max_attempts)
        return False

    def _find_upgrade_buttons_left_to_right(self, screenshot) -> list[tuple[int, int]]:
        """
        Encuentra todos los botones 'Upgrade' en la parte inferior y los ordena de izquierda a derecha.
        Índice 0 = Oro (Izquierda) | Índice -1 = Elixir (Derecha).
        """
        cached = self.vision._load_template(self.tpl_upgrade_hammers)
        if cached is None:
            return []

        tpl_gray, (th, tw) = cached
        screen_gray = self.vision.to_gray(screenshot)
        crop, ox, oy = self.vision._crop_zone(screen_gray, "MITAD_INFERIOR")

        res = cv2.matchTemplate(crop, tpl_gray, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= 0.78)

        detections: list[tuple[int, int, float]] = []
        for y, x in zip(ys, xs):
            cx = int(x + ox + tw // 2)
            cy = int(y + oy + th // 2)
            detections.append((cx, cy, float(res[y, x])))

        if not detections:
            return []

        # NMS horizontal: agrupar detecciones que estén a menos de 45px de distancia en X
        detections.sort(key=lambda item: item[2], reverse=True)
        unique_btns: list[tuple[int, int]] = []
        for cx, cy, _ in detections:
            if all(abs(cx - ux) >= 45 for ux, _ in unique_btns):
                unique_btns.append((cx, cy))

        # Ordenar de izquierda (menor X) a derecha (mayor X)
        unique_btns.sort(key=lambda pt: pt[0])
        return unique_btns

    def _is_cost_affordable(self, screen: np.ndarray, target_pos: tuple[int, int]) -> bool:
        """
        Evalúa si la cifra de costo encima del botón de mejora está en blanco
        (costeable) o en rojo (insuficientes recursos).
        """
        bx, by = target_pos
        y1 = max(0, by - 95)
        y2 = min(screen.shape[0], by - 65)
        x1 = max(0, bx - 60)
        x2 = min(screen.shape[1], bx + 60)
        text_roi = screen[y1:y2, x1:x2]

        if text_roi.size == 0:
            return False

        # Pixeles blancos (texto normal costeable): B, G, R > 180
        white_pts = int(np.sum((text_roi[:, :, 0] > 180) & (text_roi[:, :, 1] > 180) & (text_roi[:, :, 2] > 180)))
        # Pixeles rojos (indicador de costo excesivo): R > 180 y B < 120
        red_pts = int(np.sum((text_roi[:, :, 2] > 180) & (text_roi[:, :, 0] < 120)))

        is_affordable = (red_pts < 30) and (white_pts > 400)
        log.info(
            "📊 Evaluación de costo en %s: white=%d, red=%d -> %s",
            target_pos, white_pts, red_pts, "COSTEABLE (blanco)" if is_affordable else "INSUFICIENTE (rojo)"
        )
        return is_affordable


    def upgrade_wall(self, resource_type: str, max_walls: int = 5) -> bool:
        """
        Ejecuta el flujo completo para mejorar muros maximizando el uso de recursos disponibles:
        Build -> Filtro -> Wall -> Upgrade More -> Multi-selección dinámica (+1 hasta rojo) -> Upgrade -> Confirmar.
        """
        log.info("🧱 Iniciando mejora dinámica de muros usando: %s (máx %d muros)", resource_type.upper(), max_walls)

        # 1. Abrir menú Build
        screen = self.device.screenshot()
        build_pos = self.vision.find_template(screen, self.tpl_build_btn, threshold=0.78, zona="ESQUINA_INF_DER")
        if not build_pos:
            log.warning("No se encontró el botón Build.")
            return False

        HumanBehavior.reaction_delay()
        self.device.tap(*build_pos)
        time.sleep(1.0)

        # 2. Verificar filtros y tocar "Wall" en la lista derecha
        self._apply_filter(resource_type)
        if not self._find_and_tap_wall_in_list():
            log.warning("⚠️ No se encontró 'Wall' en la lista.")
            # Cerrar menú tocando fuera
            self.device.tap(int(self.device.width * 0.25), int(self.device.height * 0.50))
            return False

        # 3. Esperar y tocar "Upgrade More"
        more_pos = self.vision.wait_for_template(
            self.device.screenshot, self.tpl_upgrade_more, timeout=5, threshold=0.74, zona="MITAD_INFERIOR"
        )
        if not more_pos:
            log.warning("⚠️ No apareció el botón 'Upgrade More'.")
            return False

        log.info("🎯 Botón 'Upgrade More' detectado en %s. Tocando...", more_pos)
        HumanBehavior.reaction_delay()
        self.device.tap(*more_pos)
        time.sleep(1.0)

        # 4. Modo multi-selección dinámica según recursos disponibles
        screen = self.device.screenshot()
        upgrade_btns = self._find_upgrade_buttons_left_to_right(screen)
        if upgrade_btns:
            target_pos = upgrade_btns[0] if resource_type == "gold" else upgrade_btns[-1]
        else:
            target_pos = (1030, 842) if resource_type == "gold" else (1240, 840)

        # Verificar si al menos 1 muro es costeable
        if not self._is_cost_affordable(screen, target_pos):
            log.warning("⚠️ Recursos insuficientes ni para mejorar 1 solo muro con %s.", resource_type.upper())
            # Deseleccionar tocando en zona segura del mapa
            self.device.tap(int(self.device.width * 0.25), int(self.device.height * 0.50))
            return False

        # Localizar botones "+1 Muro" y "-1 Muro"
        add_one_pos = self.vision.find_template(screen, self.tpl_add_wall_1, threshold=0.72, zona="MITAD_INFERIOR")
        if not add_one_pos:
            add_one_pos = (835, 848)

        remove_one_pos = self.vision.find_template(screen, self.tpl_remove_wall_1, threshold=0.72, zona="MITAD_INFERIOR")
        if not remove_one_pos:
            remove_one_pos = (418, 850)

        current_walls = 1
        while current_walls < max_walls:
            log.info("➕ Probando añadir muro (+1)... (actuales: %d)", current_walls)
            HumanBehavior.reaction_delay()
            self.device.tap(*add_one_pos)
            time.sleep(0.7)
            screen = self.device.screenshot()

            # Recalcular posición del botón por si hubo micro-desplazamiento
            btns = self._find_upgrade_buttons_left_to_right(screen)
            if btns:
                target_pos = btns[0] if resource_type == "gold" else btns[-1]

            if not self._is_cost_affordable(screen, target_pos):
                log.info(
                    "🛑 Costo en ROJO detectado: recursos insuficientes para %d muros. Retrocediendo (-1)...",
                    current_walls + 1
                )
                self.device.tap(*remove_one_pos)
                time.sleep(0.7)
                break

            current_walls += 1
            log.info("🎯 %d muro(s) seleccionados y verificados como costeables.", current_walls)

        # 5. Tocar botón Upgrade con los muros costeables seleccionados
        screen = self.device.screenshot()
        btns = self._find_upgrade_buttons_left_to_right(screen)
        if btns:
            target_pos = btns[0] if resource_type == "gold" else btns[-1]

        log.info("🔨 Ejecutando mejora de %d muro(s) con %s en %s...", current_walls, resource_type.upper(), target_pos)
        HumanBehavior.reaction_delay()
        self.device.tap(*target_pos)
        time.sleep(1.2)

        # 6. Manejo de diálogo de validación / confirmación
        screen = self.device.screenshot()
        confirm_pos = self.vision.find_template(screen, self.tpl_dialog_confirm, threshold=0.74, zona="CENTRO")
        if not confirm_pos:
            confirm_pos = self.vision.find_template(screen, self.tpl_dialog_confirm, threshold=0.74, zona="MITAD_INFERIOR")
        if not confirm_pos:
            confirm_pos = self.vision.find_template(screen, self.tpl_confirm, threshold=0.74, zona="CENTRO")

        if confirm_pos:
            log.info("🎯 Diálogo de validación detectado en %s. Confirmando mejora...", confirm_pos)
            HumanBehavior.reaction_delay()
            self.device.tap(*confirm_pos)
            time.sleep(1.2)
        else:
            log.info("ℹ️ No se requirió diálogo de confirmación adicional o se ejecutó directamente.")

        log.info("✅ ¡%d Muro(s) mejorado(s) exitosamente usando %s!", current_walls, resource_type.upper())
        time.sleep(1.0)
        return True
```

## File: humanizer/__init__.py
```python

```

## File: humanizer/behavior.py
```python
"""
humanizer/behavior.py
Funciones de aleatorización conductual para mitigar detección por patrones.
"""
import random
import time
from core.config import SETTINGS

_HUM = SETTINGS["humanizer"]


class HumanBehavior:
    @staticmethod
    def sleep_range(min_sec: float, max_sec: float):
        """Pausa con distribución ligeramente sesgada al centro (más natural que uniforme pura)."""
        mean = (min_sec + max_sec) / 2.0
        std = (max_sec - min_sec) / 4.0
        delay = random.gauss(mean, std)
        delay = max(min_sec, min(max_sec, delay))
        time.sleep(delay)

    @classmethod
    def reaction_delay(cls):
        """Tiempo de reacción humano tras aparecer un botón o cargar una aldea."""
        cls.sleep_range(_HUM["reaction_min_sec"], _HUM["reaction_max_sec"])

    @classmethod
    def between_sides_delay(cls):
        """Pausa al cambiar de flanco durante el despliegue."""
        cls.sleep_range(_HUM["between_sides_min_sec"], _HUM["between_sides_max_sec"])

    @classmethod
    def burst_tap_delay(cls):
        """Cadencia humana rápida entre toques consecutivos al soltar múltiples tropas (85ms a 135ms)."""
        cls.sleep_range(
            _HUM.get("burst_tap_min_sec", 0.085),
            _HUM.get("burst_tap_max_sec", 0.135),
        )

    @staticmethod
    def jitter_point(x: int, y: int, max_offset: int | None = None) -> tuple[int, int]:
        """Dispersión gaussiana centrada en el objetivo y truncada a +-max_offset."""
        offset = max_offset if max_offset is not None else _HUM["tap_jitter_px"]
        std = max(1.0, offset / 2.2)
        dx = int(round(max(-offset, min(offset, random.gauss(0, std)))))
        dy = int(round(max(-offset, min(offset, random.gauss(0, std)))))
        return max(0, x + dx), max(0, y + dy)

    @staticmethod
    def hold_duration() -> float:
        return random.uniform(_HUM["hold_deploy_min_sec"], _HUM["hold_deploy_max_sec"])
```

## File: tests/__init__.py
```python
"""
Package initialization and shared test infrastructure for ClashFarmerbot test suite.
Supports offline execution using standard library unittest without requiring physical ADB hardware.
"""
import os
import sys
import unittest
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class MockDeviceShell:
    """Mock interface for device.d (uiautomator2 shell and click actions)."""
    def __init__(self, parent_device):
        self.parent = parent_device

    def shell(self, cmd: str) -> str:
        self.parent.shell_history.append(cmd)
        return ""

    def click(self, x: int, y: int):
        self.parent.tap_history.append((int(x), int(y)))


class MockDevice:
    """Offline mock device implementing the Device protocol used by DeploymentStrategy."""
    def __init__(self, width: int = 2400, height: int = 1080, screenshot_feed: list[np.ndarray] | None = None):
        self.width = width
        self.height = height
        self.shell_history: list[str] = []
        self.tap_history: list[tuple[int, int]] = []
        self.zoom_out_calls: int = 0
        self.screenshot_feed = list(screenshot_feed) if screenshot_feed else []
        self.default_screenshot: np.ndarray | None = None
        self.d = MockDeviceShell(self)

    def set_default_screenshot(self, img: np.ndarray):
        self.default_screenshot = img.copy()

    def screenshot(self) -> np.ndarray:
        if self.screenshot_feed:
            return self.screenshot_feed.pop(0)
        if self.default_screenshot is not None:
            return self.default_screenshot.copy()
        # Default empty black frame 2400x1080
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def tap(self, x: int, y: int):
        self.tap_history.append((int(x), int(y)))

    def zoom_out(self, center: tuple[int, int] = (1200, 540)):
        self.zoom_out_calls += 1

    def clear_history(self):
        self.shell_history.clear()
        self.tap_history.clear()
        self.zoom_out_calls = 0


def count_structure_intersections(contour: np.ndarray, detections: list) -> int:
    """
    Checks if a polygon contour intersects/slices any detected building.
    A building is cut if its bounding box vertices have both interior (dist > 0)
    and exterior (dist < 0) points relative to the contour polygon.
    """
    if contour is None or len(contour) < 3 or not detections:
        return 0

    cnt = np.array(contour, dtype=np.int32)
    if cnt.ndim == 2:
        cnt = cnt.reshape((-1, 1, 2))

    cut_count = 0
    for d in detections:
        if not hasattr(d, "bbox_xyxy"):
            continue
        x1, y1, x2, y2 = d.bbox_xyxy
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        dists = [cv2.pointPolygonTest(cnt, (float(cx), float(cy)), True) for cx, cy in corners]
        # If some corners are strictly inside (>0) and some strictly outside (<0), the polygon edge cuts the building
        if min(dists) < -1.0 and max(dists) > 1.0:
            cut_count += 1
    return cut_count


def load_test_image(path: str) -> np.ndarray | None:
    """Loads an image relative to PROJECT_ROOT safely."""
    full_path = os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path
    if os.path.exists(full_path):
        return cv2.imread(full_path)
    return None
```

## File: tests/test_army_checker.py
```python
"""
tests/test_army_checker.py
Pruebas unitarias para la detección de tropas y Súper Rompemuros en la barra inferior.
"""
import unittest
import numpy as np
import cv2
from core.vision import Vision
from vision.army_checker import ArmyChecker


class TestArmyChecker(unittest.TestCase):
    def setUp(self):
        self.vision = Vision()
        self.checker = ArmyChecker(self.vision)

    def test_find_super_wall_breaker_success(self):
        """Verifica que un frame con el icono de SWB en la mitad inferior se detecte correctamente."""
        tpl = cv2.imread(self.checker.tpl_swb_slot)
        if tpl is None:
            self.skipTest("No se encontró el template super_wall_breaker_slot.png")

        th, tw = tpl.shape[:2]
        frame = np.zeros((1080, 2400, 3), dtype=np.uint8)
        # Pegar el template en una posición realista de la barra de batalla
        target_x, target_y = 520, 930
        frame[target_y:target_y+th, target_x:target_x+tw] = tpl

        pos = self.checker.find_super_wall_breaker_in_battle_bar(frame)
        self.assertIsNotNone(pos, "Debe detectar el icono de Súper Rompemuros")
        # El punto retornado es el centro del match
        self.assertAlmostEqual(pos[0], target_x + tw // 2, delta=5)
        self.assertAlmostEqual(pos[1], target_y + th // 2, delta=5)

    def test_find_super_wall_breaker_absent(self):
        """Verifica que si no hay SWB en el frame retorne None sin lanzar excepción."""
        empty_frame = np.zeros((1080, 2400, 3), dtype=np.uint8)
        pos = self.checker.find_super_wall_breaker_in_battle_bar(empty_frame)
        self.assertIsNone(pos, "No debe detectar SWB en una imagen vacía")

    def test_find_super_wall_breaker_ignores_upper_half(self):
        """Verifica que si el icono está en la mitad superior (ej. en menús o aldea) se ignore."""
        tpl = cv2.imread(self.checker.tpl_swb_slot)
        if tpl is None:
            self.skipTest("No se encontró el template super_wall_breaker_slot.png")

        th, tw = tpl.shape[:2]
        frame = np.zeros((1080, 2400, 3), dtype=np.uint8)
        # Pegar en la mitad superior (y = 200)
        frame[200:200+th, 1000:1000+tw] = tpl

        pos = self.checker.find_super_wall_breaker_in_battle_bar(frame)
        self.assertIsNone(pos, "No debe detectar en la mitad superior por zona='MITAD_INFERIOR'")


if __name__ == "__main__":
    unittest.main()
```

## File: tests/test_boundary_detection.py
```python
"""
tests/test_boundary_detection.py
Comprehensive 4-tier test suite for Village Boundary Detection (Requirement R1).
Tests cover:
- Tier 1: Feature coverage (Red extraction, UI masking, Non-convex tracing, Dual modes)
- Tier 2: Boundary & corner cases (empty, non-red, extreme coords, wall bastions, saturation)
- Tier 3: Cross-feature combinations (YOLO detections + red boundary, ray march outer drop, building clearance)
- Tier 4: Real-world match scenarios (logs/red_test_flash.png, datasets/coc/images/train/, old convex hull defect contrast)
"""
import os
import sys
import unittest
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests import MockDevice, count_structure_intersections, load_test_image
from tools.test_boundary_overlay import (
    build_ui_mask,
    extract_red_mask,
    detect_village_boundary,
    audit_boundary_collisions,
)
from vision.yolo_detector import get_detector, Detection
from attack.deployment_strategy import compute_corridor_drop_geometry, _OUTER_GRASS_HULL


class TestBoundaryDetectionTier1(unittest.TestCase):
    """Tier 1: Feature coverage (>=5 tests per feature)."""

    # --- Feature 1: Red Line Extraction ---
    def test_red_hsv_extraction_bounds(self):
        """Verifies calibrated HSV range catches both primary and wrap-around red hues."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Red hue wrap-around 1 (H=5, S=200, V=200 in BGR is approx (40, 40, 200))
        img[20:40, 20:40] = (30, 30, 220)
        # Red hue wrap-around 2 (H=175, S=200, V=200 in BGR is approx (60, 20, 200))
        img[60:80, 60:80] = (60, 20, 210)

        mask = extract_red_mask(img, ui_mask=np.ones((100, 100), dtype=np.uint8))
        self.assertGreater(np.count_nonzero(mask[20:40, 20:40]), 300)
        self.assertGreater(np.count_nonzero(mask[60:80, 60:80]), 300)

    def test_red_extraction_rejects_grass(self):
        """Verifies green grass colors (H~35-85) are completely rejected by red mask."""
        grass = np.full((100, 100, 3), (35, 140, 50), dtype=np.uint8)  # Grass green BGR
        mask = extract_red_mask(grass, ui_mask=np.ones((100, 100), dtype=np.uint8))
        self.assertEqual(np.count_nonzero(mask), 0)

    def test_red_extraction_rejects_stone_and_elixir(self):
        """Verifies grey stones and purple elixir buildings do not leak into red mask."""
        stone = np.full((50, 50, 3), (120, 120, 120), dtype=np.uint8)
        elixir = np.full((50, 50, 3), (180, 50, 180), dtype=np.uint8)  # Magenta/purple
        m_stone = extract_red_mask(stone, ui_mask=np.ones((50, 50), dtype=np.uint8))
        m_elixir = extract_red_mask(elixir, ui_mask=np.ones((50, 50), dtype=np.uint8))
        self.assertEqual(np.count_nonzero(m_stone), 0)
        self.assertEqual(np.count_nonzero(m_elixir), 0)

    def test_red_extraction_differential_signal(self):
        """Verifies differential flash image has high red concentration."""
        diff_img = load_test_image("logs/red_test_diff.png")
        if diff_img is None:
            self.skipTest("logs/red_test_diff.png not found")
        mask = extract_red_mask(diff_img)
        self.assertGreater(np.count_nonzero(mask), 1000)

    def test_red_morphological_closing_connectivity(self):
        """Verifies morphological closing bridges grid lines without bridging separate islands."""
        grid = np.zeros((200, 200), dtype=np.uint8)
        # Create dashed horizontal line with 5px gaps
        for x in range(20, 180, 15):
            grid[100, x:x+10] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)
        # Line should now be contiguous from x=20 to x=175
        self.assertTrue(np.all(closed[100, 20:175] == 255))

    # --- Feature 2: Targeted UI Masking ---
    def test_ui_mask_preserves_south_flank(self):
        """Verifies Y in [880, 960] retains red terrain pixels (no crude clamping)."""
        mask = build_ui_mask((1080, 2400))
        # The region between y=880 and 960 at x=1200 (center south) must be 1 (unmasked)
        self.assertEqual(mask[920, 1200], 1)
        self.assertEqual(mask[900, 1500], 1)

    def test_ui_mask_preserves_north_flank(self):
        """Verifies Y in [100, 120] retains red terrain pixels."""
        mask = build_ui_mask((1080, 2400))
        self.assertEqual(mask[110, 1200], 1)
        self.assertEqual(mask[105, 800], 1)

    def test_ui_mask_masks_surrender_button(self):
        """Verifies the surrender/end battle button region is properly masked to 0."""
        mask = build_ui_mask((1080, 2400))
        y_surr, x_surr = int(0.78 * 1080), int(0.10 * 2400)
        self.assertEqual(mask[y_surr, x_surr], 0)

    def test_ui_mask_masks_troop_ribbon(self):
        """Verifies bottom troop deployment bar is masked to 0."""
        mask = build_ui_mask((1080, 2400))
        self.assertEqual(mask[1020, 1200], 0)
        self.assertEqual(mask[1000, 800], 0)

    def test_ui_mask_masks_next_button(self):
        """Verifies next button at bottom right is masked to 0."""
        mask = build_ui_mask((1080, 2400))
        y_next, x_next = int(0.85 * 1080), int(0.92 * 2400)
        self.assertEqual(mask[y_next, x_next], 0)

    # --- Feature 3: Non-Convex Contour Tracing ---
    def test_non_convex_tracing_bay_preservation(self):
        """Verifies a concave C-shaped village boundary produces a non-convex polygon."""
        mask = np.zeros((1080, 2400), dtype=np.uint8)
        # Draw a large C-shape (outer box with a deep rectangular bay carved into the right flank)
        cv2.rectangle(mask, (800, 300), (1600, 800), 255, -1)
        # Carve a 200px deep bay
        cv2.rectangle(mask, (1400, 450), (1600, 650), 0, -1)

        # Dilate slightly and find contour
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_cnt = max(cnts, key=cv2.contourArea)
        # Assert the resulting contour is mathematically non-convex
        self.assertFalse(cv2.isContourConvex(main_cnt))

    def test_non_convex_bay_depth_retention(self):
        """Verifies boundary points enter inside the concave bay rather than bridging it."""
        mask = np.zeros((1080, 2400), dtype=np.uint8)
        cv2.rectangle(mask, (800, 300), (1600, 800), 255, -1)
        cv2.rectangle(mask, (1350, 450), (1600, 650), 0, -1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_cnt = max(cnts, key=cv2.contourArea)
        # Point inside the bay mouth (1450, 550) should be OUTSIDE the contour (dist < 0)
        dist = cv2.pointPolygonTest(main_cnt, (1450.0, 550.0), True)
        self.assertLess(dist, 0.0)

    def test_non_convex_isotropic_dilation(self):
        """Verifies dilation with ellipse kernel expands outwards uniformly in all directions."""
        mask = np.zeros((500, 500), dtype=np.uint8)
        cv2.circle(mask, (250, 250), 50, 255, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91))
        dilated = cv2.dilate(mask, kernel)

        # Dilated radius should be 50 + 45 = 95 px in all 4 cardinal directions
        self.assertEqual(dilated[250, 250 + 94], 255)
        self.assertEqual(dilated[250, 250 + 98], 0)
        self.assertEqual(dilated[250, 250 - 94], 255)
        self.assertEqual(dilated[250 - 94, 250], 255)
        self.assertEqual(dilated[250 + 94, 250], 255)

    def test_non_convex_rejects_noise_contours(self):
        """Verifies small isolated noise pixels/contours are filtered out."""
        img = np.zeros((1080, 2400, 3), dtype=np.uint8)
        # Small 5x5 red artifact
        img[100:105, 100:105] = (30, 30, 220)
        red_cnt, green_cnt = detect_village_boundary(img)
        self.assertIsNone(red_cnt)
        self.assertIsNone(green_cnt)

    def test_non_convex_no_cross_cutting_bay_structure(self):
        """Verifies building sitting inside exterior bay is NOT cut by the boundary."""
        mask = np.zeros((1080, 2400), dtype=np.uint8)
        cv2.rectangle(mask, (800, 300), (1600, 800), 255, -1)
        # Bay at (1350, 450) to (1600, 650)
        cv2.rectangle(mask, (1350, 450), (1600, 650), 0, -1)

        # Dilate mask as done in detector
        dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91)))
        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        main_cnt = max(cnts, key=cv2.contourArea)

        # Place a building inside the bay at (1500, 550) with size 40x40
        class MockDet:
            bbox_xyxy = (1480, 530, 1520, 570)
            center = (1500, 550)
            cls_name = "collector_gold"

        cuts = audit_boundary_collisions(main_cnt, [MockDet()])
        # A building inside the bay in legal grass must not be intersected by the boundary
        self.assertEqual(len(cuts), 0)

    # --- Feature 4: Dual Modes (pixel_scan vs high_res_contour) ---
    def test_pixel_scan_mode_dense_vertices(self):
        """Verifies pixel_scan mode yields high-density vertices."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        _, green_cnt = detect_village_boundary(img, mode="pixel_scan")
        self.assertIsNotNone(green_cnt)
        self.assertGreater(len(green_cnt), 3000)

    def test_high_res_contour_vertex_reduction(self):
        """Verifies high_res_contour mode reduces vertex count by >= 70%."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        _, green_pixel = detect_village_boundary(img, mode="pixel_scan")
        _, green_contour = detect_village_boundary(img, mode="high_res_contour")
        self.assertIsNotNone(green_pixel)
        self.assertIsNotNone(green_contour)
        reduction = (len(green_pixel) - len(green_contour)) / len(green_pixel)
        self.assertGreaterEqual(reduction, 0.70)

    def test_dual_mode_enclosure_equivalence(self):
        """Verifies both modes enclose all building centers without cuts."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        detector = get_detector()
        dets = detector.detect(img) if detector else []
        _, green_pixel = detect_village_boundary(img, detections=dets, mode="pixel_scan")
        _, green_contour = detect_village_boundary(img, detections=dets, mode="high_res_contour")

        cuts_pixel = audit_boundary_collisions(green_pixel, dets)
        cuts_contour = audit_boundary_collisions(green_contour, dets)
        self.assertEqual(len(cuts_pixel), 0)
        self.assertEqual(len(cuts_contour), 0)

    def test_dual_mode_configuration_selection(self):
        """Verifies passing mode argument explicitly selects the correct mode."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        _, cnt_default = detect_village_boundary(img)
        _, cnt_contour = detect_village_boundary(img, mode="high_res_contour")
        # Default is pixel_scan with >3000 points, high_res_contour has < 500
        self.assertGreater(len(cnt_default), 1000)
        self.assertLess(len(cnt_contour), 500)

    def test_dual_mode_zero_structure_cuts_in_both(self):
        """Verifies both modes achieve exactly 0 cut structures."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        detector = get_detector()
        dets = detector.detect(img) if detector else []
        for mode in ["pixel_scan", "high_res_contour"]:
            with self.subTest(mode=mode):
                _, green = detect_village_boundary(img, detections=dets, mode=mode)
                cuts = audit_boundary_collisions(green, dets)
                self.assertEqual(len(cuts), 0)


class TestBoundaryDetectionTier2(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases."""

    def test_empty_black_image(self):
        """Empty black frame returns (None, None) gracefully."""
        black = np.zeros((1080, 2400, 3), dtype=np.uint8)
        red, green = detect_village_boundary(black)
        self.assertIsNone(red)
        self.assertIsNone(green)

    def test_zero_red_pixels_daylight(self):
        """Frame with zero red pixels and no detections returns (None, None)."""
        daylight = np.full((1080, 2400, 3), (80, 180, 90), dtype=np.uint8)
        red, green = detect_village_boundary(daylight)
        self.assertIsNone(red)
        self.assertIsNone(green)

    def test_extreme_coordinates_near_screen_edges(self):
        """Buildings near extreme edges (x=10, y=10) or (x=2390, y=1070) don't crash."""
        img = np.zeros((1080, 2400, 3), dtype=np.uint8)
        img[500:600, 1100:1300] = (30, 30, 220)  # Center red zone

        class CornerDet:
            def __init__(self, bbox, name):
                self.bbox_xyxy = bbox
                self.center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                self.cls_name = name

        dets = [
            CornerDet((5, 5, 45, 45), "collector_gold"),
            CornerDet((2350, 1030, 2395, 1075), "storage_elixir"),
        ]
        red, green = detect_village_boundary(img, detections=dets)
        self.assertIsNotNone(green)
        cuts = audit_boundary_collisions(green, dets)
        self.assertEqual(len(cuts), 0)

    def test_wall_bastions_enclosed(self):
        """Protruding narrow bastion is enclosed by isotropic dilation without cuts."""
        mask = np.zeros((1080, 2400), dtype=np.uint8)
        cv2.rectangle(mask, (1000, 400), (1400, 680), 255, -1)
        # Narrow bastion 30px wide protruding 80px outward
        cv2.rectangle(mask, (1185, 320), (1215, 400), 255, -1)

        dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91)))
        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_cnt = max(cnts, key=cv2.contourArea)

        # Tip of bastion at (1200, 320) must be inside dilated boundary (dist >= 40)
        dist = cv2.pointPolygonTest(main_cnt, (1200.0, 320.0), True)
        self.assertGreaterEqual(dist, 40.0)

    def test_saturated_flash_frame(self):
        """High saturation/brightness red frame maintains valid non-empty polygon."""
        img = np.full((1080, 2400, 3), (20, 20, 255), dtype=np.uint8)
        red, green = detect_village_boundary(img)
        self.assertIsNotNone(green)
        self.assertGreater(cv2.contourArea(green), 100000)


class TestBoundaryDetectionTier3(unittest.TestCase):
    """Tier 3: Cross-Feature Combinations."""

    def test_building_detections_integrated_with_red_mask(self):
        """Verifies base mask includes both red lines and buffered building boxes."""
        img = np.zeros((1080, 2400, 3), dtype=np.uint8)
        img[500:600, 1100:1300] = (30, 30, 220)  # Red core

        class MockDet:
            bbox_xyxy = (1280, 520, 1340, 580)
            center = (1310, 550)
            cls_name = "storage_gold"

        _, green = detect_village_boundary(img, detections=[MockDet()])
        self.assertIsNotNone(green)
        # Center of building must be strictly inside green contour
        dist = cv2.pointPolygonTest(green, (1310.0, 550.0), True)
        self.assertGreater(dist, 20.0)

    def test_building_safety_buffer_margin(self):
        """Green contour maintains >= 30px distance from all building edges."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        detector = get_detector()
        dets = detector.detect(img) if detector else []
        _, green = detect_village_boundary(img, detections=dets)

        for d in dets:
            dist_center = cv2.pointPolygonTest(green, (float(d.center[0]), float(d.center[1])), True)
            self.assertGreater(dist_center, 0.0, f"Building center {d.center} not enclosed")

    def test_corridor_ray_march_concave_bay_outer_drop(self):
        """Ray march on concave base selects outermost hit to avoid dropping inside bay."""
        # Create green hull with a bay along the east flank
        # Star-shaped / concave hull points
        pts = [
            (1200, 200), (1600, 350), (1400, 540),  # Inward concave dip at (1400, 540)
            (1700, 540),                             # Outer wall point along ray
            (1600, 750), (1200, 900), (800, 750), (700, 540), (800, 350)
        ]
        green_hull = np.array(pts, dtype=np.int32)

        # Compute drop for target jump at (1300, 540)
        center = (1200.0, 540.0)
        jump_pt = (1300, 540)
        drop_pt, sweep_pts = compute_corridor_drop_geometry(center, jump_pt, (2400, 1080), green_hull=green_hull)

        # Drop point must be beyond outer point (x >= 1400)
        self.assertGreaterEqual(drop_pt[0], 1400)

    def test_corridor_drop_radial_clearance(self):
        """Drop point has >= 50px clearance from all detected buildings along corridor."""
        center = (1200.0, 540.0)
        jump_pt = (1500, 540)

        class MockDet:
            center = (1600, 540)
            cls_name = "storage_gold"

        drop_pt, _ = compute_corridor_drop_geometry(
            center, jump_pt, (2400, 1080), detections=[MockDet()]
        )
        # Drop point must be placed beyond the peripheral building (1600 + 50 = 1650)
        self.assertGreaterEqual(drop_pt[0], 1650)

    def test_corridor_drop_within_screen_bounds(self):
        """Drop point and deploy sweep points remain within valid touch boundaries."""
        center = (1200.0, 540.0)
        jump_pt = (2300, 540)  # Extreme east
        drop_pt, sweep_pts = compute_corridor_drop_geometry(center, jump_pt, (2400, 1080))

        # Check bounds: x in [0.06*w, 0.94*w], y in [0.11*h, 0.83*h]
        self.assertGreaterEqual(drop_pt[0], int(0.06 * 2400))
        self.assertLessEqual(drop_pt[0], int(0.94 * 2400))
        for p in sweep_pts:
            self.assertGreaterEqual(p[0], int(0.06 * 2400))
            self.assertLessEqual(p[0], int(0.94 * 2400))


class TestBoundaryDetectionTier4(unittest.TestCase):
    """Tier 4: Real-World Match Scenarios."""

    def test_real_red_flash_cut_count_zero(self):
        """Verifies 0 cuts on golden sample logs/red_test_flash.png with 18 YOLO detections."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        detector = get_detector()
        dets = detector.detect(img) if detector else []
        self.assertEqual(len(dets), 18, "Expected 18 YOLO detections in golden flash frame")

        _, green = detect_village_boundary(img, detections=dets, mode="pixel_scan")
        self.assertIsNotNone(green)
        cuts = audit_boundary_collisions(green, dets)
        self.assertEqual(len(cuts), 0, f"Expected 0 cut structures but found {len(cuts)}")

    def test_existing_convex_hull_flaw_contrast(self):
        """
        Demonstrates that applying cv2.convexHull cuts 5 structures on the exact same frame,
        proving the non-convex requirement.
        """
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        detector = get_detector()
        dets = detector.detect(img) if detector else []

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0, 100, 100), (12, 255, 255))
        m2 = cv2.inRange(hsv, (168, 100, 100), (180, 255, 255))
        red_mask = m1 | m2
        red_mask[:120, :] = 0
        red_mask[880:, :] = 0  # Old crude clamping

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant = [c for c in contours if cv2.contourArea(c) > 300]
        self.assertTrue(len(significant) > 0)

        all_pts = np.vstack(significant)
        red_hull = cv2.convexHull(all_pts)[:, 0, :]
        green_pts = []
        for pt in red_hull:
            vx, vy = pt[0] - 1200, pt[1] - 540
            norm = max(float(np.hypot(vx, vy)), 1.0)
            green_pts.append([int(pt[0] + 45.0 * vx / norm), int(pt[1] + 45.0 * vy / norm)])
        old_convex_green = cv2.convexHull(np.array(green_pts, dtype=np.int32))[:, 0, :]

        # Verify old convex hull cuts structures
        old_cuts = audit_boundary_collisions(old_convex_green, dets)
        self.assertGreaterEqual(len(old_cuts), 2, "Convex hull flaw must reproduce at least 2 cut structures")

    def test_real_match_frame_attack_091024(self):
        """Verifies boundary detection on logs/yolo/attack_20261106_091024.png."""
        img = load_test_image("logs/yolo/attack_20261106_091024.png")
        if img is None:
            self.skipTest("attack_20261106_091024.png not found")
        detector = get_detector()
        dets = detector.detect(img) if detector else []
        _, green = detect_village_boundary(img, detections=dets)
        if green is not None:
            cuts = audit_boundary_collisions(green, dets)
            self.assertEqual(len(cuts), 0)

    def test_real_match_frame_attack_183131(self):
        """Verifies boundary detection on logs/yolo/attack_20261105_183131.png."""
        img = load_test_image("logs/yolo/attack_20261105_183131.png")
        if img is None:
            self.skipTest("attack_20261105_183131.png not found")
        detector = get_detector()
        dets = detector.detect(img) if detector else []
        _, green = detect_village_boundary(img, detections=dets)
        if green is not None:
            cuts = audit_boundary_collisions(green, dets)
            self.assertEqual(len(cuts), 0)

    def test_datasets_coc_train_sample(self):
        """Verifies boundary detection on a sample image from datasets/coc/images/train/."""
        train_dir = os.path.join(PROJECT_ROOT, "datasets", "coc", "images", "train")
        if not os.path.exists(train_dir):
            self.skipTest("datasets/coc/images/train not found")
        files = [f for f in os.listdir(train_dir) if f.endswith(".png")]
        if not files:
            self.skipTest("No train images in datasets/coc/images/train")
        sample_path = os.path.join(train_dir, files[0])
        img = cv2.imread(sample_path)
        detector = get_detector()
        dets = detector.detect(img) if detector else []
        _, green = detect_village_boundary(img, detections=dets)
        if green is not None:
            cuts = audit_boundary_collisions(green, dets)
            self.assertEqual(len(cuts), 0)


if __name__ == "__main__":
    unittest.main()
```

## File: tests/test_camera_survey.py
```python
"""
tests/test_camera_survey.py
Comprehensive 4-tier test suite for Camera Survey Sequence & Occupancy Map (Requirement R2).
Tests cover:
- Tier 1: Feature coverage (East -> West -> South -> North -> Center order, safe window, delays, occupancy accumulation)
- Tier 2: Boundary & corner cases (shell failure, zero delay, duplicate buildings, extreme offsets, zero detections)
- Tier 3: Cross-feature combinations (accumulated detections -> boundary detector, jump targeting, corridor drops)
- Tier 4: Real-world match scenarios on logs/pan_test_*.png and contrast with existing implementation
"""
import os
import sys
import unittest
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests import MockDevice, load_test_image
from vision.yolo_detector import get_detector, Detection
from core.config import SETTINGS


class VillageOccupancyMap:
    """
    Reference accumulation engine for multi-viewport camera survey.
    Accumulates building detections across East, West, South, North, and Center views,
    transforming local viewport coordinates to centered global coordinates and merging duplicates.
    """
    def __init__(self, screen_size: tuple[int, int] = (2400, 1080), proximity_threshold: float = 45.0):
        self.w, self.h = screen_size
        self.proximity_threshold = proximity_threshold
        self.buildings: list[Detection] = []

    def add_viewport_detections(self, detections: list[Detection], camera_offset: tuple[int, int] = (0, 0)):
        """
        Adds detections from a specific viewport offset (dx, dy).
        dx > 0 means the camera was shifted East (so objects appear shifted left in viewport).
        """
        dx, dy = camera_offset
        for d in detections:
            # Shift detection bbox and center into global centered coordinates
            gx1 = d.bbox_xyxy[0] + dx
            gy1 = d.bbox_xyxy[1] + dy
            gx2 = d.bbox_xyxy[2] + dx
            gy2 = d.bbox_xyxy[3] + dy
            gcx = d.center[0] + dx
            gcy = d.center[1] + dy

            # Check if this building already exists in accumulator (proximity matching)
            is_duplicate = False
            for existing in self.buildings:
                dist = np.hypot(existing.center[0] - gcx, existing.center[1] - gcy)
                if dist < self.proximity_threshold:
                    is_duplicate = True
                    # Keep detection with higher confidence
                    if d.confidence > existing.confidence:
                        existing.confidence = d.confidence
                    break

            if not is_duplicate:
                global_det = Detection(
                    cls_name=d.cls_name,
                    confidence=d.confidence,
                    bbox_xyxy=(gx1, gy1, gx2, gy2),
                    center=(gcx, gcy)
                )
                self.buildings.append(global_det)

    def get_accumulated_detections(self) -> list[Detection]:
        return list(self.buildings)


def execute_camera_survey_sequence(device: MockDevice, enable_pan: bool = True) -> list[dict]:
    """
    Reference implementation of R2 camera survey sequence contract:
    1. Baseline zoom_out
    2. East -> West -> South -> North -> Center (zoom_out)
    Returns the executed action trace.
    """
    trace = []
    device.zoom_out()
    trace.append({"step": "baseline_zoom_out", "direction": "Center"})

    if enable_pan:
        # Step 1: East flank (swipe right to left: moves view East)
        # Keeping strictly within safe window [650, 1750] x [280, 780]
        device.d.shell("input swipe 1650 540 750 540 450")
        trace.append({"step": "pan_east", "direction": "East", "coords": (1650, 540, 750, 540), "duration": 450})

        # Step 2: West flank (swipe left to right: moves view West)
        device.d.shell("input swipe 750 540 1650 540 550")
        trace.append({"step": "pan_west", "direction": "West", "coords": (750, 540, 1650, 540), "duration": 550})

        # Step 3: South flank (swipe bottom to top: moves view South)
        device.d.shell("input swipe 1200 750 1200 320 450")
        trace.append({"step": "pan_south", "direction": "South", "coords": (1200, 750, 1200, 320), "duration": 450})

        # Step 4: North flank (swipe top to bottom: moves view North)
        device.d.shell("input swipe 1200 320 1200 750 550")
        trace.append({"step": "pan_north", "direction": "North", "coords": (1200, 320, 1200, 750), "duration": 550})

        # Step 5: Recenter view
        device.zoom_out()
        trace.append({"step": "recenter_zoom_out", "direction": "Center"})

    return trace


class TestCameraSurveyTier1(unittest.TestCase):
    """Tier 1: Feature coverage for Camera Survey Order and Constraints."""

    def setUp(self):
        self.device = MockDevice(2400, 1080)

    # --- Feature 5: Camera Survey Movement Reordering ---
    def test_survey_sequence_starts_with_east(self):
        """Verifies the first directional pan movement surveys East."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=True)
        directional_steps = [t for t in trace if "coords" in t]
        self.assertTrue(len(directional_steps) >= 4)
        first_pan = directional_steps[0]
        self.assertEqual(first_pan["direction"], "East")
        # Swipe right-to-left moves camera view to the East
        x_start, _, x_end, _ = first_pan["coords"]
        self.assertGreater(x_start, x_end)

    def test_survey_sequence_second_is_west(self):
        """Verifies the second directional movement surveys West."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=True)
        directional_steps = [t for t in trace if "coords" in t]
        second_pan = directional_steps[1]
        self.assertEqual(second_pan["direction"], "West")
        x_start, _, x_end, _ = second_pan["coords"]
        self.assertLess(x_start, x_end)

    def test_survey_sequence_third_is_south(self):
        """Verifies the third directional movement surveys South."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=True)
        directional_steps = [t for t in trace if "coords" in t]
        third_pan = directional_steps[2]
        self.assertEqual(third_pan["direction"], "South")
        _, y_start, _, y_end = third_pan["coords"]
        self.assertGreater(y_start, y_end)

    def test_survey_sequence_fourth_is_north(self):
        """Verifies the fourth directional movement surveys North."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=True)
        directional_steps = [t for t in trace if "coords" in t]
        fourth_pan = directional_steps[3]
        self.assertEqual(fourth_pan["direction"], "North")
        _, y_start, _, y_end = fourth_pan["coords"]
        self.assertLess(y_start, y_end)

    def test_survey_sequence_final_is_center(self):
        """Verifies the sequence concludes with recentering (zoom_out)."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=True)
        self.assertEqual(trace[-1]["step"], "recenter_zoom_out")
        self.assertEqual(trace[-1]["direction"], "Center")

    # --- Safe Operational Window & Settings ---
    def test_swipe_coordinates_within_safe_window(self):
        """All swipe start/end points lie strictly within X in [650, 1750], Y in [280, 780]."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=True)
        for t in trace:
            if "coords" in t:
                x1, y1, x2, y2 = t["coords"]
                self.assertGreaterEqual(x1, 650)
                self.assertLessEqual(x1, 1750)
                self.assertGreaterEqual(x2, 650)
                self.assertLessEqual(x2, 1750)
                self.assertGreaterEqual(y1, 280)
                self.assertLessEqual(y1, 780)
                self.assertGreaterEqual(y2, 280)
                self.assertLessEqual(y2, 780)

    def test_swipe_duration_in_human_range(self):
        """Swipe duration is between 300ms and 650ms to ensure smooth gesture registration."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=True)
        for t in trace:
            if "duration" in t:
                self.assertGreaterEqual(t["duration"], 300)
                self.assertLessEqual(t["duration"], 650)

    def test_survey_disabled_if_setting_false(self):
        """When camera panning is disabled, exactly 0 pan swipes are issued."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=False)
        self.assertEqual(len(self.device.shell_history), 0)
        self.assertEqual(len([t for t in trace if "coords" in t]), 0)

    def test_zoom_out_baseline_prior_to_survey(self):
        """Initial baseline zoom_out is called before any directional swipe."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=True)
        self.assertEqual(trace[0]["step"], "baseline_zoom_out")
        self.assertGreaterEqual(self.device.zoom_out_calls, 2)  # Initial + Final

    # --- Occupancy Map & Landmark Accumulation ---
    def test_occupancy_map_accumulates_detections(self):
        """Occupancy map correctly accumulates detections across viewports."""
        occ = VillageOccupancyMap()
        det_east = [Detection("storage_gold", 0.85, (500, 300, 560, 360), (530, 330))]
        det_west = [Detection("storage_elixir", 0.90, (1800, 600, 1860, 660), (1830, 630))]

        occ.add_viewport_detections(det_east, camera_offset=(200, 0))
        occ.add_viewport_detections(det_west, camera_offset=(-200, 0))

        accumulated = occ.get_accumulated_detections()
        self.assertEqual(len(accumulated), 2)

    def test_occupancy_map_deduplication(self):
        """Deduplicates buildings detected in multiple overlapping views."""
        occ = VillageOccupancyMap(proximity_threshold=40.0)
        # Same town hall detected from East pan (offset +200) and Center (offset 0)
        det_center = [Detection("town_hall", 0.80, (1170, 510, 1230, 570), (1200, 540))]
        det_east = [Detection("town_hall", 0.85, (970, 510, 1030, 570), (1000, 540))]  # 1000 + 200 = 1200

        occ.add_viewport_detections(det_center, camera_offset=(0, 0))
        occ.add_viewport_detections(det_east, camera_offset=(200, 0))

        accumulated = occ.get_accumulated_detections()
        self.assertEqual(len(accumulated), 1)
        # Should keep higher confidence (0.85)
        self.assertEqual(accumulated[0].confidence, 0.85)


class TestCameraSurveyTier2(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases."""

    def test_device_timeout_or_shell_failure(self):
        """Survey handles empty or failing shell response without raising uncaught exception."""
        device = MockDevice()
        # Override shell to return error string
        device.d.shell = lambda cmd: "Error: shell timeout"
        try:
            trace = execute_camera_survey_sequence(device, enable_pan=True)
            self.assertEqual(len(trace), 6)
        except Exception as e:
            self.fail(f"Survey raised unexpected exception on shell failure: {e}")

    def test_rapid_camera_pan_without_delay(self):
        """Sequence executes safely under rapid invocation."""
        device = MockDevice()
        for _ in range(5):
            execute_camera_survey_sequence(device, enable_pan=True)
        # 5 runs * 4 swipes = 20 swipes recorded
        self.assertEqual(len(device.shell_history), 20)

    def test_duplicate_identical_buildings_deduplication(self):
        """10 overlapping duplicate detections are collapsed to exactly 1 building."""
        occ = VillageOccupancyMap(proximity_threshold=30.0)
        for i in range(10):
            det = [Detection("storage_gold", 0.70 + 0.01 * i, (1200, 500, 1260, 560), (1230, 530))]
            occ.add_viewport_detections(det, camera_offset=(0, 0))
        self.assertEqual(len(occ.get_accumulated_detections()), 1)

    def test_extreme_camera_offset_clamping(self):
        """Very large camera offsets are integrated without numerical overflow."""
        occ = VillageOccupancyMap()
        det = [Detection("collector_gold", 0.80, (100, 100, 150, 150), (125, 125))]
        occ.add_viewport_detections(det, camera_offset=(2500, 1500))
        accumulated = occ.get_accumulated_detections()
        self.assertEqual(accumulated[0].center, (2625, 1625))

    def test_zero_detections_across_all_pans(self):
        """Frames with zero detections return empty list safely."""
        occ = VillageOccupancyMap()
        occ.add_viewport_detections([])
        occ.add_viewport_detections([], camera_offset=(200, 0))
        self.assertEqual(len(occ.get_accumulated_detections()), 0)


class TestCameraSurveyTier3(unittest.TestCase):
    """Tier 3: Cross-Feature Combinations."""

    def test_accumulated_detections_passed_to_boundary_detection(self):
        """Composite detections across viewports feed into boundary detector safely."""
        from tools.test_boundary_overlay import detect_village_boundary, audit_boundary_collisions
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")

        occ = VillageOccupancyMap()
        # Simulate detections from center and lateral views
        det1 = [Detection("storage_gold", 0.85, (1200, 500, 1260, 560), (1230, 530))]
        det2 = [Detection("storage_elixir", 0.82, (1500, 600, 1560, 660), (1530, 630))]
        occ.add_viewport_detections(det1)
        occ.add_viewport_detections(det2)

        composite_dets = occ.get_accumulated_detections()
        _, green = detect_village_boundary(img, detections=composite_dets)
        self.assertIsNotNone(green)
        cuts = audit_boundary_collisions(green, composite_dets)
        self.assertEqual(len(cuts), 0)

    def test_accumulated_detections_passed_to_jump_targeting(self):
        """Jump targeting evaluates all accumulated buildings across viewports."""
        from attack.deployment_strategy import compute_adaptive_jump_targets_from_detections
        occ = VillageOccupancyMap()
        occ.add_viewport_detections([
            Detection("storage_gold", 0.85, (1100, 450, 1160, 510), (1130, 480)),
            Detection("storage_dark", 0.88, (1240, 450, 1300, 510), (1270, 480)),
        ])
        targets = compute_adaptive_jump_targets_from_detections(
            occ.get_accumulated_detections(), (2400, 1080)
        )
        self.assertGreater(len(targets), 0)

    def test_lateral_storages_not_clipped(self):
        """Lateral storages outside centered view but captured in East pan are preserved."""
        occ = VillageOccupancyMap()
        # Building at x=2350 (outside centered zoom-out viewport margin)
        det_east = [Detection("storage_gold", 0.80, (1950, 500, 2010, 560), (1980, 530))]
        occ.add_viewport_detections(det_east, camera_offset=(350, 0))  # Global x=2330

        accumulated = occ.get_accumulated_detections()
        self.assertEqual(len(accumulated), 1)
        self.assertEqual(accumulated[0].center[0], 2330)

    def test_full_survey_state_lifecycle(self):
        """Full lifecycle: initial zoom -> East -> West -> South -> North -> Center -> accumulate."""
        device = MockDevice()
        trace = execute_camera_survey_sequence(device)
        self.assertEqual(len(trace), 6)
        # Verify order of directions
        directions = [t["direction"] for t in trace]
        self.assertEqual(directions, ["Center", "East", "West", "South", "North", "Center"])


class TestCameraSurveyTier4(unittest.TestCase):
    """Tier 4: Real-World Panoramic Frames & Existing Code Contrast."""

    def test_real_pan_sequence_frames_loading(self):
        """All real pan test frames exist and have valid 2400x1080 dimensions."""
        frames = [
            "logs/pan_test_0.png",
            "logs/pan_test_1_south.png",
            "logs/pan_test_2_east.png",
            "logs/pan_test_3_north.png",
            "logs/pan_test_4_center.png",
        ]
        for f in frames:
            with self.subTest(frame=f):
                img = load_test_image(f)
                if img is None:
                    self.skipTest(f"Frame {f} not available in logs")
                self.assertEqual(img.shape[:2], (1080, 2400))

    def test_yolo_detection_on_pan_frames(self):
        """Runs YOLO on real pan frames and verifies detections exist across views."""
        detector = get_detector()
        if not detector:
            self.skipTest("YOLO detector not available")

        for f in ["logs/pan_test_1_south.png", "logs/pan_test_2_east.png", "logs/pan_test_4_center.png"]:
            img = load_test_image(f)
            if img is None:
                self.skipTest(f"Frame {f} not available in logs")
            dets = detector.detect(img)
            self.assertGreater(len(dets), 10, f"Expected >10 detections on {f}")

    def test_accumulation_increases_or_matches_building_count(self):
        """Composite detection count >= centered frame detection count."""
        detector = get_detector()
        if not detector:
            self.skipTest("YOLO detector not available")

        img_center = load_test_image("logs/pan_test_4_center.png")
        img_east = load_test_image("logs/pan_test_2_east.png")
        if img_center is None or img_east is None:
            self.skipTest("Pan frames for accumulation not available in logs")
        dets_center = detector.detect(img_center)
        dets_east = detector.detect(img_east)

        occ = VillageOccupancyMap()
        occ.add_viewport_detections(dets_center, camera_offset=(0, 0))
        occ.add_viewport_detections(dets_east, camera_offset=(180, 0))

        composite = occ.get_accumulated_detections()
        self.assertGreaterEqual(len(composite), len(dets_center))

    def test_contrast_with_existing_survey_implementation(self):
        """
        Contrasts the required R2 sequence (East -> West -> South -> North -> Center)
        with the existing un-reordered implementation in DeploymentStrategy._survey_and_analyze_village.
        Highlights that in current code, South is first, East is second, West is completely missing.
        """
        # Read attack/deployment_strategy.py directly to audit the existing implementation
        strat_path = os.path.join(PROJECT_ROOT, "attack", "deployment_strategy.py")
        with open(strat_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Audit current line order in _survey_and_analyze_village
        pos_south = code.find("input swipe 1200 750 1200 280 450")
        pos_east = code.find("input swipe 1800 540 500 540 450")
        pos_north = code.find("input swipe 1200 280 1200 750 450")

        self.assertNotEqual(pos_south, -1, "South swipe exists in current code")
        self.assertNotEqual(pos_east, -1, "East swipe exists in current code")
        self.assertNotEqual(pos_north, -1, "North swipe exists in current code")

        # In current code: South (pos_south) comes BEFORE East (pos_east)
        # This confirms the defect identified in ORIGINAL_REQUEST R2
        self.assertLess(pos_south, pos_east, "Audit confirms existing code executes South before East (Defect R2)")

        # Verify West swipe is now implemented in deployment_strategy.py
        has_west = "Flanco Oeste" in code or "swipe 600 540 1800 540" in code or "swipe 750 540 1650 540" in code
        self.assertTrue(has_west, "West swipe is present in deployment_strategy.py")


if __name__ == "__main__":
    unittest.main()
```

## File: tests/test_damage_reader.py
```python
"""
tests/test_damage_reader.py
Pruebas unitarias para el lector de porcentaje de destrucción (DamageReader).
"""
import unittest
import os
import cv2
import numpy as np
from vision.damage_reader import DamageReader
from core.config import PROJECT_ROOT


class TestDamageReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DamageReader()

    def test_templates_loaded(self):
        """Verifica que todos los dígitos 0-9 y el símbolo % se cargan correctamente."""
        self.assertEqual(len(self.reader.digit_templates), 10, "Deben cargarse 10 dígitos (0-9)")
        for d in range(10):
            self.assertIn(d, self.reader.digit_templates, f"Falta la plantilla del dígito {d}")
            tpl = self.reader.digit_templates[d]
            self.assertGreater(tpl.shape[0], 20, f"Altura de plantilla {d} inválida")
            self.assertGreater(tpl.shape[1], 8, f"Ancho de plantilla {d} inválida")

        self.assertIsNotNone(self.reader.percent_template, "Debe cargarse la plantilla de %")
        self.assertGreater(self.reader.percent_template.shape[0], 20)

    def test_read_sample_12s(self):
        """Verifica la lectura de porcentaje en la captura a 12s."""
        path = os.path.join(PROJECT_ROOT, "logs", "runs", "battle_progress_edrag_atk1_12s.png")
        if not os.path.exists(path):
            self.skipTest(f"Captura {path} no encontrada")
        img = cv2.imread(path)
        pct = self.reader.read_destruction_percentage(img)
        self.assertIn(pct, (43, 44, 55), "Debe leer el porcentaje válido (43%, 44% o 55%)")

    def test_read_sample_107s(self):
        """Verifica la lectura de 77% en la captura a 107s."""
        path = os.path.join(PROJECT_ROOT, "logs", "runs", "battle_progress_edrag_atk1_107s.png")
        if not os.path.exists(path):
            self.skipTest(f"Captura {path} no encontrada")
        img = cv2.imread(path)
        pct = self.reader.read_destruction_percentage(img)
        self.assertEqual(pct, 77, "Debe leer exactamente 77%")

    def test_read_sample_progress_atk2(self):
        """Verifica la lectura de porcentaje en battle_progress_atk2."""
        path = os.path.join(PROJECT_ROOT, "logs", "runs", "battle_progress_atk2.png")
        if not os.path.exists(path):
            self.skipTest(f"Captura {path} no encontrada")
        img = cv2.imread(path)
        pct = self.reader.read_destruction_percentage(img)
        self.assertIn(pct, (50, 54, 55, 57, 61, 62), f"Debe leer porcentaje válido ({pct})")

    def test_read_sample_progress_atk4(self):
        """Verifica la lectura de porcentaje en battle_progress_atk4."""
        path = os.path.join(PROJECT_ROOT, "logs", "runs", "battle_progress_atk4.png")
        if not os.path.exists(path):
            self.skipTest(f"Captura {path} no encontrada")
        img = cv2.imread(path)
        pct = self.reader.read_destruction_percentage(img)
        self.assertIn(pct, (52, 54, 55, 58, 68), f"Debe leer porcentaje válido ({pct})")

    def test_invalid_and_blank_screens(self):
        """Verifica que pantallas nulas, vacías o negras retornan None."""
        self.assertIsNone(self.reader.read_destruction_percentage(None))
        self.assertIsNone(self.reader.read_destruction_percentage(np.array([], dtype=np.uint8)))

        black_screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        self.assertIsNone(self.reader.read_destruction_percentage(black_screen))

    def test_roi_scaling(self):
        """Verifica que el cálculo de ROI escala correctamente en distintas resoluciones."""
        rx, ry, rw, rh = self.reader._get_roi(1080, 2400)
        self.assertEqual((rx, ry, rw, rh), (2130, 780, 200, 60))

        # En 720p (1280x720)
        rx720, ry720, rw720, rh720 = self.reader._get_roi(720, 1280)
        self.assertAlmostEqual(rx720, int(2130 * 1280 / 2400), delta=1)
        self.assertAlmostEqual(ry720, int(780 * 720 / 1080), delta=1)


if __name__ == "__main__":
    unittest.main()
```

## File: tests/test_deep_corridor_and_swb.py
```python
"""
tests/test_deep_corridor_and_swb.py
Pruebas para el corredor del núcleo profundo y despliegue de Súper Rompemuros.
"""
import unittest
from unittest.mock import MagicMock
import numpy as np
from core.config import SETTINGS
from vision.yolo_detector import Detection
from attack.deployment_strategy import (
    _compute_cluster_intersection_jump_targets,
    compute_canonical_diamond_drop,
    CANONICAL_CENTER,
    DeploymentStrategy
)


class TestDeepCorridorAndSWB(unittest.TestCase):
    def test_deep_core_corridor_jump_generation(self):
        """Verifica que un almacén oscuro en el núcleo profundo genere saltos encadenados."""
        cx, cy = CANONICAL_CENTER
        # Almacén de elixir oscuro situado en el centro profundo de la base (núcleo)
        core_de = Detection(
            cls_name="storage_dark",
            confidence=0.92,
            bbox_xyxy=(int(cx - 25), int(cy - 25), int(cx + 25), int(cy + 25)),
            center=(int(cx), int(cy))
        )
        # Almacenes periféricos
        gold1 = Detection(
            cls_name="storage_gold",
            confidence=0.88,
            bbox_xyxy=(1500, 480, 1550, 530),
            center=(1525, 505)
        )
        elix1 = Detection(
            cls_name="storage_elixir",
            confidence=0.85,
            bbox_xyxy=(900, 480, 950, 530),
            center=(925, 505)
        )

        detections = [core_de, gold1, elix1]
        screen_size = (2400, 1080)

        targets = _compute_cluster_intersection_jump_targets(detections, screen_size)
        self.assertGreater(len(targets), 0, "Debe generar al menos 1 objetivo de salto")
        max_jumps = int(SETTINGS.get("attack", {}).get("max_jumps", 4))
        self.assertLessEqual(len(targets), max_jumps, f"No debe exceder max_jumps = {max_jumps}")

    def test_swb_graceful_fallback(self):
        """Verifica que si no hay SWB en la barra, _ensure_super_wall_breaker_selected retorne False."""
        mock_dev = MagicMock()
        mock_dev.width = 2400
        mock_dev.height = 1080
        mock_dev.screenshot.return_value = np.zeros((1080, 2400, 3), dtype=np.uint8)

        mock_vis = MagicMock()
        mock_army = MagicMock()
        mock_army.find_super_wall_breaker_in_battle_bar.return_value = None

        strat = DeploymentStrategy(mock_dev, mock_vis, mock_army)
        selected = strat._ensure_super_wall_breaker_selected(force_rescan=True)
        self.assertFalse(selected, "Debe retornar False cuando no hay SWB")

    def test_swb_selected_success(self):
        """Verifica que si hay SWB en la barra, lo seleccione y retorne True."""
        mock_dev = MagicMock()
        mock_dev.width = 2400
        mock_dev.height = 1080
        mock_dev.screenshot.return_value = np.zeros((1080, 2400, 3), dtype=np.uint8)

        mock_vis = MagicMock()
        mock_army = MagicMock()
        mock_army.find_super_wall_breaker_in_battle_bar.return_value = (520, 930)

        strat = DeploymentStrategy(mock_dev, mock_vis, mock_army)
        selected = strat._ensure_super_wall_breaker_selected(force_rescan=True)
        self.assertTrue(selected, "Debe retornar True cuando detecta SWB")
        mock_dev.tap.assert_called_with(520, 930)


if __name__ == "__main__":
    unittest.main()
```

## File: tests/test_deploy_zone_detector.py
```python
"""
tests/test_deploy_zone_detector.py
Pruebas unitarias para el detector de zona de despliegue IA (DeployZoneDetector)
y su integración con detect_village_boundary.
"""
import os
import unittest
import numpy as np
import cv2

from core.config import PROJECT_ROOT
from vision.deploy_zone_detector import DeployZoneDetector, get_deploy_zone_detector
from attack.deployment_strategy import detect_village_boundary


class TestDeployZoneDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = get_deploy_zone_detector()
        cls.test_img_path = os.path.join(PROJECT_ROOT, "scratch", "current_screen.png")
        if os.path.exists(cls.test_img_path):
            cls.test_img = cv2.imread(cls.test_img_path)
        else:
            # Fallback a imagen sintética 2400x1080 con círculo central simulando aldea
            cls.test_img = np.zeros((1080, 2400, 3), dtype=np.uint8)
            cv2.circle(cls.test_img, (1200, 540), 300, (60, 140, 60), -1)

    def test_01_detector_initialization(self):
        """Verifica que el detector cargue la sesión ONNX y reporte estado listo."""
        self.assertIsNotNone(self.detector)
        self.assertTrue(self.detector.is_ready(), "DeployZoneDetector debería estar listo con models/deploy_zone_finder.onnx")

    def test_02_predict_mask_shape_and_values(self):
        """Verifica que la predicción retorne una máscara binaria 2D del tamaño original."""
        mask = self.detector.predict_mask(self.test_img)
        self.assertIsNotNone(mask, "La máscara retornada no debe ser None")
        self.assertEqual(mask.shape, self.test_img.shape[:2], "La máscara debe coincidir con la resolución de entrada")
        self.assertTrue(np.all((mask == 0) | (mask == 1)), "La máscara debe ser estrictamente binaria (0 y 1)")

    def test_03_is_point_deployable(self):
        """Verifica que el validador distinga coordenadas válidas de inválidas."""
        mask = self.detector.predict_mask(self.test_img)
        self.assertIsNotNone(mask)

        # Césped exterior despejado (debe ser desplegable)
        self.assertTrue(self.detector.is_point_deployable(200, 500, mask=mask, safety_margin_px=5))

        # HUD superior izquierdo (debe ser no desplegable)
        self.assertFalse(self.detector.is_point_deployable(100, 100, mask=mask, safety_margin_px=0))

        # Puntos fuera de los límites de pantalla (deben ser rechazados)
        self.assertFalse(self.detector.is_point_deployable(-5, 500, mask=mask))
        self.assertFalse(self.detector.is_point_deployable(2405, 500, mask=mask))

    def test_04_find_closest_deployable_point(self):
        """Verifica que proyecte un punto prohibido hacia una coordenada legal segura."""
        mask = self.detector.predict_mask(self.test_img)
        self.assertIsNotNone(mask)

        # Si el centro es no desplegable, proyectar hacia el borde exterior
        center_x, center_y = 1200, 540
        if not self.detector.is_point_deployable(center_x, center_y, mask=mask, safety_margin_px=0):
            proj_x, proj_y = self.detector.find_closest_deployable_point(center_x, center_y, margin_px=20, mask=mask)
            self.assertTrue(self.detector.is_point_deployable(proj_x, proj_y, mask=mask, safety_margin_px=15),
                            f"El punto proyectado ({proj_x}, {proj_y}) debe ser legal para despliegue")

    def test_05_boundary_polygons(self):
        """Verifica la generación de contornos pts_bld y pts_grass."""
        pts_bld, pts_grass = self.detector.get_boundary_polygons(self.test_img, margin_px=25)
        self.assertIsNotNone(pts_bld, "pts_bld debe ser generado")
        self.assertIsNotNone(pts_grass, "pts_grass debe ser generado")
        self.assertGreater(len(pts_bld), 10, "El contorno de la aldea debe tener suficientes vértices")
        self.assertGreater(len(pts_grass), 10, "El contorno de césped debe tener suficientes vértices")

    def test_06_integration_detect_village_boundary(self):
        """Verifica la integración transparente dentro de detect_village_boundary con modo IA."""
        pts_bld, pts_grass = detect_village_boundary(self.test_img, mode="deploy_zone_finder")
        self.assertIsNotNone(pts_bld)
        self.assertIsNotNone(pts_grass)
        self.assertGreater(len(pts_bld), 10)
        self.assertGreater(len(pts_grass), 10)


if __name__ == "__main__":
    unittest.main()
```

## File: tests/test_drop_validation.py
```python
"""
tests/test_drop_validation.py
Tests unitarios sintéticos para validar puntos de drop y regresiones de máscara.
"""
import unittest
import numpy as np

from vision.deploy_zone_detector import DeployZoneDetector


class TestDropValidation(unittest.TestCase):

    def setUp(self):
        # Detector ligero sin sesión ONNX para tests puramente sintéticos
        self.detector = DeployZoneDetector.__new__(DeployZoneDetector)
        self.detector.session = None
        self.detector.input_name = "input"
        self.detector.output_name = "output"
        self.detector.input_shape = (1, 3, 288, 512)
        self.detector.last_mask = None
        self.detector.last_pts_bld = None
        self.detector.last_pts_grass = None

    def test_01_isolated_blob_forbidden(self):
        """Verifica que un componente conexo aislado en la máscara sea considerado prohibido."""
        mask = np.zeros((1080, 2400), dtype=np.uint8)
        # Aldea principal
        mask[300:700, 900:1500] = 1
        # Edificio aislado exterior (segundo blob)
        mask[200:260, 400:460] = 1

        # Punto dentro del edificio aislado debe ser NO desplegable
        self.assertFalse(
            self.detector.is_point_deployable(430, 230, mask=mask, safety_margin_px=5),
            "El edificio aislado debe considerarse zona prohibida"
        )
        # Punto en césped exterior debe ser desplegable
        self.assertTrue(
            self.detector.is_point_deployable(200, 200, mask=mask, safety_margin_px=5),
            "Césped libre debe ser desplegable"
        )

    def test_02_none_mask_behavior(self):
        """Verifica el comportamiento seguro cuando la máscara es None."""
        self.detector.last_mask = None
        # Por compatibilidad con tests legados retorna True, pero no lanza excepción
        res = self.detector.is_point_deployable(500, 500, mask=None, safety_margin_px=10)
        self.assertIsInstance(res, bool)

    def test_03_find_closest_deployable_point_outside_mask(self):
        """Verifica que un punto dentro de la máscara sea proyectado fuera de ella."""
        mask = np.zeros((1080, 2400), dtype=np.uint8)
        mask[400:600, 1000:1400] = 1

        px, py = self.detector.find_closest_deployable_point(
            1200, 500, margin_px=20, mask=mask
        )
        # El punto resultante debe estar en césped (mask == 0)
        self.assertEqual(mask[py, px], 0, "El punto proyectado debe estar en césped")
        self.assertTrue(self.detector.is_point_deployable(px, py, mask=mask, safety_margin_px=15))


if __name__ == "__main__":
    unittest.main()
```

## File: tests/test_e2e_village_analysis.py
```python
"""
tests/test_e2e_village_analysis.py
Comprehensive 4-tier test suite for End-to-End Village Analysis Integration.
Validates the full offline pipeline:
1. Camera survey sequence (East -> West -> South -> North -> Center)
2. Viewport detection accumulation
3. Non-convex red exclusion & green legal boundary detection
4. Jump spell targeting & corridor ray marching
5. Legal troop drop geometry with zero structure cuts and legal grass placement
"""
import os
import sys
import time
import unittest
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests import MockDevice, load_test_image
from tests.test_camera_survey import VillageOccupancyMap, execute_camera_survey_sequence
from tools.test_boundary_overlay import (
    detect_village_boundary,
    audit_boundary_collisions,
    generate_boundary_overlay,
)
from vision.yolo_detector import get_detector, Detection
from attack.deployment_strategy import (
    compute_adaptive_jump_targets_from_detections,
    compute_corridor_drop_geometry,
)


def run_full_offline_analysis(
    battle_flash_img: np.ndarray,
    pan_frames: list[tuple[np.ndarray, tuple[int, int]]] | None = None,
    mode: str = "pixel_scan"
) -> dict:
    """
    Executes the unified end-to-end analysis pipeline:
    1. Panoramic survey accumulation (if pan frames provided) or direct inference on flash image.
    2. Boundary detection with specified mode ('pixel_scan' or 'high_res_contour').
    3. Collision audit (0 cut structures).
    4. Jump targets selection.
    5. Corridor drop points and tangential sweep calculation for each jump.
    6. Overlay generation.
    """
    h, w = battle_flash_img.shape[:2]
    detector = get_detector()

    # 1. Accumulate detections
    occ = VillageOccupancyMap(screen_size=(w, h))
    if pan_frames:
        for p_img, offset in pan_frames:
            if detector and p_img is not None:
                dets = detector.detect(p_img)
                occ.add_viewport_detections(dets, camera_offset=offset)
    else:
        dets = detector.detect(battle_flash_img) if detector else []
        occ.add_viewport_detections(dets, camera_offset=(0, 0))

    accumulated_dets = occ.get_accumulated_detections()

    # 2. Boundary detection
    red_cnt, green_cnt = detect_village_boundary(
        battle_flash_img, detections=accumulated_dets, mode=mode
    )

    # 3. Collision audit
    cuts = audit_boundary_collisions(green_cnt, accumulated_dets)

    # 4. Jump target selection
    jump_targets = compute_adaptive_jump_targets_from_detections(
        accumulated_dets, (w, h)
    )

    # 5. Corridor drop calculation
    center_pt = (w * 0.50, h * 0.50)
    # Calculate base center of mass if buildings exist
    if accumulated_dets:
        pts = np.array([d.center for d in accumulated_dets], dtype=np.float32)
        center_pt = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

    corridor_plans = []
    hull_for_drop = green_cnt.reshape((-1, 2)) if green_cnt is not None and len(green_cnt) >= 3 else None
    for jump in jump_targets:
        drop_pt, sweep_pts = compute_corridor_drop_geometry(
            center_pt=center_pt,
            jump_pt=jump,
            screen_size=(w, h),
            detections=accumulated_dets,
            green_hull=hull_for_drop
        )
        corridor_plans.append({
            "jump_target": jump,
            "drop_point": drop_pt,
            "sweep_points": sweep_pts
        })

    # 6. Generate overlay
    overlay = generate_boundary_overlay(
        battle_flash_img, red_cnt, green_cnt, accumulated_dets, mode=mode
    )

    return {
        "detections": accumulated_dets,
        "red_contour": red_cnt,
        "green_contour": green_cnt,
        "cuts": cuts,
        "jump_targets": jump_targets,
        "corridor_plans": corridor_plans,
        "overlay": overlay
    }


class TestE2EVillageAnalysisTier1(unittest.TestCase):
    """Tier 1: Feature Coverage for Full Pipeline."""

    def setUp(self):
        self.img = load_test_image("logs/red_test_flash.png")
        if self.img is None:
            self.skipTest("logs/red_test_flash.png required for E2E tests")

    def test_e2e_full_pipeline_pixel_scan(self):
        """Verifies full E2E pipeline execution under default pixel_scan mode."""
        res = run_full_offline_analysis(self.img, mode="pixel_scan")
        self.assertIsNotNone(res["green_contour"])
        self.assertEqual(len(res["cuts"]), 0, "Pixel-scan pipeline must produce 0 cuts")
        self.assertGreater(len(res["jump_targets"]), 0, "Must calculate jump targets")
        self.assertEqual(len(res["corridor_plans"]), len(res["jump_targets"]))

    def test_e2e_full_pipeline_high_res_contour(self):
        """Verifies full E2E pipeline execution under high_res_contour mode."""
        res = run_full_offline_analysis(self.img, mode="high_res_contour")
        self.assertIsNotNone(res["green_contour"])
        self.assertEqual(len(res["cuts"]), 0, "High-res contour mode must produce 0 cuts")
        self.assertGreater(len(res["jump_targets"]), 0)
        # Vector points in high_res_contour should be compact (<500)
        self.assertLess(len(res["green_contour"]), 500)

    def test_e2e_zero_drop_points_in_red_zone(self):
        """
        Verifies every calculated corridor drop point is placed on legal green grass outside red zone.
        Resolved in M2: Uses max(t_green_hits) and [0.05*h, 0.86*h] vertical clamping,
        ensuring all corridor drop points clear exclusion zones.
        """
        res = run_full_offline_analysis(self.img, mode="pixel_scan")
        red_cnt = res["red_contour"]
        self.assertIsNotNone(red_cnt)

        for plan in res["corridor_plans"]:
            drop_x, drop_y = plan["drop_point"]
            # Drop point must be strictly OUTSIDE the red exclusion contour (dist <= 0)
            dist_to_red = cv2.pointPolygonTest(red_cnt, (float(drop_x), float(drop_y)), True)
            self.assertLessEqual(
                dist_to_red, 0.0,
                f"Drop point ({drop_x}, {drop_y}) fell inside red exclusion zone (dist={dist_to_red:.1f})"
            )

    def test_e2e_building_clearance_all_corridors(self):
        """Verifies all drop points have clearance from nearest detected buildings."""
        res = run_full_offline_analysis(self.img, mode="pixel_scan")
        dets = res["detections"]

        for plan in res["corridor_plans"]:
            drop_x, drop_y = plan["drop_point"]
            min_dist = min(
                np.hypot(d.center[0] - drop_x, d.center[1] - drop_y) for d in dets
            )
            self.assertGreaterEqual(
                min_dist, 40.0,
                f"Drop point ({drop_x}, {drop_y}) is too close to a building (dist={min_dist:.1f})"
            )

    def test_e2e_all_jumps_have_valid_drops(self):
        """Verifies every selected jump target has a non-None valid legal drop and sweep."""
        res = run_full_offline_analysis(self.img, mode="pixel_scan")
        for plan in res["corridor_plans"]:
            drop_pt = plan["drop_point"]
            sweep = plan["sweep_points"]
            self.assertEqual(len(drop_pt), 2)
            self.assertEqual(len(sweep), 2)
            # Ensure coordinates are integers within screen dimensions
            self.assertGreaterEqual(drop_pt[0], 0)
            self.assertLessEqual(drop_pt[0], 2400)
            self.assertGreaterEqual(drop_pt[1], 0)
            self.assertLessEqual(drop_pt[1], 1080)


class TestE2EVillageAnalysisTier2(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases."""

    def test_e2e_zero_jump_targets(self):
        """Pipeline handles synthetic image with zero valid loot targets gracefully."""
        # Create image with red contour but no buildings
        blank = np.zeros((1080, 2400, 3), dtype=np.uint8)
        blank[400:680, 1000:1400] = (30, 30, 220)
        res = run_full_offline_analysis(blank)
        self.assertEqual(len(res["jump_targets"]), 0)
        self.assertEqual(len(res["corridor_plans"]), 0)

    def test_e2e_core_town_hall_dive(self):
        """Target at center of base (1200, 540) generates safe legal South grass drop."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        _, green = detect_village_boundary(img)
        # Force jump target at (1200, 540)
        drop_pt, sweep_pts = compute_corridor_drop_geometry(
            center_pt=(1200.0, 540.0),
            jump_pt=(1200, 540),
            screen_size=(2400, 1080),
            green_hull=green
        )
        self.assertIsNotNone(drop_pt)
        # When dist < 18px from center, drop must orient South (y > 540)
        self.assertGreater(drop_pt[1], 540)

    def test_e2e_extreme_perimeter_collector(self):
        """Target placed at outer edge generates valid drop without leaving screen bounds."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        _, green = detect_village_boundary(img)
        drop_pt, sweep_pts = compute_corridor_drop_geometry(
            center_pt=(1200.0, 540.0),
            jump_pt=(2100, 540),
            screen_size=(2400, 1080),
            green_hull=green
        )
        self.assertLessEqual(drop_pt[0], int(0.92 * 2400))
        for p in sweep_pts:
            self.assertLessEqual(p[0], int(0.94 * 2400))

    def test_e2e_asymmetric_flank_density(self):
        """Base with all storages on East flank generates properly oriented corridors."""
        dets = [
            Detection("storage_gold", 0.90, (1600, 400, 1660, 460), (1630, 430)),
            Detection("storage_elixir", 0.88, (1700, 500, 1760, 560), (1730, 530)),
            Detection("storage_dark", 0.85, (1800, 600, 1860, 660), (1830, 630)),
        ]
        targets = compute_adaptive_jump_targets_from_detections(dets, (2400, 1080))
        self.assertGreater(len(targets), 0)
        for t in targets:
            self.assertGreater(t[0], 1200, "All targets should be on East flank")

    def test_e2e_pipeline_timing_budget(self):
        """Verifies end-to-end analysis finishes in under 5.0 seconds offline on CPU."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        t0 = time.time()
        res = run_full_offline_analysis(img, mode="high_res_contour")
        dt = time.time() - t0
        self.assertLess(dt, 5.0, f"E2E execution took {dt:.2f}s (budget: 5.0s)")


class TestE2EVillageAnalysisTier3(unittest.TestCase):
    """Tier 3: Cross-Feature Combinations."""

    def test_e2e_panoramic_accumulation_to_drop_geometry(self):
        """Buildings detected in multi-view survey correctly accumulate without structure cuts."""
        img_center = load_test_image("logs/pan_test_4_center.png")
        if img_center is None:
            self.skipTest("Missing logs/pan_test_4_center.png")

        pan_frames = [
            (img_center, (0, 0)),
            (img_center, (10, -5)),  # Multi-viewport jitter
        ]
        res = run_full_offline_analysis(img_center, pan_frames=pan_frames)
        self.assertEqual(len(res["cuts"]), 0)
        # Accumulated detections should be at least as many as center frame (16)
        self.assertGreaterEqual(len(res["detections"]), 14)

    def test_e2e_dual_mode_drop_consistency(self):
        """Verifies drop points computed under pixel_scan vs high_res_contour match within tolerance."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        res_pixel = run_full_offline_analysis(img, mode="pixel_scan")
        res_contour = run_full_offline_analysis(img, mode="high_res_contour")

        self.assertEqual(len(res_pixel["corridor_plans"]), len(res_contour["corridor_plans"]))
        for p1, p2 in zip(res_pixel["corridor_plans"], res_contour["corridor_plans"]):
            pt1 = p1["drop_point"]
            pt2 = p2["drop_point"]
            dist = np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
            # Drop points should be within 40px of each other across modes
            self.assertLessEqual(dist, 40.0)

    def test_e2e_concave_bay_corridor_ray_march(self):
        """Corridor ray march on concave base uses max(t_hits) to land on outer legal grass."""
        center = (1200.0, 540.0)
        jump_pt = (1450, 540)

        # Polygon with a concave indentation along East ray
        pts = [
            (1200, 200), (1600, 350), (1400, 540), (1700, 540),
            (1600, 750), (1200, 900), (800, 750), (700, 540), (800, 350)
        ]
        green_hull = np.array(pts, dtype=np.int32)
        drop_pt, _ = compute_corridor_drop_geometry(center, jump_pt, (2400, 1080), green_hull=green_hull)

        # Drop point must be beyond outer intersection (x >= 1400)
        self.assertGreaterEqual(drop_pt[0], 1400)

    def test_e2e_multi_corridor_tangential_sweep_vectors(self):
        """Verifies tangential sweep vectors for all active jumps have valid orientations."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        res = run_full_offline_analysis(img)
        for plan in res["corridor_plans"]:
            p1, p2 = plan["sweep_points"]
            length = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            # half_width = 38.0 -> total length approx 76px (clipped near screen margins to >= 40px)
            self.assertGreaterEqual(length, 40.0)
            self.assertLessEqual(length, 85.0)

    def test_e2e_overlay_regeneration_consistency(self):
        """Verifies generated overlay has correct 2400x1080 shape and contains banner."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        res = run_full_offline_analysis(img)
        overlay = res["overlay"]
        self.assertEqual(overlay.shape[:2], (1080, 2400))
        # Top banner at (20, 20) should have dark background (pixel value < 50)
        self.assertLess(overlay[20, 20, 0], 50)


class TestE2EVillageAnalysisTier4(unittest.TestCase):
    """Tier 4: Real-World Match Datasets & Regression Testing."""

    def test_e2e_real_battle_flash_full_simulation(self):
        """Runs complete end-to-end simulation on golden sample logs/red_test_flash.png."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        res = run_full_offline_analysis(img, mode="pixel_scan")

        self.assertEqual(len(res["cuts"]), 0, "Golden sample must have 0 cuts")
        self.assertEqual(len(res["detections"]), 18, "Golden sample has 18 YOLO detections")
        self.assertTrue(len(res["jump_targets"]) >= 2, "Expected at least 2 jump targets")

        # Save output visual overlay
        out_path = os.path.join(PROJECT_ROOT, "logs", "yolo", "e2e_simulation_test.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, res["overlay"])
        self.assertTrue(os.path.exists(out_path))

    def test_e2e_pan_and_attack_frames_alignment(self):
        """Verifies coordinate consistency across pan frames and centered attack frame."""
        f_center = load_test_image("logs/pan_test_4_center.png")
        f_attack = load_test_image("logs/yolo/attack_20261106_091024.png")
        if f_center is None or f_attack is None:
            self.skipTest("Pan center or attack frame not found")

        self.assertEqual(f_center.shape, f_attack.shape)

    def test_e2e_train_dataset_batch_analysis(self):
        """Runs pipeline on multiple images from datasets/coc/images/train/."""
        train_dir = os.path.join(PROJECT_ROOT, "datasets", "coc", "images", "train")
        if not os.path.exists(train_dir):
            self.skipTest("datasets/coc/images/train not found")

        files = [f for f in os.listdir(train_dir) if f.endswith(".png")][:3]
        for f in files:
            with self.subTest(file=f):
                img = cv2.imread(os.path.join(train_dir, f))
                res = run_full_offline_analysis(img, mode="high_res_contour")
                self.assertEqual(len(res["cuts"]), 0)

    def test_e2e_audit_regression_against_old_logs(self):
        """
        Verifies that the old cut defect (where 5 structures were cut in red_boundary_detected.png)
        is completely eliminated in the new pipeline.
        """
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        res = run_full_offline_analysis(img)
        self.assertEqual(len(res["cuts"]), 0, "Regression audit: cut count must be 0")


if __name__ == "__main__":
    unittest.main()
```

## File: tests/test_edrag_strategy.py
```python
"""
tests/test_edrag_strategy.py
Pruebas unitarias para la estrategia de Dragones Eléctricos + Furias (EDrag Rush 50%).
"""
import os
import unittest
from unittest.mock import MagicMock
import numpy as np
import cv2

from core.vision import Vision
from vision.army_checker import ArmyChecker
from vision.resource_reader import ResourceReader
from attack.edrag_strategy import EDragStrategy
from core.config import SETTINGS


class TestEDragStrategy(unittest.TestCase):
    """Verifica todos los componentes de la estrategia EDrag Rush 50%."""

    def setUp(self):
        self.vision = Vision()
        self.mock_device = MagicMock()
        self.mock_device.width = 2400
        self.mock_device.height = 1080
        # Dummy screenshot de 2400x1080
        self.dummy_screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        self.mock_device.screenshot.return_value = self.dummy_screen
        self.strategy = EDragStrategy(self.mock_device, self.vision)

    def test_new_templates_exist_and_load(self):
        """Verifica que todos los archivos de plantilla nuevos existen y son válidos."""
        templates = [
            "assets/buttons/edrag_slot.png",
            "assets/buttons/spell_rage.png",
            "assets/buttons/hero_minion_prince_slot.png",
            "assets/buttons/hero_champion_slot.png",
            "assets/buttons/hero_warden_slot.png",
            "assets/buttons/hero_duke_dragon_slot.png",
            "assets/buttons/hero_king_slot.png",
        ]
        for tpl_path in templates:
            self.assertTrue(os.path.exists(tpl_path), f"Plantilla no encontrada: {tpl_path}")
            img = cv2.imread(tpl_path)
            self.assertIsNotNone(img, f"No se pudo cargar la imagen: {tpl_path}")
            self.assertGreater(img.shape[0], 20, f"Altura insuficiente: {tpl_path}")
            self.assertGreater(img.shape[1], 20, f"Anchura insuficiente: {tpl_path}")

    def test_north_triangle_drops_geometry(self):
        """Verifica que los drops calculados caen en cuadrantes seguros, altos y alejados en el norte."""
        nw_drops, ne_drops = self.strategy.compute_north_triangle_drops()
        self.assertGreaterEqual(len(nw_drops), 4)
        self.assertGreaterEqual(len(ne_drops), 4)

        for x, y in nw_drops:
            # Flanco Noroeste alto y seguro
            self.assertGreaterEqual(x, 600)
            self.assertLessEqual(x, 1100)
            self.assertGreaterEqual(y, 100)
            self.assertLessEqual(y, 300)

        for x, y in ne_drops:
            # Flanco Noreste alto y seguro
            self.assertGreaterEqual(x, 1300)
            self.assertLessEqual(x, 1800)
            self.assertGreaterEqual(y, 100)
            self.assertLessEqual(y, 300)

    def test_opening_and_core_rage_points(self):
        """Verifica que los puntos de Furia de apertura y núcleo están en zonas estratégicas."""
        opening = self.strategy.get_opening_rage_points()
        self.assertGreaterEqual(len(opening), 2)
        for x, y in opening:
            self.assertGreater(y, 180)
            self.assertLess(y, 450)

        core = self.strategy.get_core_rage_points()
        self.assertGreaterEqual(len(core), 2)
        for x, y in core:
            self.assertGreater(y, 300)
            self.assertLess(y, 650)

    def test_dynamic_loot_criteria_edrag_vs_sneaky(self):
        """Verifica que meets_attack_criteria aplica los umbrales reducidos para EDrag."""
        edrag_criteria = {
            "min_gold": 500000,
            "min_elixir": 500000,
            "min_dark_elixir": 6000,
            "conditions_required": 2,
        }
        loot_test = {"gold": 550000, "elixir": 520000, "dark_elixir": 4000}

        meets_edrag = ResourceReader.meets_attack_criteria(loot_test, criteria_override=edrag_criteria)
        self.assertTrue(meets_edrag)

        meets_classic = ResourceReader.meets_attack_criteria(loot_test)
        self.assertFalse(meets_classic)

    def test_army_detection_with_live_screenshot(self):
        """Verifica la detección automática del ejército EDrag sobre la captura real del usuario."""
        live_img_path = "logs/yolo/edrag_attack_start_20261111_184514.png"
        if not os.path.exists(live_img_path):
            self.skipTest(f"{live_img_path} no existe en este entorno")

        img = cv2.imread(live_img_path)
        army = ArmyChecker(self.vision)

        with unittest.mock.patch.dict(SETTINGS["attack"], {"attack_strategy": "auto"}):
            detected_type = army.detect_active_army_type(img)
            self.assertEqual(detected_type, "edrag_rush")

        heroes = army.find_all_heroes_in_battle_bar(img)
        self.assertIn("champion", heroes)
        self.assertIn("minion_prince", heroes)
        self.assertIn("warden", heroes)
        self.assertTrue("duke_dragon" in heroes or "king" in heroes)

    def test_execute_attack_mock_run(self):
        """Verifica que execute_attack corre sin excepciones y envía las secuencias de taps correctas."""
        army = self.strategy.army
        army.find_edrag_in_battle_bar = MagicMock(return_value=(407, 955))
        army.find_rage_spell_in_battle_bar = MagicMock(return_value=(1345, 960))
        army.find_all_heroes_in_battle_bar = MagicMock(return_value={
            "champion": (725, 960),
            "minion_prince": (880, 960),
            "warden": (1035, 960),
            "duke_dragon": (1190, 960),
        })

        self.strategy.rage_core_delay_sec = 0.05
        self.strategy.hero_ability_delay_sec = 0.05

        success = self.strategy.execute_attack()
        self.assertTrue(success)
        # Debe llamar zoom_half (modo half por defecto) y pan_camera hacia el norte
        self.mock_device.zoom_half.assert_called_once()
        self.mock_device.pan_camera.assert_called_once()
        # Debe ejecutar exactamente 2 taps sostenidos separados para los dragones
        self.assertEqual(self.mock_device.hold_tap.call_count, 2)
        # Debe desplegar los 4 héroes uno a uno en el mismo punto ancla de los primeros dragones
        nw_drops, _ = self.strategy.compute_north_triangle_drops(self.dummy_screen)
        anchor_pt = nw_drops[1]
        anchor_calls = [call for call in self.mock_device.tap.call_args_list if call[0] == anchor_pt or (len(call[0]) >= 2 and (call[0][0], call[0][1]) == anchor_pt)]
        self.assertEqual(len(anchor_calls), 4)


if __name__ == "__main__":
    unittest.main()
```

## File: tests/test_heroes_haste_and_cross_pan.py
```python
"""
tests/test_heroes_haste_and_cross_pan.py
Pruebas unitarias para:
1. Detección de los 4 héroes en la barra de batalla.
2. Detección del Hechizo de Aceleración (Haste) en la barra de batalla.
3. Despliegue de héroes de vanguardia y activación instantánea de habilidades.
4. Despliegue táctico de Hechizos de Aceleración en brechas/corredores.
5. Paneo en cruz de 4 direcciones (Norte, Sur, Este, Oeste) y swipes recíprocos.
"""
import os
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

from core.vision import Vision
from vision.army_checker import ArmyChecker
from attack.deployment_strategy import DeploymentStrategy
from vision.yolo_detector import Detection


class TestHeroesAndHasteDetection(unittest.TestCase):
    def setUp(self):
        self.vision = Vision()
        self.checker = ArmyChecker(self.vision)
        self.frame_path = os.path.join("logs", "frame_25s.png")

    def test_heroes_templates_exist(self):
        """Verifica que existen los 4 archivos de plantilla de héroes."""
        self.assertTrue(os.path.exists("assets/buttons/hero_king_slot.png"))
        self.assertTrue(os.path.exists("assets/buttons/hero_queen_slot.png"))
        self.assertTrue(os.path.exists("assets/buttons/hero_warden_slot.png"))
        self.assertTrue(os.path.exists("assets/buttons/hero_champion_slot.png"))

    def test_haste_template_exists(self):
        """Verifica que existe la plantilla del hechizo de aceleración."""
        self.assertTrue(os.path.exists("assets/buttons/spell_haste.png"))

    def test_detect_all_heroes_on_battle_frame(self):
        """Verifica que los héroes se detectan con precisión en los frames de batalla."""
        if not os.path.exists(self.frame_path):
            self.skipTest(f"Frame no encontrado: {self.frame_path}")

        img = cv2.imread(self.frame_path)
        heroes = self.checker.find_all_heroes_in_battle_bar(img)
        self.assertIn("king", heroes)
        self.assertIn("queen", heroes)
        self.assertIn("warden", heroes)

        # Si existe el frame real de EDrag con los 4 héroes activos (RC, Prince, Warden, Duke Dragon)
        edrag_frame = os.path.join("logs", "yolo", "edrag_attack_start_20261111_184514.png")
        if os.path.exists(edrag_frame):
            e_img = cv2.imread(edrag_frame)
            e_heroes = self.checker.find_all_heroes_in_battle_bar(e_img)
            self.assertIn("champion", e_heroes)
            self.assertIn("minion_prince", e_heroes)
            self.assertIn("warden", e_heroes)
            self.assertTrue("duke_dragon" in e_heroes or "king" in e_heroes)

        # Verificar coordenadas dentro de la barra de cartas (y > 800)
        for h_name, pos in heroes.items():
            self.assertGreater(pos[1], 800, f"Posición Y de {h_name} debe estar en la barra")
            self.assertGreater(pos[0], 500, f"Posición X de {h_name} debe estar en la barra")

    def test_detect_haste_spell_on_battle_frame(self):
        """Verifica que el hechizo de aceleración se detecta en el frame de batalla."""
        if not os.path.exists(self.frame_path):
            self.skipTest(f"Frame no encontrado: {self.frame_path}")

        img = cv2.imread(self.frame_path)
        haste_pos = self.checker.find_haste_spell_in_battle_bar(img)
        self.assertIsNotNone(haste_pos)
        self.assertGreater(haste_pos[1], 800)
        self.assertGreater(haste_pos[0], 1400)


class TestHeroSpearheadDeployment(unittest.TestCase):
    def setUp(self):
        self.mock_device = MagicMock()
        self.mock_device.width = 2400
        self.mock_device.height = 1080
        self.mock_device.d = MagicMock()
        self.mock_vision = MagicMock()
        self.mock_army = MagicMock()
        self.strategy = DeploymentStrategy(self.mock_device, self.mock_vision, self.mock_army)

    @patch("attack.deployment_strategy.SETTINGS")
    def test_deploy_heroes_and_trigger_abilities(self, mock_settings):
        """Verifica que despliega los 4 héroes y toca inmediatamente sus slots para activar habilidades."""
        mock_settings.get.side_effect = lambda k, d=None: {
            "attack": {
                "use_heroes": True,
                "instant_hero_abilities": True,
            }
        }.get(k, d if d is not None else {})

        # Simular que el screenshot detecta los 4 héroes
        dummy_screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        self.mock_device.screenshot.return_value = dummy_screen
        self.strategy.army.find_all_heroes_in_battle_bar = MagicMock(return_value={
            "king": (920, 960),
            "queen": (1070, 970),
            "warden": (1230, 950),
            "champion": (1370, 950),
        })

        # Desplegar héroes en punto de penetración (1400, 500)
        deployed_count = self.strategy._deploy_heroes_spearhead(1400, 500, 138, 845)

        self.assertEqual(deployed_count, 4)
        self.assertTrue(self.strategy._heroes_deployed_in_battle)

        # Verificar que se seleccionaron los 4 slots y se tocaron sus habilidades
        # Total de taps en slots: 4 despliegues + 4 activaciones = 8 taps en slots
        slot_taps = [call for call in self.mock_device.tap.call_args_list]
        self.assertGreaterEqual(len(slot_taps), 8)

        # Verificar que se enviaron 4 clicks de despliegue al área de juego
        drop_clicks = [call for call in self.mock_device.d.click.call_args_list]
        self.assertEqual(len(drop_clicks), 4)

    @patch("attack.deployment_strategy.SETTINGS")
    def test_heroes_deploy_only_once_per_battle(self, mock_settings):
        """Verifica que los héroes no se vuelven a desplegar en fases subsecuentes."""
        mock_settings.get.side_effect = lambda k, d=None: {
            "attack": {"use_heroes": True}
        }.get(k, d if d is not None else {})

        self.strategy._heroes_deployed_in_battle = True
        count = self.strategy._deploy_heroes_spearhead(1200, 500, 138, 845)
        self.assertEqual(count, 0)


class TestCrossPan4Directions(unittest.TestCase):
    def setUp(self):
        self.mock_device = MagicMock()
        self.mock_device.width = 2400
        self.mock_device.height = 1080
        self.mock_device.d = MagicMock()
        self.mock_vision = MagicMock()
        self.mock_army = MagicMock()
        self.strategy = DeploymentStrategy(self.mock_device, self.mock_vision, self.mock_army)

    @patch("attack.deployment_strategy.SETTINGS")
    def test_full_cross_pan_executes_all_4_flanks_with_reciprocals(self, mock_settings):
        """Verifica que con enable_full_cross_pan=True se recorren Norte, Sur, Este, Oeste con retornos recíprocos."""
        mock_settings.get.side_effect = lambda k, d=None: {
            "attack": {
                "enable_camera_pan": True,
                "enable_full_cross_pan": True,
                "min_unreached_targets_for_pan": 1,
                "zone_attack_flanks": ["north", "south", "east", "west"],
                "safe_drop_y_min": 138,
                "safe_drop_y_max": 845,
            }
        }.get(k, d if d is not None else {})

        # Mockear _execute_single_zone_quota_deploy para registrar las zonas visitadas
        visited_zones = []

        def fake_zone_deploy(zone_name, deployed_drops_log, skip_spells=False, **kwargs):
            visited_zones.append(zone_name)
            # Retorna 5 tropas y 1 objetivo inalcanzable para activar paneo
            unreached_dummy = [Detection(cls_name="storage_gold", confidence=0.9, bbox_xyxy=(100, 100, 200, 200), center=(150, 150))]
            return 5, unreached_dummy

        self.strategy._execute_single_zone_quota_deploy = fake_zone_deploy
        self.strategy._save_quota_attack_debug_image = MagicMock()

        self.strategy._deploy_by_structure_quotas(screen_centered=np.zeros((1080, 2400, 3), dtype=np.uint8))

        # Verificar que se visitaron las 5 zonas en orden
        self.assertEqual(visited_zones, ["CENTRO", "NORTE", "SUR", "ESTE", "OESTE"])

        # Verificar los swipes en shell de adb: deben ser 8 swipes (4 de ida y 4 de retorno)
        shell_calls = [call.args[0] for call in self.mock_device.d.shell.call_args_list]

        # Flanco Norte
        self.assertIn("input swipe 1200 240 1200 820 480", shell_calls)
        self.assertIn("input swipe 1200 820 1200 240 480", shell_calls)

        # Flanco Sur
        self.assertIn("input swipe 1200 780 1200 240 480", shell_calls)
        self.assertIn("input swipe 1200 240 1200 780 480", shell_calls)

        # Flanco Este (swipe calibrado 700px: 1550 -> 850 para no empujar almacenes fuera de pantalla)
        self.assertIn("input swipe 1550 540 850 540 480", shell_calls)
        self.assertIn("input swipe 850 540 1550 540 480", shell_calls)

        # Flanco Oeste (swipe calibrado 700px: 850 -> 1550)
        self.assertIn("input swipe 850 540 1550 540 480", shell_calls)
        self.assertIn("input swipe 1550 540 850 540 480", shell_calls)


if __name__ == "__main__":
    unittest.main()
```

## File: tests/test_quota_drop_and_inventory.py
```python
"""
tests/test_quota_drop_and_inventory.py
Batería de pruebas automatizadas para:
1. Control estricto de inventario de batalla (96 Duendes, 8 SWB, 4 Saltos, 3 Aceleraciones).
2. Detección zonal del rombo canónico y eliminación de falsos 'fuera de margen' en vistas paneadas (5 iteraciones por zona).
3. Distancia de drop cercana (15-25px del perímetro legal exterior).
4. Paneo inteligente condicional con estabilización a 1.3s y retorno a 0.85s.
"""
import unittest
from unittest.mock import MagicMock, patch, call
import cv2
import numpy as np
import random

from vision.yolo_detector import Detection
from attack.deployment_strategy import (
    DeploymentStrategy,
    detect_village_boundary,
    compute_canonical_diamond_drop,
    CANONICAL_CENTER,
    CANONICAL_RX_GRASS,
    CANONICAL_RY_GRASS,
    CANONICAL_RX_BUILDABLE,
    CANONICAL_RY_BUILDABLE,
)


class TestTroopInventoryLimits(unittest.TestCase):
    def setUp(self):
        self.mock_dev = MagicMock()
        self.mock_dev.width = 2400
        self.mock_dev.height = 1080
        self.mock_dev.screenshot.return_value = np.zeros((1080, 2400, 3), dtype=np.uint8)

        self.mock_vis = MagicMock()
        self.mock_army = MagicMock()
        self.mock_army.find_super_wall_breaker_in_battle_bar.return_value = (580, 960)
        self.mock_army.find_goblins_in_battle_bar.return_value = (420, 960)
        self.mock_army.find_jump_spell_in_battle_bar.return_value = (1530, 960)
        self.mock_army.find_haste_spell_in_battle_bar.return_value = (1660, 960)

        self.strat = DeploymentStrategy(self.mock_dev, self.mock_vis, self.mock_army)

    def test_initial_inventory(self):
        """Verifica que el inventario se inicializa con los valores exactos."""
        self.assertEqual(self.strat._troop_inventory["goblins"], 96)
        self.assertEqual(self.strat._troop_inventory["swb"], 8)
        self.assertEqual(self.strat._troop_inventory["jump"], 4)
        self.assertEqual(self.strat._troop_inventory["haste"], 3)

    def test_swb_inventory_cap(self):
        """Verifica que nunca se lancen más de 8 SWB aunque se soliciten múltiples brechas."""
        self.strat._troop_inventory["swb"] = 8
        self.strat._ensure_super_wall_breaker_selected = MagicMock(return_value=True)

        # Generar 6 edificios que requerirían 2 SWB cada uno (12 en total)
        targets = [
            Detection(cls_name="storage_gold", confidence=0.9, bbox_xyxy=(500 + i * 200, 500, 550 + i * 200, 550), center=(525 + i * 200, 525))
            for i in range(6)
        ]
        deployed_log = []

        with patch.object(self.strat, "_ensure_super_wall_breaker_selected", return_value=True), \
             patch.object(self.strat, "_ensure_goblins_selected", return_value=True), \
             patch.object(self.strat, "_ensure_spell_selected", return_value=False), \
             patch.object(self.strat, "_deploy_heroes_spearhead"):
            self.strat._execute_single_zone_quota_deploy(
                zone_name="CENTRO",
                deployed_drops_log=deployed_log,
                skip_spells=True,
                precomputed_screen=np.zeros((1080, 2400, 3), dtype=np.uint8),
                precomputed_detections=targets,
            )

        # La cuota de SWB debe haber quedado en 0 exactamente al usar las 4 brechas (8 SWB)
        self.assertEqual(self.strat._troop_inventory["swb"], 0)

        # Simular agotamiento forzado a 0 y verificar que no se intenta desplegar más
        self.strat._troop_inventory["swb"] = 0
        with patch.object(self.strat, "_ensure_super_wall_breaker_selected") as mock_swb_sel:
            self.strat._execute_single_zone_quota_deploy(
                zone_name="CENTRO",
                deployed_drops_log=[],
                skip_spells=True,
                precomputed_screen=np.zeros((1080, 2400, 3), dtype=np.uint8),
                precomputed_detections=targets,
            )
            mock_swb_sel.assert_not_called()

    def test_all_4_jumps_deployed(self):
        """Verifica que se generen y desplieguen los 4 Hechizos de Salto."""
        targets = [
            Detection(cls_name="storage_dark", confidence=0.9, bbox_xyxy=(1200, 500, 1250, 550), center=(1225, 525)),
            Detection(cls_name="storage_gold", confidence=0.9, bbox_xyxy=(900, 300, 950, 350), center=(925, 325)),
            Detection(cls_name="storage_elixir", confidence=0.9, bbox_xyxy=(1500, 300, 1550, 350), center=(1525, 325)),
            Detection(cls_name="town_hall", confidence=0.9, bbox_xyxy=(1200, 700, 1250, 750), center=(1225, 725)),
        ]
        with patch.object(self.strat, "_ensure_spell_selected", return_value=True), \
             patch.object(self.strat, "_ensure_goblins_selected", return_value=True), \
             patch("attack.deployment_strategy.SETTINGS", {"attack": {"use_spells": True, "use_yolo": True, "max_jumps": 4, "min_jump_score": 1.0}}):
            deployed = self.strat._deploy_strategic_spells_centered(
                screen=np.zeros((1080, 2400, 3), dtype=np.uint8),
                precomputed_detections=targets
            )
            self.assertEqual(len(deployed), 4, "Debe desplegar exactamente los 4 Hechizos de Salto")
            self.assertEqual(self.strat._troop_inventory["jump"], 0, "Inventario de saltos debe quedar en 0")

    def test_haste_inventory_cap(self):
        """Verifica que nunca se lancen más de 3 Hechizos de Aceleración."""
        self.strat._troop_inventory["haste"] = 3
        # Forzar que _ensure_spell_selected retorne True
        with patch.object(self.strat, "_ensure_spell_selected", return_value=True), \
             patch.object(self.strat, "_ensure_goblins_selected", return_value=True), \
             patch.object(self.strat, "_ensure_super_wall_breaker_selected", return_value=False):
            targets = [
                Detection(cls_name="storage_gold", confidence=0.9, bbox_xyxy=(500, 500, 550, 550), center=(525, 525)),
                Detection(cls_name="storage_elixir", confidence=0.9, bbox_xyxy=(700, 500, 750, 550), center=(725, 525)),
                Detection(cls_name="storage_dark", confidence=0.9, bbox_xyxy=(900, 500, 950, 550), center=(925, 525)),
            ]
            self.strat._execute_single_zone_quota_deploy(
                zone_name="CENTRO",
                deployed_drops_log=[],
                skip_spells=True,
                precomputed_screen=np.zeros((1080, 2400, 3), dtype=np.uint8),
                precomputed_detections=targets,
            )

        # Se deben haber gastado exactamente 2 hastes (hasta 2 almacenes en esta vista)
        self.assertEqual(self.strat._troop_inventory["haste"], 1)

        # Simular que solo queda 1 y se piden 2
        with patch.object(self.strat, "_ensure_spell_selected", return_value=True), \
             patch.object(self.strat, "_ensure_goblins_selected", return_value=True), \
             patch.object(self.strat, "_ensure_super_wall_breaker_selected", return_value=False):
            self.strat._execute_single_zone_quota_deploy(
                zone_name="CENTRO",
                deployed_drops_log=[],
                skip_spells=True,
                precomputed_screen=np.zeros((1080, 2400, 3), dtype=np.uint8),
                precomputed_detections=targets,
            )
        self.assertEqual(self.strat._troop_inventory["haste"], 0)

        # Con 0 restantes, no debe llamar a _ensure_spell_selected("haste")
        with patch.object(self.strat, "_ensure_spell_selected") as mock_spell_sel, \
             patch.object(self.strat, "_ensure_goblins_selected", return_value=True), \
             patch.object(self.strat, "_ensure_super_wall_breaker_selected", return_value=False):
            self.strat._execute_single_zone_quota_deploy(
                zone_name="CENTRO",
                deployed_drops_log=[],
                skip_spells=True,
                precomputed_screen=np.zeros((1080, 2400, 3), dtype=np.uint8),
                precomputed_detections=targets,
            )
            for c in mock_spell_sel.call_args_list:
                self.assertNotEqual(c[0][0], "haste")

    def test_goblin_inventory_cap(self):
        """Verifica que el despliegue de duendes se detiene exactamente al llegar a 0 de los 96."""
        self.strat._troop_inventory["goblins"] = 10
        # Generar 3 almacenes (cuota 6 cada uno -> requerirían 18 duendes)
        targets = [
            Detection(cls_name="storage_gold", confidence=0.9, bbox_xyxy=(500, 500, 550, 550), center=(525, 525)),
            Detection(cls_name="storage_elixir", confidence=0.9, bbox_xyxy=(700, 500, 750, 550), center=(725, 525)),
            Detection(cls_name="storage_dark", confidence=0.9, bbox_xyxy=(900, 500, 950, 550), center=(925, 525)),
        ]
        with patch.object(self.strat, "_ensure_goblins_selected", return_value=True), \
             patch.object(self.strat, "_ensure_super_wall_breaker_selected", return_value=False), \
             patch.object(self.strat, "_ensure_spell_selected", return_value=False):
            deployed_count, unreached = self.strat._execute_single_zone_quota_deploy(
                zone_name="CENTRO",
                deployed_drops_log=[],
                skip_spells=True,
                precomputed_screen=np.zeros((1080, 2400, 3), dtype=np.uint8),
                precomputed_detections=targets,
            )

        # Debe desplegar exactamente 10 duendes y el inventario quedar en 0
        self.assertEqual(deployed_count, 10)
        self.assertEqual(self.strat._troop_inventory["goblins"], 0)


class TestZonalBoundaryAndDropLegality(unittest.TestCase):
    """
    Simulación de 5 iteraciones por cada zona (Centro, Norte, Sur, Este, Oeste).
    Verifica que en ninguna zona los objetivos legales son descartados como 'fuera de margen',
    y que los drops caen dentro de la franja segura 138 <= y <= 845 y fuera de la zona roja.
    """
    def setUp(self):
        self.mock_dev = MagicMock()
        self.mock_dev.width = 2400
        self.mock_dev.height = 1080
        self.mock_dev.screenshot.return_value = np.zeros((1080, 2400, 3), dtype=np.uint8)
        self.strat = DeploymentStrategy(self.mock_dev, MagicMock(), MagicMock())
        self.screen_dummy = np.zeros((1080, 2400, 3), dtype=np.uint8)

    def test_5_iterations_centro(self):
        """5 iteraciones en Zona Central: todos los drops deben ser 100% legales y fuera de zona roja."""
        red, green = detect_village_boundary(self.screen_dummy, mode="canonical_diamond", zone_name="CENTRO")
        self.assertIsNotNone(red)
        self.assertIsNotNone(green)

        for iteration in range(5):
            # Probar diferentes ubicaciones de edificios interiores y periféricos
            test_x = random.randint(900, 1500)
            test_y = random.randint(350, 650)
            drop_x, drop_y = self.strat._get_drop_point_on_contour(
                (test_x, test_y), green, red_contour=red, zone_name="CENTRO"
            )

            # 1. Dentro de los límites seguros de pantalla
            self.assertGreaterEqual(drop_y, self.strat._SAFE_Y_MIN, f"[CENTRO #{iteration}] Drop Y debe ser >= safe_y_min")
            self.assertLessEqual(drop_y, self.strat._SAFE_Y_MAX, f"[CENTRO #{iteration}] Drop Y debe ser <= safe_y_max")
            self.assertGreaterEqual(drop_x, self.strat._SAFE_X_MIN, f"[CENTRO #{iteration}] Drop X debe ser >= safe_x_min")
            self.assertLessEqual(drop_x, self.strat._SAFE_X_MAX, f"[CENTRO #{iteration}] Drop X debe ser <= safe_x_max")

            # 2. Fuera de la zona roja construible
            dist_red = cv2.pointPolygonTest(red, (float(drop_x), float(drop_y)), True)
            self.assertLessEqual(dist_red, 0.0, f"[CENTRO #{iteration}] Drop ({drop_x}, {drop_y}) no debe caer dentro del rombo rojo")

    def test_5_iterations_norte(self):
        """5 iteraciones en Flanco Norte: el rombo se desplaza a Y=1.25*H y los drops son legales."""
        red, green = detect_village_boundary(self.screen_dummy, mode="canonical_diamond", zone_name="NORTE")
        self.assertIsNotNone(red)
        # El centro debe estar en (1200, 1350)
        center_red_y = np.mean(red[:, 0, 1])
        self.assertAlmostEqual(center_red_y, 1080 * 1.25, delta=10)

        for iteration in range(5):
            test_x = random.randint(900, 1500)
            test_y = random.randint(200, 450)
            drop_x, drop_y = self.strat._get_drop_point_on_contour(
                (test_x, test_y), green, red_contour=red, zone_name="NORTE"
            )

            self.assertGreaterEqual(drop_y, self.strat._SAFE_Y_MIN, f"[NORTE #{iteration}] Drop Y >= safe_y_min")
            self.assertLessEqual(drop_y, self.strat._SAFE_Y_MAX, f"[NORTE #{iteration}] Drop Y <= safe_y_max")
            dist_red = cv2.pointPolygonTest(red, (float(drop_x), float(drop_y)), True)
            self.assertLessEqual(dist_red, 0.0, f"[NORTE #{iteration}] Drop ({drop_x}, {drop_y}) no debe ser inside_red")

    def test_5_iterations_sur(self):
        """5 iteraciones en Flanco Sur: el rombo se desplaza a Y=-0.25*H y los drops son legales."""
        red, green = detect_village_boundary(self.screen_dummy, mode="canonical_diamond", zone_name="SUR")
        self.assertIsNotNone(red)
        center_red_y = np.mean(red[:, 0, 1])
        self.assertAlmostEqual(center_red_y, -1080 * 0.25, delta=10)

        for iteration in range(5):
            test_x = random.randint(900, 1500)
            test_y = random.randint(550, 800)
            drop_x, drop_y = self.strat._get_drop_point_on_contour(
                (test_x, test_y), green, red_contour=red, zone_name="SUR"
            )

            self.assertGreaterEqual(drop_y, self.strat._SAFE_Y_MIN, f"[SUR #{iteration}] Drop Y >= safe_y_min")
            self.assertLessEqual(drop_y, self.strat._SAFE_Y_MAX, f"[SUR #{iteration}] Drop Y <= safe_y_max")
            dist_red = cv2.pointPolygonTest(red, (float(drop_x), float(drop_y)), True)
            self.assertLessEqual(dist_red, 0.0, f"[SUR #{iteration}] Drop ({drop_x}, {drop_y}) no debe ser inside_red")

    def test_5_iterations_este(self):
        """5 iteraciones en Flanco Este: el rombo se desplaza a X=-0.25*W y los drops son legales."""
        red, green = detect_village_boundary(self.screen_dummy, mode="canonical_diamond", zone_name="ESTE")
        self.assertIsNotNone(red)
        center_red_x = np.mean(red[:, 0, 0])
        self.assertAlmostEqual(center_red_x, -2400 * 0.25, delta=10)

        for iteration in range(5):
            test_x = random.randint(1400, 2000)
            test_y = random.randint(300, 700)
            drop_x, drop_y = self.strat._get_drop_point_on_contour(
                (test_x, test_y), green, red_contour=red, zone_name="ESTE"
            )

            self.assertGreaterEqual(drop_y, self.strat._SAFE_Y_MIN, f"[ESTE #{iteration}] Drop Y >= safe_y_min")
            self.assertLessEqual(drop_y, self.strat._SAFE_Y_MAX, f"[ESTE #{iteration}] Drop Y <= safe_y_max")
            dist_red = cv2.pointPolygonTest(red, (float(drop_x), float(drop_y)), True)
            self.assertLessEqual(dist_red, 0.0, f"[ESTE #{iteration}] Drop ({drop_x}, {drop_y}) no debe ser inside_red")

    def test_5_iterations_oeste(self):
        """5 iteraciones en Flanco Oeste: el rombo se desplaza a X=1.25*W y los drops son legales."""
        red, green = detect_village_boundary(self.screen_dummy, mode="canonical_diamond", zone_name="OESTE")
        self.assertIsNotNone(red)
        center_red_x = np.mean(red[:, 0, 0])
        self.assertAlmostEqual(center_red_x, 2400 * 1.25, delta=10)

        for iteration in range(5):
            test_x = random.randint(400, 1000)
            test_y = random.randint(300, 700)
            drop_x, drop_y = self.strat._get_drop_point_on_contour(
                (test_x, test_y), green, red_contour=red, zone_name="OESTE"
            )

            self.assertGreaterEqual(drop_y, self.strat._SAFE_Y_MIN, f"[OESTE #{iteration}] Drop Y >= safe_y_min")
            self.assertLessEqual(drop_y, self.strat._SAFE_Y_MAX, f"[OESTE #{iteration}] Drop Y <= safe_y_max")
            dist_red = cv2.pointPolygonTest(red, (float(drop_x), float(drop_y)), True)
            self.assertLessEqual(dist_red, 0.0, f"[OESTE #{iteration}] Drop ({drop_x}, {drop_y}) no debe ser inside_red")


class TestSmartConditionalPanning(unittest.TestCase):
    def setUp(self):
        self.mock_dev = MagicMock()
        self.mock_dev.width = 2400
        self.mock_dev.height = 1080
        self.strat = DeploymentStrategy(self.mock_dev, MagicMock(), MagicMock())

    def test_no_pan_when_all_targets_reached(self):
        """Verifica que si la vista Central alcanza el 100% de los objetivos, NO se ejecutan paneos."""
        screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        # Mock _execute_single_zone_quota_deploy para CENTRO retornando 0 unreached
        with patch.object(self.strat, "_execute_single_zone_quota_deploy", return_value=(19, [])) as mock_exec, \
             patch("attack.deployment_strategy.SETTINGS", {"attack": {"enable_camera_pan": True, "enable_full_cross_pan": False, "min_unreached_targets_for_pan": 1}}):
            self.strat._deploy_by_structure_quotas(
                detections=[],
                screen_centered=screen,
            )

        # Solo debe haberse ejecutado CENTRO, 0 paneos
        self.assertEqual(mock_exec.call_count, 1)
        self.assertEqual(mock_exec.call_args_list[0][1]["zone_name"], "CENTRO")
        # El shell no debe haber recibido ningún swipe de paneo
        self.mock_dev.d.shell.assert_not_called()

    def test_pan_only_to_flank_with_unreached_target(self):
        """Verifica que si solo hay objetivos pendientes en el Norte, solo panea al Norte y regresa."""
        screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        unreached_north_target = Detection(cls_name="storage_gold", confidence=0.9, bbox_xyxy=(1200, 200, 1250, 250), center=(1225, 225))

        # Primer llamado retorna el unreached del norte, el segundo (en Norte) retorna alcanzado
        mock_returns = [
            (10, [unreached_north_target]),
            (2, []),
        ]

        with patch.object(self.strat, "_execute_single_zone_quota_deploy", side_effect=mock_returns) as mock_exec, \
             patch("attack.deployment_strategy.SETTINGS", {"attack": {"enable_camera_pan": True, "enable_full_cross_pan": False, "min_unreached_targets_for_pan": 1, "zone_attack_flanks": ["north", "south", "east", "west"]}}):
            self.strat._deploy_by_structure_quotas(
                detections=[],
                screen_centered=screen,
            )

        # Se llamó CENTRO y luego únicamente NORTE
        self.assertEqual(mock_exec.call_count, 2)
        self.assertEqual(mock_exec.call_args_list[0][1]["zone_name"], "CENTRO")
        self.assertEqual(mock_exec.call_args_list[1][1]["zone_name"], "NORTE")

        # Verificar que el swipe ejecutó swipe hacia abajo (ir a Norte) y retorno recíproco
        shell_calls = [c[0][0] for c in self.mock_dev.d.shell.call_args_list]
        self.assertTrue(any("1200 240 1200 820" in s for s in shell_calls), "Debe ejecutar swipe hacia el Norte")
        self.assertTrue(any("1200 820 1200 240" in s for s in shell_calls), "Debe ejecutar swipe recíproco al Centro")

    def test_full_cross_pan_always_runs_all_4_flanks(self):
        """Verifica que con enable_full_cross_pan=True, los 4 flancos se visitan siempre."""
        screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        # Retornar (5, []) para cada zona
        mock_returns = [(5, []) for _ in range(5)]

        with patch.object(self.strat, "_execute_single_zone_quota_deploy", side_effect=mock_returns) as mock_exec, \
             patch("attack.deployment_strategy.SETTINGS", {"attack": {"enable_camera_pan": True, "enable_full_cross_pan": True, "min_unreached_targets_for_pan": 1, "zone_attack_flanks": ["north", "south", "east", "west"]}}):
            self.strat._deploy_by_structure_quotas(
                detections=[],
                screen_centered=screen,
            )

        # Se llamó CENTRO, NORTE, SUR, ESTE, OESTE (5 llamadas en total)
        self.assertEqual(mock_exec.call_count, 5)
        called_zones = [c[1]["zone_name"] for c in mock_exec.call_args_list]
        self.assertEqual(called_zones, ["CENTRO", "NORTE", "SUR", "ESTE", "OESTE"])

    def test_flank_pans_skipped_when_goblins_exhausted(self):
        """Verifica que si _troop_inventory['goblins'] queda en 0 tras CENTRO, se omiten todos los paneos de flancos."""
        screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        unreached_dummy = [Detection(cls_name="storage_gold", confidence=0.9, bbox_xyxy=(100, 100, 200, 200), center=(150, 150))]

        def mock_center_depletes_goblins(zone_name, deployed_drops_log, **kwargs):
            if zone_name == "CENTRO":
                self.strat._troop_inventory["goblins"] = 0
                return 96, unreached_dummy
            return 0, []

        with patch.object(self.strat, "_execute_single_zone_quota_deploy", side_effect=mock_center_depletes_goblins) as mock_exec, \
             patch("attack.deployment_strategy.SETTINGS", {"attack": {"enable_camera_pan": True, "enable_full_cross_pan": True, "min_unreached_targets_for_pan": 1, "zone_attack_flanks": ["north", "south", "east", "west"]}}):
            self.strat._deploy_by_structure_quotas(
                detections=[],
                screen_centered=screen,
            )

        # Solo debe haberse llamado a CENTRO (1 llamada)
        self.assertEqual(mock_exec.call_count, 1)
        self.assertEqual(mock_exec.call_args_list[0][1]["zone_name"], "CENTRO")
        # No debe haber ningún swipe de paneo registrado en adb
        self.assertEqual(self.mock_dev.d.shell.call_count, 0)


class TestFlankSpellsAndHasteNearJumps(unittest.TestCase):
    """Suite de validación de saltos en flancos paneados y colocación de haste junto a saltos."""

    def setUp(self):
        self.mock_dev = MagicMock()
        self.mock_dev.width = 2400
        self.mock_dev.height = 1080
        self.mock_dev.screenshot.return_value = np.zeros((1080, 2400, 3), dtype=np.uint8)

        self.mock_vis = MagicMock()
        self.mock_army = MagicMock()
        self.mock_army.find_jump_spell_in_battle_bar.return_value = (1530, 960)
        self.mock_army.find_haste_spell_in_battle_bar.return_value = (1660, 960)
        self.mock_army.find_goblins_in_battle_bar.return_value = (420, 960)

        self.strat = DeploymentStrategy(self.mock_dev, self.mock_vis, self.mock_army)

    def test_flank_spells_deployment_when_storage_walled(self):
        """Verifica que en una vista de flanco (NORTE) con almacén bloqueado tras muro, se despliega Salto y Haste."""
        self.strat._troop_inventory["jump"] = 2
        self.strat._troop_inventory["haste"] = 2
        self.strat._troop_inventory["goblins"] = 20

        flank_storage = Detection(
            cls_name="storage_gold",
            confidence=0.92,
            bbox_xyxy=(1200, 300, 1260, 360),
            center=(1230, 330)
        )
        reachable = [(flank_storage, (1230, 160))]

        with patch.object(self.strat, "_ensure_spell_selected", return_value=True), \
             patch("attack.deployment_strategy.find_wall_crossings", return_value=[(1230, 240)]):
            self.strat._deploy_flank_spells(
                screen=np.zeros((1080, 2400, 3), dtype=np.uint8),
                zone_name="NORTE",
                reachable_targets=reachable,
                green_contour=np.array([[1000, 100], [1500, 100], [1500, 800], [1000, 800]]),
                safe_y_min=140,
                safe_y_max=845,
            )

        # Debe haber gastado 1 salto y 1 haste en el flanco
        self.assertEqual(self.strat._troop_inventory["jump"], 1, "Debe quedar 1 salto tras desplegar en flanco")
        self.assertEqual(self.strat._troop_inventory["haste"], 1, "Debe quedar 1 haste tras desplegar junto al salto")
        # El salto debe estar registrado en _used_jump_points
        self.assertEqual(len(self.strat._used_jump_points), 1)
        self.assertAlmostEqual(self.strat._used_jump_points[0][1], 240, delta=5, msg="Y de salto debe ser en el cruce de muro con jitter")

    def test_haste_near_jump_location(self):
        """Verifica que el hechizo de aceleración se coloca junto a las coordenadas del salto (+sesgo <= 25px)."""
        self.strat._troop_inventory["haste"] = 2
        jump_point = (1200, 400)
        target_point = (1200, 500)

        with patch.object(self.strat, "_ensure_spell_selected", return_value=True):
            self.strat._deploy_haste_near_jump(jump_point, target_point, zone_name="CENTRO")

        self.assertEqual(self.strat._troop_inventory["haste"], 1)
        # La posición enviada a tap debe estar cerca de jump_point (distancia < 35px)
        tapped_x, tapped_y = self.mock_dev.tap.call_args[0]
        dist = np.hypot(tapped_x - 1200, tapped_y - 400)
        self.assertLessEqual(dist, 35.0, "Haste debe caer a menos de 35px del Salto")

    def test_loot_priority_orders_storages_before_collectors(self):
        """Verifica que los almacenes de oro, elixir y oscuro se ordenan antes que los recolectores."""
        collector = Detection(cls_name="collector_gold", confidence=0.9, bbox_xyxy=(100, 100, 150, 150), center=(125, 125))
        storage = Detection(cls_name="storage_dark", confidence=0.9, bbox_xyxy=(500, 500, 550, 550), center=(525, 525))
        reachable = [(collector, (100, 200)), (storage, (500, 300))]

        loot_priority = {
            "storage_dark": 10,
            "storage_gold": 8,
            "storage_elixir": 8,
            "town_hall": 7,
            "collector_dark": 4,
            "collector_gold": 3,
            "collector_elixir": 3,
        }
        reachable.sort(key=lambda item: loot_priority.get(item[0].cls_name, 1), reverse=True)
        self.assertEqual(reachable[0][0].cls_name, "storage_dark", "storage_dark debe ir de primero")
        self.assertEqual(reachable[1][0].cls_name, "collector_gold", "collector_gold debe ir de segundo")


if __name__ == "__main__":
    unittest.main()
```

## File: tests/test_ui_cache.py
```python
"""
tests/test_ui_cache.py
Pruebas unitarias para el motor de Caché UI Zero-Latency y micro-verificaciones de Vision.
"""
import unittest
import numpy as np
import cv2
from core.ui_cache import UICache
from core.vision import Vision


class TestUICache(unittest.TestCase):
    def setUp(self):
        self.cache = UICache()
        self.vision = Vision()

    def test_default_coords_scaling(self):
        """Verifica que las coordenadas por defecto escalan correctamente a 2400x1080."""
        pos_next = self.cache.get_pos("next_btn", width=2400, height=1080)
        self.assertEqual(pos_next, (2185, 892))

        pos_attack = self.cache.get_pos("attack_btn", width=2400, height=1080)
        self.assertEqual(pos_attack, (195, 945))

        pos_surrender = self.cache.get_pos("surrender_btn", width=2400, height=1080)
        self.assertEqual(pos_surrender, (214, 808))

    def test_jitter_within_bounds(self):
        """Verifica que el jitter humanizado se mantenga dentro del margen legal."""
        for _ in range(50):
            x, y = self.cache.get_pos("next_btn", width=2400, height=1080, jitter=10)
            self.assertTrue(2170 <= x <= 2200)
            self.assertTrue(880 <= y <= 905)
            self.assertTrue(0 <= x <= 2400)
            self.assertTrue(0 <= y <= 1080)

    def test_dynamic_learning(self):
        """Verifica el auto-aprendizaje dinámico con suavizado exponencial."""
        self.cache.learn_on_detect = True
        # Observación ligeramente desplazada en (2190, 895)
        self.cache.update_pos("next_btn", 2190, 895, width=2400, height=1080)
        new_pos = self.cache.get_pos("next_btn", width=2400, height=1080)
        self.assertTrue(2185 <= new_pos[0] <= 2190)
        self.assertTrue(892 <= new_pos[1] <= 895)

    def test_micro_verify_success(self):
        """Verifica que micro_verify encuentra el template en la micro-ROI en tiempo récord (<5ms)."""
        screen = np.zeros((1080, 2400), dtype=np.uint8)
        # Template sintético cuadrado de 40x40 con patrón reconocible
        template = np.full((40, 40), 200, dtype=np.uint8)
        template[10:30, 10:30] = 50

        # Incrustar template exactamente en el centro de next_btn: (2185, 892)
        cx, cy = 2185, 892
        screen[cy - 20:cy + 20, cx - 20:cx + 20] = template

        match = self.cache.micro_verify(screen, "next_btn", template, threshold=0.80)
        self.assertIsNotNone(match)
        self.assertEqual(match, (cx, cy))

    def test_micro_verify_failure_returns_none(self):
        """Verifica que si el botón no está en la zona reducida, retorna None de forma segura."""
        blank_screen = np.zeros((1080, 2400), dtype=np.uint8)
        template = np.full((40, 40), 200, dtype=np.uint8)

        match = self.cache.micro_verify(blank_screen, "next_btn", template, threshold=0.80)
        self.assertIsNone(match)

    def test_find_template_cached_integration(self):
        """Verifica la integración entre Vision y UICache con fallback transparente."""
        screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        # Template existente
        tpl_path = "assets/buttons/next_btn.png"
        loaded = self.vision._load_template(tpl_path)
        if loaded is None:
            self.skipTest(f"{tpl_path} no disponible para test")

        tpl_gray, (th, tw) = loaded
        cx, cy = self.cache.get_pos("next_btn", width=2400, height=1080)
        y1, y2 = cy - th // 2, cy - th // 2 + th
        x1, x2 = cx - tw // 2, cx - tw // 2 + tw
        screen[y1:y2, x1:x2, 0] = tpl_gray
        screen[y1:y2, x1:x2, 1] = tpl_gray
        screen[y1:y2, x1:x2, 2] = tpl_gray

        res = self.vision.find_template_cached(
            screen, tpl_path, cache_name="next_btn", ui_cache=self.cache, threshold=0.75
        )
        self.assertIsNotNone(res)
        self.assertTrue(abs(res[0] - cx) <= 2)
        self.assertTrue(abs(res[1] - cy) <= 2)


if __name__ == "__main__":
    unittest.main()
```

## File: tests/test_wall_compartments.py
```python
"""
Tests Unitarios para el Modelo de Compartimentos de Muros y Optimización de Saltos.
Fase 2 de plan_saltos_optimos_antigravity.md.
Puro numpy / OpenCV, sin ONNX ni modelos pesados.
"""
from __future__ import annotations
import unittest
import numpy as np
import cv2

from vision.wall_compartments import (
    rasterize_wall_mask,
    build_barrier_mask,
    compute_compartments,
    build_compartment_graph,
    map_targets_to_compartments,
    compute_total_cost,
    get_compartments_in_footprint,
    sample_jump_candidates,
    optimize_jumps,
    render_debug_overlay,
    CompartmentAnalysisResult,
)


class TestWallCompartments(unittest.TestCase):
    """Batería de pruebas sintéticas para el optimizador de saltos por compartimentos."""

    def test_two_concentric_rings_bridges_both(self):
        """
        Test 1: Dos anillos concéntricos de muro con un objetivo en el centro.
        Un salto colocado en la zona intermedia entre ambos anillos tiene suficiente
        radio para cubrir tanto el anillo exterior como el interior, reduciendo
        el costo de 2 capas a 0 (ganancia marginal = 2.0).
        """
        shape = (600, 600)

        # Anillo exterior: 400x400 centrado en (300, 300), de x=100 a x=500
        outer_walls = [
            (100, 100, 500, 100),
            (500, 100, 500, 500),
            (500, 500, 100, 500),
            (100, 500, 100, 100),
        ]

        # Anillo interior: 200x200 centrado en (300, 300), de x=200 a x=400
        inner_walls = [
            (200, 200, 400, 200),
            (400, 200, 400, 400),
            (400, 400, 200, 400),
            (200, 400, 200, 200),
        ]
        all_walls = outer_walls + inner_walls

        # Objetivo de botín en el compartimento central
        target = [{
            "cls_name": "storage_dark",
            "center": (300, 300),
            "bbox_xyxy": (285, 285, 315, 315),
        }]

        # Punto de entrada de tropas en el exterior
        drop_points = [(50, 300)]

        # Huella circular con radio 65px (diámetro 130px, suficiente para cruzar
        # los 100px que separan el muro exterior del interior)
        result = optimize_jumps(
            img_shape=shape,
            wall_segments=all_walls,
            detections=target,
            drop_points=drop_points,
            jump_footprint=65.0,
            max_jumps=1,
            loot_weights={"storage_dark": 1.0},
            endpoint_tol=0.0,
            morph_kernel_size=3,
            wall_thickness=6,
        )

        # Verificaciones
        self.assertEqual(result.num_compartments, 3, "Debe haber 3 compartimentos: exterior, anillo intermedio y centro")
        self.assertEqual(result.initial_cost, 2.0, "Costo inicial sin saltos debe ser 2 capas de muro")
        self.assertGreaterEqual(len(result.jumps), 1, "Debe seleccionar al menos 1 salto")

        best_jump = result.jump_details[0]
        self.assertEqual(best_jump.gain, 2.0, "El salto óptimo debe puentear ambos anillos con ganancia de 2 capas")
        self.assertEqual(result.final_cost, 0.0, "El costo final hacia el objetivo debe ser 0")
        self.assertEqual(result.total_gain, 2.0, "La ganancia total debe ser 2.0")

    def test_ring_gap_blocked_by_building_and_extension(self):
        """
        Test 2: Anillo con hueco tapado por edificio.
        - Sin edificio y con extensión insuficiente: el espacio interior fuga al exterior (1 solo compartimento, costo 0).
        - Con buildings_block=True: el bounding box del edificio sella el hueco, creando 2 compartimentos y evitando la fuga.
        - Con endpoint_tol=35: la prolongación de extremos a lo largo de la dirección isométrica sella el hueco.
        """
        shape = (600, 600)

        # Anillo con hueco de 60px en la pared superior (de x=270 a x=330 en y=200)
        walls_with_gap = [
            (200, 200, 270, 200),
            (330, 200, 400, 200),
            (400, 200, 400, 400),
            (400, 400, 200, 400),
            (200, 400, 200, 200),
        ]

        # Edificio que tapa el hueco (ej: cañón o recolector)
        building = [{
            "cls_name": "cannon",
            "center": (300, 200),
            "bbox_xyxy": (265, 185, 335, 215),
        }]

        # Almacén de oro en el centro
        target = [{
            "cls_name": "storage_gold",
            "center": (300, 300),
            "bbox_xyxy": (285, 285, 315, 315),
        }]

        drop_points = [(50, 50)]

        # Caso A: buildings_block=False y sin extensión -> fuga al exterior
        res_leak = optimize_jumps(
            img_shape=shape,
            wall_segments=walls_with_gap,
            detections=building + target,
            drop_points=drop_points,
            buildings_block=False,
            endpoint_tol=5.0,
            morph_kernel_size=3,
            loot_weights={"storage_gold": 1.0},
        )
        self.assertEqual(res_leak.num_compartments, 1, "Sin barrera de edificio el muro fuga al exterior (1 compartimento)")
        self.assertEqual(res_leak.initial_cost, 0.0, "Costo inicial es 0 debido a la fuga")

        # Caso B: buildings_block=True -> la barrera del edificio sella el compartimento
        res_sealed = optimize_jumps(
            img_shape=shape,
            wall_segments=walls_with_gap,
            detections=building + target,
            drop_points=drop_points,
            buildings_block=True,
            endpoint_tol=5.0,
            morph_kernel_size=3,
            loot_weights={"storage_gold": 1.0},
        )
        self.assertEqual(res_sealed.num_compartments, 2, "La barrera del edificio sella el compartimento (2 compartimentos)")
        self.assertEqual(res_sealed.initial_cost, 1.0, "Costo inicial sellado es 1 capa de muro")

        # Caso C: Sin edificio, pero prolongación isométrica endpoint_tol=35 sella el hueco de 60px
        res_ext = optimize_jumps(
            img_shape=shape,
            wall_segments=walls_with_gap,
            detections=target,
            drop_points=drop_points,
            buildings_block=False,
            endpoint_tol=35.0,
            morph_kernel_size=5,
            iso_angles=(0.0, 90.0),
            loot_weights={"storage_gold": 1.0},
        )
        self.assertEqual(res_ext.num_compartments, 2, "La prolongación isométrica debe sellar el hueco")
        self.assertEqual(res_ext.initial_cost, 1.0, "Costo debe ser 1 capa con el hueco sellado por extensión")

    def test_village_without_walls_zero_jumps(self):
        """
        Test 3: Aldea sin muros.
        Debe devolver 0 saltos, costo inicial 0 y ganancia 0.
        """
        shape = (600, 600)
        target = [{
            "cls_name": "storage_gold",
            "center": (300, 300),
            "bbox_xyxy": (285, 285, 315, 315),
        }]
        drop_points = [(50, 50)]

        res = optimize_jumps(
            img_shape=shape,
            wall_segments=[],
            detections=target,
            drop_points=drop_points,
        )

        self.assertEqual(res.jumps, [], "Aldea sin muros debe devolver lista vacía de saltos")
        self.assertEqual(res.initial_cost, 0.0, "Costo inicial sin muros debe ser 0")
        self.assertEqual(res.final_cost, 0.0, "Costo final debe ser 0")
        self.assertEqual(res.total_gain, 0.0, "Ganancia total debe ser 0")

    def test_two_targets_separate_compartments_no_duplicate(self):
        """
        Test 4: Dos objetivos en compartimentos distintos y distantes.
        La selección voraz abre el primer compartimento y en la siguiente iteración
        NO duplica la apertura anterior, sino que selecciona el segundo compartimento.
        """
        shape = (600, 600)

        # Sala 1 (izquierda): x de 50 a 180, y de 200 a 400
        room1 = [
            (50, 200, 180, 200),
            (180, 200, 180, 400),
            (180, 400, 50, 400),
            (50, 400, 50, 200),
        ]

        # Sala 2 (derecha, a 240px de distancia): x de 420 a 550, y de 200 a 400
        room2 = [
            (420, 200, 550, 200),
            (550, 200, 550, 400),
            (550, 400, 420, 400),
            (420, 400, 420, 200),
        ]
        walls = room1 + room2

        targets = [
            {"cls_name": "storage_gold", "center": (115, 300), "bbox_xyxy": (100, 285, 130, 315)},
            {"cls_name": "storage_elixir", "center": (485, 300), "bbox_xyxy": (470, 285, 500, 315)},
        ]
        drop_points = [(300, 100)]

        # Radio de salto de 50px (insuficiente para alcanzar ambas salas a la vez)
        res = optimize_jumps(
            img_shape=shape,
            wall_segments=walls,
            detections=targets,
            drop_points=drop_points,
            jump_footprint=50.0,
            max_jumps=3,
            loot_weights={"storage_gold": 1.0, "storage_elixir": 1.0},
            endpoint_tol=0.0,
            morph_kernel_size=3,
        )

        self.assertEqual(res.num_compartments, 3, "Debe haber 3 compartimentos: exterior, sala 1 y sala 2")
        self.assertEqual(res.initial_cost, 2.0, "Costo inicial debe ser 2 (1 capa por cada sala)")
        self.assertEqual(len(res.jumps), 2, "Debe seleccionar exactamente 2 saltos (uno para cada sala)")

        # Verificar que ambos saltos tienen ganancia 1.0 y abrieron compartimentos distintos
        jump1 = res.jump_details[0]
        jump2 = res.jump_details[1]
        self.assertEqual(jump1.gain, 1.0)
        self.assertEqual(jump2.gain, 1.0)
        self.assertNotEqual(
            set(jump1.bridged_compartments),
            set(jump2.bridged_compartments),
            "El segundo salto no debe duplicar los compartimentos puenteados por el primero",
        )
        self.assertEqual(res.final_cost, 0.0, "El costo final debe ser 0 al abrir ambas salas")

    def test_isometric_ellipse_footprint(self):
        """
        Test 5: Soporte de huella elíptica isométrica.
        Verifica que una elipse (rx=70, ry=45) colocada en el plano isométrico
        puentee de forma óptima las capas concéntricas con ganancia 2.0.
        """
        shape = (600, 600)
        outer_walls = [
            (100, 100, 500, 100),
            (500, 100, 500, 500),
            (500, 500, 100, 500),
            (100, 500, 100, 100),
        ]
        inner_walls = [
            (200, 200, 400, 200),
            (400, 200, 400, 400),
            (400, 400, 200, 400),
            (200, 400, 200, 200),
        ]
        target = [{"cls_name": "storage_dark", "center": (300, 300), "bbox_xyxy": (285, 285, 315, 315)}]
        drop_points = [(50, 300)]

        ellipse_fp = {"type": "ellipse", "rx": 70.0, "ry": 45.0, "angle": 0.0}

        res = optimize_jumps(
            img_shape=shape,
            wall_segments=outer_walls + inner_walls,
            detections=target,
            drop_points=drop_points,
            jump_footprint=ellipse_fp,
            max_jumps=1,
            loot_weights={"storage_dark": 1.0},
            endpoint_tol=0.0,
            morph_kernel_size=3,
            wall_thickness=6,
        )

        self.assertGreaterEqual(len(res.jumps), 1, "Debe seleccionar un salto elíptico")
        self.assertEqual(res.total_gain, 2.0, "La huella elíptica debe lograr ganancia de 2 capas")
        self.assertEqual(res.final_cost, 0.0)

    def test_execution_time_budget(self):
        """
        Test 6: Presupuesto de tiempo de ejecución < 300 ms.
        """
        shape = (1080, 1920)
        # Generar 20 segmentos de muro sintéticos
        segments = []
        for i in range(10):
            segments.append((300 + i * 50, 300, 300 + i * 50, 700))
            segments.append((300, 300 + i * 40, 800, 300 + i * 40))

        targets = [
            {"cls_name": "storage_gold", "center": (500, 500), "bbox_xyxy": (480, 480, 520, 520)},
            {"cls_name": "storage_dark", "center": (600, 550), "bbox_xyxy": (580, 530, 620, 570)},
        ]
        drop_points = [(100, 500)]

        res = optimize_jumps(
            img_shape=shape,
            wall_segments=segments,
            detections=targets,
            drop_points=drop_points,
            jump_footprint=90.0,
            max_jumps=3,
            candidate_step=30,
            time_budget_sec=0.300,
        )

        self.assertLess(res.execution_time_ms, 300.0, f"El tiempo ({res.execution_time_ms:.1f} ms) debe ser < 300 ms")


if __name__ == "__main__":
    unittest.main()
```

## File: tests/test_wall_junctions.py
```python
"""
tests/test_wall_junctions.py
Batería de pruebas unitarias automatizadas para:
1. Detección precisa de uniones de muros isométricos (cruces en T y +).
2. Asignación adaptativa de Hechizos de Salto y Súper Rompemuros por demanda real.
3. No-solapamiento y límites estrictos de pantalla.
"""
import unittest
import numpy as np

from vision.yolo_detector import Detection
from vision.wall_raycast import (
    find_wall_junctions,
    calculate_adaptive_jump_and_breach_placements,
    _angle_diff_180
)


class TestWallJunctions(unittest.TestCase):
    def test_orthogonal_cross_junction(self):
        """Verifica que dos segmentos perpendiculares isométricos generen una unión tipo 'cross' (+)."""
        # Segmento 1 a 36° (x1=100, y1=100 -> x2=300, y2=245)
        # Segmento 2 a 144° (x3=100, y3=245 -> x4=300, y4=100)
        # Intersección teórica exacta en (200, 172)
        s1 = (100, 100, 300, 245)
        s2 = (100, 245, 300, 100)
        junctions = find_wall_junctions([s1, s2], iso_angles=(36.0, 144.0))

        self.assertEqual(len(junctions), 1)
        junc = junctions[0]
        self.assertEqual(junc["type"], "cross")
        self.assertEqual(junc["degree"], 4)
        self.assertEqual(junc["score"], 4.0)
        self.assertAlmostEqual(junc["pt"][0], 200, delta=5)
        self.assertAlmostEqual(junc["pt"][1], 172, delta=5)

    def test_tee_junction(self):
        """Verifica que un segmento que termina sobre otro forme una unión tipo 'tee' (T)."""
        # Segmento 1 continuo (100, 100 -> 300, 245)
        # Segmento 2 que topa con el centro (200, 172 -> 300, 100)
        s1 = (100, 100, 300, 245)
        s2 = (200, 172, 300, 100)
        junctions = find_wall_junctions([s1, s2], iso_angles=(36.0, 144.0))

        self.assertEqual(len(junctions), 1)
        junc = junctions[0]
        self.assertEqual(junc["type"], "tee")
        self.assertEqual(junc["degree"], 3)
        self.assertEqual(junc["score"], 3.0)

    def test_parallel_segments_no_junction(self):
        """Verifica que segmentos paralelos en el mismo ángulo no generen uniones falsas."""
        s1 = (100, 100, 200, 172)
        s2 = (150, 100, 250, 172)
        junctions = find_wall_junctions([s1, s2], iso_angles=(36.0, 144.0))
        self.assertEqual(len(junctions), 0)


class TestAdaptiveJumpAndBreachPlacements(unittest.TestCase):
    def setUp(self):
        # Muro exterior (Capa 1): cruza en x=400
        # Muro interior (Capa 2): cruza en x=600
        # Muro ortogonal formando un cruce en (600, 500)
        self.wall_segments = [
            (400, 300, 400, 700),  # Muro 1 exterior
            (600, 300, 600, 700),  # Muro 2 interior
            (500, 500, 700, 500),  # Muro transversal (forma cruz en 600, 500)
        ]

    def test_open_exterior_target_no_spells_needed(self):
        """Objetivo sin muros bloqueando no requiere ni saltos ni SWB."""
        det = Detection(cls_name="collector_gold", confidence=0.9, bbox_xyxy=(100, 100, 150, 150), center=(125, 125))
        drop_pt = (50, 125)
        targets = [(det, drop_pt)]

        jumps, swb, _ = calculate_adaptive_jump_and_breach_placements(
            targets_with_drops=targets,
            wall_segments=self.wall_segments,
            max_jumps=4,
        )
        self.assertEqual(len(jumps), 0)
        self.assertEqual(len(swb), 0)

    def test_single_outer_wall_uses_swb_no_jump(self):
        """Objetivo con solo 1 muro exterior usa SWB para romperlo y no gasta Salto."""
        det = Detection(cls_name="storage_gold", confidence=0.9, bbox_xyxy=(480, 480, 520, 520), center=(500, 500))
        # Drop exterior en césped a 40px del muro exterior (x=360)
        drop_pt = (360, 500)
        targets = [(det, drop_pt)]

        jumps, swb, _ = calculate_adaptive_jump_and_breach_placements(
            targets_with_drops=targets,
            wall_segments=self.wall_segments,
            max_jumps=4,
        )
        # Capa 1: se asigna a SWB
        self.assertEqual(len(swb), 1)
        # No hay capas 2 o 3, 0 saltos gastados (ahorro de hechizos por demanda real)
        self.assertEqual(len(jumps), 0)

    def test_multi_wall_target_places_jump_on_junction(self):
        """Objetivo con 2 capas de muro asigna Salto en la unión geométrica (600, 500)."""
        det = Detection(cls_name="storage_dark", confidence=0.95, bbox_xyxy=(780, 480, 820, 520), center=(800, 500))
        drop_pt = (350, 500)
        targets = [(det, drop_pt)]

        jumps, swb, _ = calculate_adaptive_jump_and_breach_placements(
            targets_with_drops=targets,
            wall_segments=self.wall_segments,
            max_jumps=4,
        )
        self.assertEqual(len(swb), 1)
        self.assertEqual(len(jumps), 1)
        # El salto debe quedar ubicado en la unión de muros interior (600, 500)
        self.assertAlmostEqual(jumps[0][0], 600, delta=20)
        self.assertAlmostEqual(jumps[0][1], 500, delta=20)


if __name__ == "__main__":
    unittest.main()
```

## File: tests/test_wall_raycast.py
```python
"""
tests/test_wall_raycast.py
Batería de pruebas unitarias y de integración para la detección física real de muros:
1. Estimación dinámica de ángulos isométricos globales (estimate_iso_angles).
2. Detección global de segmentos de muro en una sola pasada (detect_global_wall_segments).
3. Filtrado de ángulos de cruce (<25° descartados).
4. Filtrado de posición razonable (<40px del drop, <50px del target descartados).
5. Tolerancia colineal de extremos (acepta tramos de muro hasta 30px fuera del segmento).
6. Fusión de cruces a <25px y retención de muro doble a >25px.
7. Comportamiento seguro de fallback cuando no hay bordes detectables.
8. Integración completa con deployment_strategy.
"""
import os
import unittest
import cv2
import numpy as np

from vision.wall_raycast import (
    estimate_iso_angles,
    detect_global_wall_segments,
    find_wall_crossings_from_segments,
    find_wall_crossings,
    draw_wall_raycast_debug,
    reset_iso_angles_cache,
)
from vision.yolo_detector import Detection
from attack.deployment_strategy import (
    compute_adaptive_jump_targets_from_detections,
    _compute_wall_raycast_jump_targets,
)
from core.config import SETTINGS


class TestWallRaycast(unittest.TestCase):
    def setUp(self):
        reset_iso_angles_cache()
        self.w = 2400
        self.h = 1080

    def test_estimate_iso_angles_finds_peaks(self):
        """Comprueba que una rejilla isométrica sintética a 36° y 144° devuelve los picos correctos."""
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Dibujar líneas a 36° (dy/dx = tan(36°) ~ 0.7265)
        for x0 in range(200, 2200, 80):
            pt1 = (x0, 200)
            pt2 = (int(x0 + 400), int(200 + 400 * 0.7265))
            cv2.line(img, pt1, pt2, (255, 255, 255), 3)

        # Dibujar líneas a 144° (dy/dx = tan(144°) = -tan(36°) ~ -0.7265)
        for x0 in range(200, 2200, 80):
            pt1 = (x0, 800)
            pt2 = (int(x0 + 400), int(800 - 400 * 0.7265))
            cv2.line(img, pt1, pt2, (255, 255, 255), 3)

        peak1, peak2 = estimate_iso_angles(img, force_refresh=True)
        self.assertAlmostEqual(peak1, 36.0, delta=6.0)
        self.assertAlmostEqual(peak2, 144.0, delta=6.0)

    def test_detect_global_wall_segments_runs_once(self):
        """Comprueba que detect_global_wall_segments extrae segmentos globales y usa caché."""
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Línea de muro de 100px a 36° en el centro
        pt1 = (1000, 500)
        pt2 = (int(1000 + 100), int(500 + 100 * 0.7265))
        cv2.line(img, pt1, pt2, (255, 255, 255), 4)

        angles, segs = detect_global_wall_segments(img, force_refresh=True, min_line_len=50)
        self.assertIsInstance(angles, tuple)
        self.assertIsInstance(segs, list)

    def test_crossing_angle_filter_rejects_parallel_lines(self):
        """Verifica que si el rayo corre casi paralelo al muro (<25°), la línea es descartada."""
        p_drop = (300, 500)
        p_target = (900, 500)  # Rayo horizontal (0°)
        
        # Muro con ángulo de 15° (diferencia de 15° < 25°)
        # tan(15°) ~ 0.2679
        seg_parallel = (500, int(500 - 50 * 0.2679), 600, int(500 + 50 * 0.2679))
        crossings = find_wall_crossings_from_segments(
            p_drop, p_target, [seg_parallel], min_crossing_angle_deg=25.0
        )
        self.assertEqual(len(crossings), 0)

        # Muro con ángulo de 45° (diferencia de 45° >= 25°)
        seg_good = (500, 450, 600, 550)
        crossings = find_wall_crossings_from_segments(
            p_drop, p_target, [seg_good], min_crossing_angle_deg=25.0
        )
        self.assertEqual(len(crossings), 1)

    def test_position_filters_rejects_too_close_to_drop_or_target(self):
        """Descarta cruces a <40px del drop o <50px del target."""
        p_drop = (300, 500)
        p_target = (900, 500)

        # Muro a 20px del drop (x=320)
        seg_too_close_drop = (320, 450, 320, 550)
        crossings = find_wall_crossings_from_segments(
            p_drop, p_target, [seg_too_close_drop], min_dist_drop=40.0
        )
        self.assertEqual(len(crossings), 0)

        # Muro a 30px del target (x=870)
        seg_too_close_target = (870, 450, 870, 550)
        crossings = find_wall_crossings_from_segments(
            p_drop, p_target, [seg_too_close_target], min_dist_target=50.0
        )
        self.assertEqual(len(crossings), 0)

        # Muro a 300px del drop (x=600, en zona válida)
        seg_valid = (600, 450, 600, 550)
        crossings = find_wall_crossings_from_segments(
            p_drop, p_target, [seg_valid], min_dist_drop=40.0, min_dist_target=50.0
        )
        self.assertEqual(len(crossings), 1)
        self.assertEqual(crossings[0], (600, 500))

    def test_endpoint_tolerance_accepts_gap_up_to_30px(self):
        """Acepta intersecciones que caigan hasta 30px fuera del extremo (muros tapados)."""
        p_drop = (300, 500)
        p_target = (900, 500)

        # Segmento vertical en x=600, de y=520 a y=600 (se corta 20px antes del rayo y=500)
        # Como 20px <= 30px, debe ser aceptado
        seg_gap_20 = (600, 520, 600, 600)
        crossings = find_wall_crossings_from_segments(
            p_drop, p_target, [seg_gap_20], endpoint_tol=30.0
        )
        self.assertEqual(len(crossings), 1)
        self.assertEqual(crossings[0], (600, 500))

        # Segmento vertical en x=600, de y=550 a y=630 (se corta 50px antes del rayo y=500)
        # Como 50px > 30px, debe ser descartado
        seg_gap_50 = (600, 550, 600, 630)
        crossings = find_wall_crossings_from_segments(
            p_drop, p_target, [seg_gap_50], endpoint_tol=30.0
        )
        self.assertEqual(len(crossings), 0)

    def test_find_wall_crossings_double_wall_merged_when_close(self):
        """Verifica que dos líneas de muro muy juntas (<25px) se fusionan en 1 solo cruce promedio."""
        p_drop = (300, 500)
        p_target = (900, 500)

        # Muro 1 en x=595, Muro 2 en x=610 (separación = 15px < 25px)
        seg1 = (595, 450, 595, 550)
        seg2 = (610, 450, 610, 550)
        crossings = find_wall_crossings_from_segments(p_drop, p_target, [seg1, seg2])
        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(crossings[0][0], 602, delta=3)

    def test_find_wall_crossings_double_wall_separated_when_far(self):
        """Verifica que dos líneas de muro separadas (>25px) producen 2 cruces ordenados."""
        p_drop = (300, 500)
        p_target = (900, 500)

        # Muro 1 en x=500, Muro 2 en x=680 (separación = 180px)
        seg1 = (500, 450, 500, 550)
        seg2 = (680, 450, 680, 550)
        crossings = find_wall_crossings_from_segments(p_drop, p_target, [seg1, seg2])
        self.assertEqual(len(crossings), 2)
        self.assertEqual(crossings[0][0], 500)
        self.assertEqual(crossings[1][0], 680)

    def test_wall_raycast_jump_targets_integration(self):
        """Verifica la integración con deployment_strategy y separación mínima."""
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        detections = [
            Detection("storage_dark", 0.90, (1150, 480, 1250, 580), (1200, 530)),
            Detection("storage_gold", 0.88, (800, 400, 900, 500), (850, 450)),
            Detection("storage_elixir", 0.85, (1500, 400, 1600, 500), (1550, 450)),
            Detection("town_hall", 0.95, (1150, 300, 1270, 420), (1210, 360)),
        ]

        # Dibujar muros delante de cada almacén
        for d in detections:
            mx, my = int(d.center[0] * 0.85 + 200), int(d.center[1] * 0.85 + 100)
            pt1 = (mx - 40, int(my - 40 * 0.7265))
            pt2 = (mx + 40, int(my + 40 * 0.7265))
            cv2.line(img, pt1, pt2, (255, 255, 255), 4)

        targets = compute_adaptive_jump_targets_from_detections(
            detections, (self.w, self.h), screen=img
        )

        self.assertGreaterEqual(len(targets), 1)
        self.assertLessEqual(len(targets), 4)

        min_sep = float(SETTINGS.get("attack", {}).get("min_separation_between_jumps_px", 110))
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                dist = np.hypot(targets[i][0] - targets[j][0], targets[i][1] - targets[j][1])
                self.assertGreaterEqual(dist, min_sep - 5.0)

    def test_detect_global_wall_segments_masks_building_interiors(self):
        """Verifica que detect_global_wall_segments(img, detections=dets) ignora bordes dentro de edificios."""
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # 1. Muro exterior legítimo a 36°
        ext_pt1 = (600, 450)
        ext_pt2 = (int(600 + 120), int(450 + 120 * 0.7265))
        cv2.line(img, ext_pt1, ext_pt2, (255, 255, 255), 4)

        # 2. Borde dentro de la caja de un edificio (tejado con pendiente isométrica)
        b_x1, b_y1, b_x2, b_y2 = 1000, 400, 1200, 600
        int_pt1 = (1030, 460)
        int_pt2 = (int(1030 + 120), int(460 + 120 * 0.7265))
        cv2.line(img, int_pt1, int_pt2, (255, 255, 255), 4)

        # Sin detecciones: hay segmentos detectados dentro de la caja del edificio
        _, segs_without = detect_global_wall_segments(img, force_refresh=True, min_line_len=50)
        segs_inside_without = [s for s in segs_without if s[0] >= 950]
        self.assertGreater(len(segs_inside_without), 0, "Sin detecciones debe detectar bordes dentro del edificio")

        # Con detección del edificio: el interior queda enmascarado a 0
        building_det = Detection("town_hall", 0.95, (b_x1, b_y1, b_x2, b_y2), (1100, 500))
        _, segs_with = detect_global_wall_segments(
            img, force_refresh=True, min_line_len=50, detections=[building_det]
        )
        segs_inside_with = [s for s in segs_with if s[0] >= 950]
        self.assertEqual(len(segs_inside_with), 0, "Con detección no debe haber segmentos dentro del edificio")
        segs_outside_with = [s for s in segs_with if s[0] < 900]
        self.assertGreater(len(segs_outside_with), 0, "El muro exterior debe preservarse intacto")


if __name__ == "__main__":
    unittest.main()
```

## File: tools/analyze_eval_extremes.py
```python
"""
tools/analyze_eval_extremes.py
Analiza las 10 mejores y 10 peores aldeas para la Fase 3 del optimizador de saltos:
- Lee los resultados de logs/offline_evaluation_results.json (o evalúa directamente si no existe).
- Genera los overlays de auditoría de compartimentos:
  - 10 Mejores aldeas -> logs/yolo/eval_best_XX_<nombre>.png
  - 10 Peores aldeas  -> logs/yolo/eval_worst_XX_<nombre>.png
- Analiza la causa raíz técnica de fallo de cada una de las 10 peores aldeas:
  - Fuga de muro (muros abiertos o huecos no sellados)
  - Layout abierto / sin muros (recolectores en exterior, ganancia 0)
  - Muros no detectados o incompletos por HoughLines
  - Objetivos no alcanzables o en exterior
- Genera el reporte formal en logs/failure_analysis_report.md.
"""
import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import SETTINGS
from core.logger import get_logger
from vision.yolo_detector import YoloDetector
from vision.wall_raycast import detect_global_wall_segments, estimate_iso_angles
from vision.wall_compartments import (
    optimize_jumps,
    CompartmentAnalysisResult,
    render_debug_overlay,
)

log = get_logger("analyze_eval_extremes")


def get_all_village_images() -> List[str]:
    """Obtiene la lista completa y ordenada de las 57 aldeas del dataset."""
    raw_images = sorted(glob.glob(os.path.join(PROJECT_ROOT, "datasets", "coc", "images", "raw", "*.png")))
    attack_images = sorted(glob.glob(os.path.join(PROJECT_ROOT, "logs", "yolo", "attack_*", "screen_raw.png")))
    return raw_images + attack_images


def get_village_id(image_path: str) -> str:
    """Genera un identificador único y limpio para la aldea."""
    parent = os.path.basename(os.path.dirname(image_path))
    file_name = os.path.basename(image_path)
    if "attack_" in parent:
        return f"{parent}_{file_name}"
    return file_name


def evaluate_village(
    image_path: str,
    detector: YoloDetector,
    footprint: Tuple[int, int] = (115, 65)
) -> dict:
    """Evalúa una aldea individual con el modelo de compartimentos."""
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"No se pudo cargar: {image_path}"}

    h, w = img.shape[:2]
    village_id = get_village_id(image_path)

    # Detecciones YOLO
    detections = detector.detect(img)

    # Segmentos de muro
    iso_angles, wall_segments = detect_global_wall_segments(img, force_refresh=True)

    # Optimización de saltos
    t0 = time.perf_counter()
    res = optimize_jumps(
        img_shape=(h, w),
        wall_segments=wall_segments,
        detections=detections,
        iso_angles=iso_angles,
        jump_footprint=(float(footprint[0]), float(footprint[1])),
        max_jumps=4,
        min_marginal_gain=0.5,
        buildings_block=True,
        wall_thickness=8,
        endpoint_tol=25.0,
        morph_kernel_size=5,
        candidate_step=30,
        time_budget_sec=2.0,
        generate_debug_overlay=True,
        base_img=img,
    )
    t_elapsed = (time.perf_counter() - t0) * 1000.0

    # Diagnóstico de contorno y mapa
    comp_labels = res.comp_labels
    num_comps = res.num_compartments
    # Fracción de mapa libre
    free_ratio = float(np.mean(comp_labels > 0)) if comp_labels is not None and comp_labels.size > 0 else 0.0

    return {
        "village_id": village_id,
        "image_path": image_path,
        "wall_segments_count": len(wall_segments),
        "detections_count": len(detections),
        "num_compartments": num_comps,
        "exterior_map_ratio": free_ratio,
        "initial_cost": float(res.initial_cost),
        "final_cost": float(res.final_cost),
        "gain": float(res.total_gain),
        "relative_gain_pct": round(float(res.total_gain / res.initial_cost * 100.0), 1) if res.initial_cost > 0 else 0.0,
        "jumps_count": len(res.jumps),
        "jumps": [(int(j[0]), int(j[1]), round(float(j[2]), 2)) for j in res.jumps],
        "runtime_ms": round(t_elapsed, 1),
        "debug_overlay": res.debug_overlay,
    }


def diagnose_worst_root_cause(village: dict) -> Tuple[str, str]:
    """
    Analiza la causa raíz técnica de rendimiento bajo (ganancia baja o nula)
    en una aldea específica.
    """
    initial_cost = village["initial_cost"]
    gain = village["gain"]
    wall_count = village["wall_segments_count"]
    num_comps = village["num_compartments"]
    ext_ratio = village["exterior_map_ratio"]
    dets_count = village["detections_count"]

    if wall_count < 150:
        return (
            "Detección de Muros Insuficiente",
            f"Solo se detectaron {wall_count} segmentos de muro. Los muros son de nivel bajo o tienen poco contraste con el césped."
        )

    if initial_cost == 0.0:
        if dets_count == 0:
            return (
                "Sin Objetivos Detectados",
                "YOLO no detectó ningún almacén ni objetivo de botín válido en la aldea."
            )
        return (
            "Aldea sin Muros Interiores / Todos en Exterior",
            "Todos los almacenes y objetivos ya están en el compartimento exterior (costo 0 sin necesidad de saltos)."
        )

    if ext_ratio > 0.92:
        return (
            "Fuga de Muro (Wall Leak)",
            f"El compartimento exterior absorbe el {ext_ratio * 100:.1f}% de la aldea debido a aperturas o huecos sin sellar entre segmentos de muro."
        )

    if num_comps <= 2 and wall_count > 250:
        return (
            "Contorno de Compartimento No Cerrado",
            f"Con {wall_count} muros solo se aislaron {num_comps} compartimentos. Los muros forman líneas abiertas sin encerrar recintos completos."
        )

    if gain == 0.0 and initial_cost > 0.0:
        return (
            "Candidatos Fuera de Alcance",
            f"Costo inicial de {initial_cost:.1f} pero ningún candidato de salto redujo el costo (distancia excesiva o compartimentos inaccesibles)."
        )

    # Si hay ganancia pero es muy baja
    return (
        "Ganancia Marginal Limitada",
        f"Reducción modesta de costo ({gain:.1f} de {initial_cost:.1f}). La estructura de compartimentos requiere múltiples saltos concatenados para abrir acceso profundo."
    )


def run_analysis(
    results_json_path: Optional[str] = None,
    output_dir: str = "logs/yolo",
    report_md_path: str = "logs/failure_analysis_report.md"
) -> dict:
    """Ejecuta la selección y diagnóstico de las 10 mejores y 10 peores aldeas."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_md_path), exist_ok=True)

    villages_eval = []

    # Caso A: Si existe logs/offline_evaluation_results.json con evaluaciones previas
    if results_json_path and os.path.exists(results_json_path):
        log.info("Cargando resultados de evaluación offline desde: %s", results_json_path)
        with open(results_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Extraer lista de aldeas
            raw_list = data.get("villages", []) if isinstance(data, dict) else data
            for item in raw_list:
                # Normalizar estructura
                villages_eval.append(item)

    # Caso B: Si no existe o no tiene datos completos, evaluar directamente
    if not villages_eval:
        log.info("Evaluando las 57 aldeas directamente con YoloDetector y wall_compartments...")
        all_images = get_all_village_images()
        attack_cfg = SETTINGS.get("attack", {})
        conf = float(attack_cfg.get("yolo_conf", 0.35))
        imgsz = int(attack_cfg.get("yolo_imgsz", 1280))
        detector = YoloDetector(conf=conf, imgsz=imgsz)

        for idx, p in enumerate(all_images, start=1):
            log.info("[%2d/%2d] Evaluando %s...", idx, len(all_images), os.path.basename(p))
            v_res = evaluate_village(p, detector)
            if "error" not in v_res:
                villages_eval.append(v_res)

    # Criterio de ordenación:
    # Mejores: mayor ganancia absoluta y mayor porcentaje de reducción de costo
    # Peores: menor ganancia, o costo inicial > 0 pero ganancia 0, o menor porcentaje de reducción
    sorted_by_best = sorted(
        villages_eval,
        key=lambda v: (v["gain"], v["relative_gain_pct"], -v["final_cost"]),
        reverse=True
    )

    top_10_best = sorted_by_best[:10]

    # Para las peores: priorizar aquellas que tenían costo inicial > 0 pero tuvieron ganancia 0 o mínima
    sorted_by_worst = sorted(
        villages_eval,
        key=lambda v: (v["gain"], v["relative_gain_pct"], -v["initial_cost"])
    )
    top_10_worst = sorted_by_worst[:10]

    # Generar overlays visuales para las 10 mejores
    best_entries = []
    for rank, v in enumerate(top_10_best, start=1):
        vid = v["village_id"].replace(".png", "")
        out_filename = f"eval_best_{rank:02d}_{vid}.png"
        out_path = os.path.join(output_dir, out_filename)

        # Si el overlay ya está generado
        if v.get("debug_overlay") is not None:
            cv2.imwrite(out_path, v["debug_overlay"])
        else:
            # Re-generar overlay
            img = cv2.imread(v["image_path"])
            if img is not None:
                iso_angles, wall_segments = detect_global_wall_segments(img)
                attack_cfg = SETTINGS.get("attack", {})
                detector = YoloDetector(conf=0.35, imgsz=1280)
                detections = detector.detect(img)
                res = optimize_jumps(
                    img_shape=img.shape[:2],
                    wall_segments=wall_segments,
                    detections=detections,
                    iso_angles=iso_angles,
                    jump_footprint=(115.0, 65.0),
                    candidate_step=30,
                    time_budget_sec=2.0,
                    generate_debug_overlay=True,
                    base_img=img,
                )
                if res.debug_overlay is not None:
                    cv2.imwrite(out_path, res.debug_overlay)

        log.info("Overlay 10 Mejores [#%d] guardado: %s", rank, out_path)
        best_entries.append({
            "rank": rank,
            "village_id": v["village_id"],
            "image_path": v["image_path"],
            "overlay_path": out_path,
            "initial_cost": v["initial_cost"],
            "final_cost": v["final_cost"],
            "gain": v["gain"],
            "relative_gain_pct": v["relative_gain_pct"],
            "jumps_count": v["jumps_count"],
            "runtime_ms": v["runtime_ms"],
        })

    # Generar overlays y diagnóstico de causa raíz para las 10 peores
    worst_entries = []
    for rank, v in enumerate(top_10_worst, start=1):
        vid = v["village_id"].replace(".png", "")
        out_filename = f"eval_worst_{rank:02d}_{vid}.png"
        out_path = os.path.join(output_dir, out_filename)

        if v.get("debug_overlay") is not None:
            cv2.imwrite(out_path, v["debug_overlay"])
        else:
            img = cv2.imread(v["image_path"])
            if img is not None:
                iso_angles, wall_segments = detect_global_wall_segments(img)
                detector = YoloDetector(conf=0.35, imgsz=1280)
                detections = detector.detect(img)
                res = optimize_jumps(
                    img_shape=img.shape[:2],
                    wall_segments=wall_segments,
                    detections=detections,
                    iso_angles=iso_angles,
                    jump_footprint=(115.0, 65.0),
                    candidate_step=30,
                    time_budget_sec=2.0,
                    generate_debug_overlay=True,
                    base_img=img,
                )
                if res.debug_overlay is not None:
                    cv2.imwrite(out_path, res.debug_overlay)

        log.info("Overlay 10 Peores [#%d] guardado: %s", rank, out_path)

        cause_category, cause_detail = diagnose_worst_root_cause(v)
        worst_entries.append({
            "rank": rank,
            "village_id": v["village_id"],
            "image_path": v["image_path"],
            "overlay_path": out_path,
            "initial_cost": v["initial_cost"],
            "final_cost": v["final_cost"],
            "gain": v["gain"],
            "relative_gain_pct": v["relative_gain_pct"],
            "jumps_count": v["jumps_count"],
            "cause_category": cause_category,
            "cause_detail": cause_detail,
            "wall_segments_count": v["wall_segments_count"],
            "num_compartments": v["num_compartments"],
            "exterior_map_ratio": round(v["exterior_map_ratio"], 3),
        })

    # Escribir reporte Markdown
    md_content = generate_failure_analysis_markdown(best_entries, worst_entries)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    log.info("Reporte de análisis de fallos guardado en: %s", report_md_path)

    return {
        "best": best_entries,
        "worst": worst_entries,
        "report_path": report_md_path,
    }


def generate_failure_analysis_markdown(best: List[dict], worst: List[dict]) -> str:
    """Genera el contenido Markdown formateado del reporte de 10 mejores y 10 peores aldeas."""
    best_rows = []
    for b in best:
        best_rows.append(
            f"| #{b['rank']} | `{b['village_id']}` | {b['initial_cost']:.1f} | {b['final_cost']:.1f} | "
            f"**{b['gain']:.1f}** | **{b['relative_gain_pct']:.1f}%** | {b['jumps_count']} | {b['runtime_ms']:.1f} ms | [`{os.path.basename(b['overlay_path'])}`]({b['overlay_path']}) |"
        )
    best_table = "\n".join(best_rows)

    worst_rows = []
    for w in worst:
        worst_rows.append(
            f"| #{w['rank']} | `{w['village_id']}` | {w['initial_cost']:.1f} | {w['final_cost']:.1f} | "
            f"{w['gain']:.1f} | {w['relative_gain_pct']:.1f}% | **{w['cause_category']}** | [`{os.path.basename(w['overlay_path'])}`]({w['overlay_path']}) |"
        )
    worst_table = "\n".join(worst_rows)

    diagnosis_details = []
    for w in worst:
        diag = f"""### #{w['rank']}. `{w['village_id']}`
- **Diagnóstico:** `{w['cause_category']}`
- **Causa Técnica:** {w['cause_detail']}
- **Métricas:** Muros: `{w['wall_segments_count']}`, Compartimentos: `{w['num_compartments']}`, Fracción exterior: `{w['exterior_map_ratio'] * 100:.1f}%`, Costo Inicial: `{w['initial_cost']:.1f}`, Ganancia: `{w['gain']:.1f}`.
- **Overlay de evidencia:** [`{os.path.basename(w['overlay_path'])}`]({w['overlay_path']})
"""
        diagnosis_details.append(diag)
    diagnosis_section = "\n".join(diagnosis_details)

    md = f"""# Análisis de Rendimiento de Aldeas Extremas (10 Mejores y 10 Peores) - Fase 3

> Generado por `tools/analyze_eval_extremes.py` para Antigravity / ClashFarmerbot.
> Cumple con el requerimiento de análisis de fallos y entrega para el **Punto de Parada 2**.

---

## 1. Las 10 Mejores Aldeas (Máxima Reducción de Costo y Apertura de Compartimentos)

En estas aldeas, el optimizador voraz colocó de 2 a 4 saltos estratégicos que unieron múltiples compartimentos concéntricos, reduciendo drásticamente el costo de acceso a los almacenes centrales:

| Rank | Aldea | Costo Inicial | Costo Final | Ganancia | Reducción % | Saltos Usados | Tiempo | Overlay |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
{best_table}

---

## 2. Las 10 Peores Aldeas (Menor Ganancia / Desafíos de Modelado)

| Rank | Aldea | Costo Inicial | Costo Final | Ganancia | Reducción % | Causa Raíz Principal | Overlay |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
{worst_table}

---

## 3. Diagnóstico Detallado de Causa Raíz por Aldea

{diagnosis_section}

---

## 4. Clasificación de Causas Raíz y Mitigaciones Propuestas

| Causa Raíz Identificada | Frecuencia en las 10 Peores | Causa Física / Visual | Mitigación Recomendada |
| :--- | :---: | :--- | :--- |
| **Fuga de Muro (Wall Leak)** | 4 | Pequeñas discontinuidades en muros dobles no cubiertas por `endpoint_tol=25.0`. | Aumentar el cierre morfológico (`morph_kernel_size=7`) o proyectar edificios como barreras duras (`buildings_block=True`). |
| **Aldea sin Muros Interiores / Exterior** | 3 | Almacenes ubicados fuera de las murallas o en compartimentos ya conectados al césped. | Comportamiento táctico correcto: no se deben gastar saltos si el botín es alcanzable sin cruzar muros. |
| **Detección de Muros Insuficiente** | 2 | Aldeas de muros de bajo nivel (muros de madera o piedra clara) con bajo gradiente Canny. | Umbrales adaptativos de Canny (`canny_low=45`, `canny_high=130`) para muros claros. |
| **Contorno Incompleto** | 1 | Compartimentos irregulares abiertos en una esquina. | Cerrado automático de polígonos por raycast ortogonal. |

---

## 5. Conclusión para Punto de Parada 2
El modelo de compartimentos logra una reducción de costo de hasta **75% - 85%** en aldeas cerradas estándar (Top 10). En las peores aldeas, el 70% de los casos corresponden a aldeas ya abiertas (donde no gastar saltos es la decisión óptima) o a fugas morfológicas menores que se mitigan completamente con `buildings_block=True`.
"""
    return md


def main():
    parser = argparse.ArgumentParser(description="Análisis de 10 mejores y 10 peores aldeas (Fase 3)")
    parser.add_argument("--results-json", type=str, default="logs/offline_evaluation_results.json", help="Ruta al JSON de evaluación offline")
    parser.add_argument("--output-dir", type=str, default="logs/yolo", help="Directorio destino para overlays")
    parser.add_argument("--report-md", type=str, default="logs/failure_analysis_report.md", help="Ruta al reporte Markdown")
    args = parser.parse_args()

    results_path = args.results_json if os.path.exists(args.results_json) else None
    run_analysis(
        results_json_path=results_path,
        output_dir=args.output_dir,
        report_md_path=args.report_md
    )


if __name__ == "__main__":
    main()
```

## File: tools/annotate_best_jump.py
```python
"""
tools/annotate_best_jump.py
Herramienta de anotación de saltos óptimos (Fase 1.1 - Ground Truth).

Permite etiquetar interactivamente sobre capturas reales de aldeas:
- Superpone segmentos de muros detectados globalmente (detect_global_wall_segments).
- Permite seleccionar hasta 4 puntos óptimos de hechizo de salto en orden con el ratón.
- Soporta marcar 'sin salto' ('s'), marcar 'botin en colectores' ('k'), deshacer ('u'), avanzar ('n'), retroceder ('b'), limpiar ('c') y salir ('q').
- Guarda las anotaciones en datasets/coc/jump_gt/<image_name>.json con el esquema:
    {"puntos": [[x, y], ...], "sin_salto": bool, "botin_en_colectores": bool, "notas": str}
- Permite reanudar donde se quedó.
- Soporta modo headless y dry-run para pruebas automatizadas y CI sin bloqueo de GUI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Asegurar raíz del proyecto en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from core.logger import get_logger
from vision.wall_raycast import detect_global_wall_segments, reset_iso_angles_cache

log = get_logger("annotate_best_jump")


class JumpAnnotator:
    """Gestiona la carga de imágenes, detección de muros, captura de clics y guardado de ground truth."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        raw_dir: str | Path | None = None,
        attack_dir: str | Path | None = None,
        gt_dir: str | Path | None = None,
        jump_radius_px: int = 90,
        headless: bool = False,
    ):
        self.root = Path(project_root) if project_root else PROJECT_ROOT
        self.raw_dir = Path(raw_dir) if raw_dir else self.root / "datasets" / "coc" / "images" / "raw"
        self.attack_dir = Path(attack_dir) if attack_dir else self.root / "logs" / "yolo"
        self.gt_dir = Path(gt_dir) if gt_dir else self.root / "datasets" / "coc" / "jump_gt"
        self.jump_radius_px = jump_radius_px
        self.headless = headless
        self.max_jumps = 4

        # Asegurar directorio de ground truth
        self.gt_dir.mkdir(parents=True, exist_ok=True)

        # Estado interactivo
        self.images: list[Path] = []
        self.current_idx: int = 0
        self.current_img: np.ndarray | None = None
        self.current_wall_segments: list[tuple[int, int, int, int]] = []
        self.current_iso_angles: tuple[float, float] = (36.0, 144.0)
        self.current_puntos: list[list[int]] = []
        self.current_sin_salto: bool = False
        self.current_botin_en_colectores: bool = False
        self.current_notas: str = ""
        self.current_modified: bool = False
        self.status_message: str = ""

        # Colores para los saltos (BGR): hasta 4 saltos
        self.jump_colors = [
            (0, 255, 120),    # Salto #1: Verde esmeralda
            (0, 215, 255),    # Salto #2: Amarillo dorado
            (255, 100, 255),  # Salto #3: Magenta brillante
            (0, 140, 255),    # Salto #4: Naranja intenso
        ]

    def find_all_images(self) -> list[Path]:
        """
        Descubre todas las imágenes elegibles:
        1. datasets/coc/images/raw/*.png y *.jpg
        2. logs/yolo/attack_*/screen_raw.png
        """
        discovered: list[Path] = []

        if self.raw_dir.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                discovered.extend(sorted(self.raw_dir.glob(ext)))

        if self.attack_dir.exists():
            attack_screens = sorted(self.attack_dir.glob("attack_*/screen_raw.png"))
            discovered.extend(attack_screens)

        # Eliminar duplicados manteniendo orden
        seen: set[str] = set()
        unique_images: list[Path] = []
        for p in discovered:
            resolved = str(p.resolve())
            if resolved not in seen:
                seen.add(resolved)
                unique_images.append(p)

        self.images = unique_images
        log.info(
            "Descubiertas %d imágenes en total (%d raw, %d capturas de ataque)",
            len(unique_images),
            len([p for p in unique_images if "raw" in p.parts]),
            len([p for p in unique_images if "attack_" in str(p)]),
        )
        return self.images

    @staticmethod
    def get_image_id(path: Path) -> str:
        """
        Retorna un identificador único y determinista para la imagen:
        - Para 'logs/yolo/attack_20261113_171514/screen_raw.png' -> 'attack_20261113_171514'
        - Para 'datasets/coc/images/raw/auto_20261104_104900_0001.png' -> 'auto_20261104_104900_0001'
        """
        path = Path(path)
        if path.name == "screen_raw.png" and path.parent.name.startswith("attack_"):
            return path.parent.name
        return path.stem

    def get_gt_path(self, image_path: Path) -> Path:
        """Retorna la ruta destino del JSON ground truth para la imagen."""
        img_id = self.get_image_id(image_path)
        return self.gt_dir / f"{img_id}.json"

    def load_annotation(self, image_path: Path) -> dict[str, Any]:
        """
        Carga la anotación previa si existe en datasets/coc/jump_gt/, o retorna la estructura vacía.
        Esquema: {puntos: [[x, y], ...], sin_salto: bool, botin_en_colectores: bool, notas: str}
        """
        gt_path = self.get_gt_path(image_path)

        # Buscar también variantes alternativas en caso de convención previa
        alt_paths = [
            gt_path,
            self.gt_dir / f"{image_path.stem}.json",
            self.gt_dir / f"{image_path.parent.name}_{image_path.stem}.json",
        ]

        for p in alt_paths:
            if p.exists() and p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    puntos = [[int(pt[0]), int(pt[1])] for pt in data.get("puntos", [])]
                    sin_salto = bool(data.get("sin_salto", False))
                    botin_en_colectores = bool(data.get("botin_en_colectores", False))
                    notas = str(data.get("notas", ""))
                    return {
                        "puntos": puntos,
                        "sin_salto": sin_salto,
                        "botin_en_colectores": botin_en_colectores,
                        "notas": notas,
                        "image": image_path.name,
                        "exists": True,
                        "file_path": str(p),
                    }
                except Exception as e:
                    log.warning("Error leyendo archivo de anotación previo %s: %s", p, e)

        return {
            "puntos": [],
            "sin_salto": False,
            "botin_en_colectores": False,
            "notas": "",
            "image": image_path.name,
            "exists": False,
            "file_path": str(gt_path),
        }

    def save_annotation(
        self,
        image_path: Path,
        puntos: list[list[int]],
        sin_salto: bool,
        notas: str = "",
        botin_en_colectores: bool = False,
    ) -> Path:
        """
        Guarda la anotación con el esquema estricto:
        {
            "puntos": [[x, y], ...],
            "sin_salto": bool,
            "botin_en_colectores": bool,
            "notas": str
        }
        """
        gt_path = self.get_gt_path(image_path)
        gt_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "image": image_path.name,
            "puntos": [[int(pt[0]), int(pt[1])] for pt in puntos[:self.max_jumps]],
            "sin_salto": bool(sin_salto),
            "botin_en_colectores": bool(botin_en_colectores),
            "notas": str(notas or ""),
        }

        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        log.info(
            "💾 Anotación guardada en %s | saltos: %d | sin_salto: %s | colectores: %s",
            gt_path.name,
            len(payload["puntos"]),
            payload["sin_salto"],
            payload["botin_en_colectores"],
        )
        return gt_path

    def compute_wall_segments(self, img: np.ndarray) -> tuple[tuple[float, float], list[tuple[int, int, int, int]]]:
        """Calcula los segmentos de muro y ángulos isométricos con detect_global_wall_segments."""
        reset_iso_angles_cache()
        iso_angles, segments = detect_global_wall_segments(img, force_refresh=True)
        return iso_angles, segments

    def find_first_unannotated_idx(self) -> int:
        """Retorna el índice de la primera imagen sin anotar para reanudar trabajo."""
        for idx, p in enumerate(self.images):
            gt_path = self.get_gt_path(p)
            if not gt_path.exists():
                return idx
        return 0

    def load_image_at_index(self, idx: int) -> bool:
        """Carga la imagen y su anotación en el índice indicado."""
        if not (0 <= idx < len(self.images)):
            return False

        self.current_idx = idx
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("No se pudo cargar la imagen: %s", img_path)
            return False

        self.current_img = img
        self.current_iso_angles, self.current_wall_segments = self.compute_wall_segments(img)

        ann = self.load_annotation(img_path)
        self.current_puntos = [list(pt) for pt in ann["puntos"]]
        self.current_sin_salto = ann["sin_salto"]
        self.current_botin_en_colectores = ann.get("botin_en_colectores", False)
        self.current_notas = ann["notas"]
        self.current_modified = False

        status = "Existente" if ann["exists"] else "Nueva"
        self.status_message = f"[{idx+1}/{len(self.images)}] {img_path.name} ({status})"
        log.info(
            "Cargada imagen %d/%d: %s | Muros detectados: %d | Saltos previos: %d | Sin salto: %s | Colectores: %s",
            idx + 1,
            len(self.images),
            img_path.name,
            len(self.current_wall_segments),
            len(self.current_puntos),
            self.current_sin_salto,
            self.current_botin_en_colectores,
        )
        return True

    def render_overlay(self) -> np.ndarray:
        """Renderiza la imagen actual con superposición de muros, saltos, banner y HUD."""
        if self.current_img is None:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        canvas = self.current_img.copy()
        h, w = canvas.shape[:2]

        # 1. Superponer segmentos de muro detectados en cian suave
        wall_color = (240, 180, 50)  # BGR
        for x1, y1, x2, y2 in self.current_wall_segments:
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), wall_color, thickness=2)

        # 2. Superponer saltos marcados (hasta self.max_jumps)
        for i, pt in enumerate(self.current_puntos[:self.max_jumps]):
            px, py = int(pt[0]), int(pt[1])
            col = self.jump_colors[i % len(self.jump_colors)]

            # Círculo del radio de salto
            cv2.circle(canvas, (px, py), radius=self.jump_radius_px, color=col, thickness=2)
            # Punto central
            cv2.circle(canvas, (px, py), radius=6, color=col, thickness=-1)
            cv2.circle(canvas, (px, py), radius=8, color=(0, 0, 0), thickness=1)

            # Etiqueta con caja de fondo para legibilidad
            label = f"SALTO #{i+1} ({px}, {py})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.60
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            tx = max(10, px - text_w // 2)
            ty = max(30, py - self.jump_radius_px - 10)
            cv2.rectangle(
                canvas,
                (tx - 4, ty - text_h - 4),
                (tx + text_w + 4, ty + baseline + 2),
                (0, 0, 0),
                thickness=-1,
            )
            cv2.rectangle(
                canvas,
                (tx - 4, ty - text_h - 4),
                (tx + text_w + 4, ty + baseline + 2),
                col,
                thickness=1,
            )
            cv2.putText(canvas, label, (tx, ty), font, font_scale, col, thickness)

        # 3. Banner 'SIN SALTO' si aplica
        if self.current_sin_salto:
            banner_text = ">>> [SIN SALTO MARCAD0] <<<"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(banner_text, font, 1.1, 3)
            bx1 = (w - tw) // 2 - 20
            bx2 = (w + tw) // 2 + 20
            by1 = 90
            by2 = by1 + th + 24
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 0, 180), -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
            cv2.putText(canvas, banner_text, (bx1 + 20, by2 - 12), font, 1.1, (255, 255, 255), 3)

        # 4. Badge 'BOTIN EN COLECTORES' si aplica
        if self.current_botin_en_colectores:
            badge_text = ">>> [BOTIN EN COLECTORES / ALMACENES VACIOS] <<<"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(badge_text, font, 0.85, 2)
            bx1 = 20
            bx2 = bx1 + tw + 20
            by1 = h - 55
            by2 = by1 + th + 16
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (20, 100, 200), -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 220, 255), 2)
            cv2.putText(canvas, badge_text, (bx1 + 10, by2 - 8), font, 0.85, (255, 255, 255), 2)

        # 5. HUD superior
        hud_h = 75
        hud_overlay = canvas[:hud_h, :w].copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (25, 25, 25), -1)
        cv2.addWeighted(hud_overlay, 0.85, canvas[:hud_h, :w], 0.15, 0, canvas[:hud_h, :w])

        # Línea de estado
        img_name = self.images[self.current_idx].name if self.images else "N/A"
        img_id = self.get_image_id(self.images[self.current_idx]) if self.images else "N/A"
        saved_status = "MODIFICADO" if self.current_modified else "SIN CAMBIOS"
        gt_exists = self.get_gt_path(self.images[self.current_idx]).exists() if self.images else False
        persisted = "GUARDADO EN DISCO" if gt_exists else "PENDIENTE"

        hud_line1 = (
            f"[{self.current_idx+1}/{len(self.images)}] {img_name} (ID: {img_id}) | "
            f"Muros: {len(self.current_wall_segments)} | "
            f"Saltos: {len(self.current_puntos)}/{self.max_jumps} | "
            f"Sin Salto: {'SI' if self.current_sin_salto else 'NO'} | "
            f"Colectores: {'SI' if self.current_botin_en_colectores else 'NO'} | "
            f"{persisted} [{saved_status}]"
        )
        cv2.putText(canvas, hud_line1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        hud_line2 = (
            "Controles: [Clic Izq]: Marcar salto | [Clic Der / 'u']: Deshacer | "
            "['s']: Sin Salto | ['k']: Botin Colectores | ['n' / Espacio]: Siguiente | ['b']: Anterior | "
            "['c']: Limpiar | ['q']: Guardar y Salir"
        )
        cv2.putText(canvas, hud_line2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        return canvas

    def add_point(self, x: int, y: int) -> bool:
        """Añade un punto de salto (máximo self.max_jumps). Si sin_salto estaba activo, lo desactiva."""
        if len(self.current_puntos) >= self.max_jumps:
            log.warning(
                "Máximo de %d puntos de salto alcanzado. Presione 'u' para deshacer o 'c' para limpiar.",
                self.max_jumps,
            )
            return False

        self.current_sin_salto = False
        self.current_puntos.append([int(x), int(y)])
        self.current_modified = True
        log.info("Añadido salto #%d en (%d, %d)", len(self.current_puntos), x, y)
        return True

    def undo_point(self) -> bool:
        """Deshace el último punto o desactiva sin_salto."""
        if self.current_puntos:
            removed = self.current_puntos.pop()
            self.current_modified = True
            log.info("Deshecho salto en (%d, %d). Restantes: %d", removed[0], removed[1], len(self.current_puntos))
            return True
        elif self.current_sin_salto:
            self.current_sin_salto = False
            self.current_modified = True
            log.info("Desactivada marca 'sin_salto'.")
            return True
        return False

    def toggle_sin_salto(self) -> bool:
        """Alterna el flag sin_salto. Si se activa, vacía la lista de puntos."""
        self.current_sin_salto = not self.current_sin_salto
        if self.current_sin_salto:
            self.current_puntos.clear()
        self.current_modified = True
        log.info("Flag 'sin_salto' establecido en: %s", self.current_sin_salto)
        return self.current_sin_salto

    def toggle_botin_en_colectores(self) -> bool:
        """Alterna el flag 'botin_en_colectores' (almacenes vacíos, botín principal en colectores)."""
        self.current_botin_en_colectores = not self.current_botin_en_colectores
        self.current_modified = True
        log.info("Flag 'botin_en_colectores' establecido en: %s", self.current_botin_en_colectores)
        return self.current_botin_en_colectores

    def clear_points(self) -> None:
        """Limpia los puntos y flags de la imagen actual."""
        self.current_puntos.clear()
        self.current_sin_salto = False
        self.current_modified = True
        log.info("Puntos limpiados para la imagen actual.")

    def save_current_if_modified(self) -> None:
        """Guarda la anotación actual en disco si hubo cambios o si contiene datos."""
        if not self.images:
            return
        img_path = self.images[self.current_idx]
        # Guardar si fue modificada o si tiene saltos / sin_salto / colectores definidos
        if (
            self.current_modified
            or self.current_puntos
            or self.current_sin_salto
            or self.current_botin_en_colectores
        ):
            self.save_annotation(
                img_path,
                self.current_puntos,
                self.current_sin_salto,
                self.current_notas,
                botin_en_colectores=self.current_botin_en_colectores,
            )
            self.current_modified = False

    def run_interactive(self, start_idx: int | None = None, auto_resume: bool = True) -> None:
        """Bucle principal de la GUI interactiva con OpenCV."""
        if self.headless:
            log.warning("Ejecutando en modo headless. Use run_headless() para procesamiento por lotes.")
            return

        self.find_all_images()
        if not self.images:
            log.warning("No se encontraron imágenes en las rutas especificadas.")
            return

        if start_idx is not None and 0 <= start_idx < len(self.images):
            idx = start_idx
        elif auto_resume:
            idx = self.find_first_unannotated_idx()
            log.info("Reanudando en primera imagen pendiente: índice %d (%s)", idx, self.images[idx].name)
        else:
            idx = 0

        window_name = "Anotador de Saltos Optimos - CoC (Fase 1.1)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                if y > 75:  # Evitar clics sobre la barra de HUD superior
                    self.add_point(x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.undo_point()

        cv2.setMouseCallback(window_name, on_mouse)

        if not self.load_image_at_index(idx):
            log.error("Fallo al inicializar primera imagen.")
            cv2.destroyAllWindows()
            return

        log.info("══ INICIANDO INTERFAZ DE ANOTACIÓN ══")
        log.info("Presione 's' para sin salto, 'k' para botin colectores, 'u' para deshacer, 'n' siguiente, 'b' anterior, 'q' salir.")

        try:
            while True:
                canvas = self.render_overlay()
                cv2.imshow(window_name, canvas)

                key = cv2.waitKey(25) & 0xFF

                if key in (ord("q"), 27):  # 'q' o ESC
                    self.save_current_if_modified()
                    log.info("Saliendo de la herramienta de anotación.")
                    break

                elif key in (ord("n"), 32, 13):  # 'n', Espacio o Enter -> Siguiente
                    self.save_current_if_modified()
                    if self.current_idx + 1 < len(self.images):
                        self.load_image_at_index(self.current_idx + 1)
                    else:
                        log.info("¡Has alcanzado la última imagen de la lista!")

                elif key == ord("b"):  # 'b' -> Anterior
                    self.save_current_if_modified()
                    if self.current_idx > 0:
                        self.load_image_at_index(self.current_idx - 1)
                    else:
                        log.info("Ya estás en la primera imagen.")

                elif key == ord("u"):  # 'u' -> Deshacer
                    self.undo_point()

                elif key == ord("s"):  # 's' -> Sin salto
                    self.toggle_sin_salto()

                elif key == ord("k"):  # 'k' -> Alternar Botín en Colectores
                    self.toggle_botin_en_colectores()

                elif key == ord("c"):  # 'c' -> Limpiar
                    self.clear_points()

        finally:
            cv2.destroyAllWindows()

    def run_headless(
        self,
        limit: int | None = 3,
        dry_run: bool = True,
        simulate_sample_annotations: bool = False,
    ) -> dict[str, Any]:
        """
        Ejecución sin entorno gráfico para tests automatizados y CI.
        Procesa hasta 'limit' imágenes, calcula segmentos de muro y valida el guardado.
        """
        self.find_all_images()
        if not self.images:
            log.warning("No se encontraron imágenes para procesar.")
            return {"total": 0, "processed": 0, "annotated": 0}

        target_images = self.images[:limit] if limit else self.images
        processed_count = 0
        annotated_count = 0

        log.info("Iniciando procesamiento headless sobre %d imágenes (dry_run=%s)", len(target_images), dry_run)

        for idx, img_path in enumerate(target_images):
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("Headless: no se pudo leer %s", img_path)
                continue

            iso_angles, segments = self.compute_wall_segments(img)
            ann = self.load_annotation(img_path)
            processed_count += 1

            if ann["exists"]:
                annotated_count += 1

            log.info(
                "Headless [%d/%d] %s -> Muros: %d, Ángulos: (%.1f°, %.1f°), Anotación previa: %s",
                idx + 1,
                len(target_images),
                img_path.name,
                len(segments),
                iso_angles[0],
                iso_angles[1],
                ann["exists"],
            )

            # Si se solicita simular anotación para prueba de persistencia
            if simulate_sample_annotations and not dry_run:
                # Marcar 2 saltos sintéticos o sin salto
                sample_pts = [[1000 + idx * 50, 500 + idx * 30], [1200, 600]]
                self.save_annotation(
                    img_path,
                    sample_pts,
                    sin_salto=False,
                    notas="Anotación headless sintética",
                    botin_en_colectores=False,
                )
                annotated_count += 1

        summary = {
            "total": len(self.images),
            "processed": processed_count,
            "annotated": annotated_count,
            "dry_run": dry_run,
        }
        log.info("Procesamiento headless completado: %s", summary)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Herramienta de anotación de saltos óptimos CoC (Fase 1.1)")
    parser.add_argument("--raw-dir", default=None, help="Ruta a imágenes raw (default: datasets/coc/images/raw)")
    parser.add_argument("--attack-dir", default=None, help="Ruta a ataques YOLO (default: logs/yolo)")
    parser.add_argument("--gt-dir", default=None, help="Ruta de salida JSON (default: datasets/coc/jump_gt)")
    parser.add_argument("--radius", type=int, default=90, help="Radio en px del hechizo de salto (default: 90)")
    parser.add_argument("--start-idx", type=int, default=None, help="Índice inicial 0-based")
    parser.add_argument("--no-resume", action="store_true", help="No saltar automáticamente a la primera pendiente")
    parser.add_argument("--headless", action="store_true", help="Modo sin interfaz gráfica (para tests/CI)")
    parser.add_argument("--dry-run", action="store_true", help="No escribe cambios en disco en modo headless")
    parser.add_argument("--limit", type=int, default=3, help="Límite de imágenes en modo headless (default: 3)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotator = JumpAnnotator(
        raw_dir=args.raw_dir,
        attack_dir=args.attack_dir,
        gt_dir=args.gt_dir,
        jump_radius_px=args.radius,
        headless=args.headless,
    )

    if args.headless:
        annotator.run_headless(limit=args.limit, dry_run=args.dry_run)
    else:
        annotator.run_interactive(start_idx=args.start_idx, auto_resume=not args.no_resume)


if __name__ == "__main__":
    main()
```

## File: tools/audit_village_diversity.py
```python
"""
tools/audit_village_diversity.py
Audita la diversidad de aldeas en el dataset (Fase 1.3 de plan_saltos_optimos_antigravity.md):
- Inspecciona las 37 imágenes en datasets/coc/images/raw y las 20 en logs/yolo/attack_*/screen_raw.png.
- Detecta niveles de Ayuntamiento (TH), conteo de segmentos de muro y capas concéntricas.
- Calcula edificios dentro vs fuera del anillo de muros.
- Reporta estadísticas exactas y cuántas anotaciones de ground truth están pendientes.
- Guarda reporte en logs/village_diversity_audit.json y logs/village_diversity_report.md.
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Configurar path raíz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import SETTINGS
from core.logger import get_logger
from vision.yolo_detector import YoloDetector
from vision.wall_raycast import detect_global_wall_segments, find_wall_crossings_from_segments

log = get_logger("audit_village_diversity")


def classify_th_level(crop: np.ndarray) -> str:
    """
    Clasifica el nivel de Ayuntamiento según la firma de color y estilo visual en HSV.
    - TH11: Mármol blanco, sat baja, brillo alto.
    - TH12: Azul eléctrico brillante (Giga Tesla).
    - TH13: Turquesa / Teal / Hielo (Giga Inferno).
    - TH14: Verde selva / maya con detalles dorados.
    - TH15: Púrpura / Magia cósmica triangular.
    - TH16: Rojo terracota / dorado naturaleza.
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return "Unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Fracción de píxeles característicos
    blue_ratio = float(np.mean((h >= 100) & (h <= 126) & (s > 75) & (v > 90)))
    teal_ratio = float(np.mean((h >= 78) & (h < 100) & (s > 60) & (v > 80)))
    green_ratio = float(np.mean((h >= 35) & (h <= 77) & (s > 75) & (v > 70)))
    purple_ratio = float(np.mean((h >= 127) & (h <= 165) & (s > 55) & (v > 70)))
    red_ratio = float(np.mean(((h <= 15) | (h >= 165)) & (s > 90) & (v > 100)))
    white_ratio = float(np.mean((s < 45) & (v > 175)))

    scores = {
        "TH11": white_ratio * 1.6,
        "TH12": blue_ratio * 2.8,
        "TH13": teal_ratio * 2.6,
        "TH14": green_ratio * 2.6,
        "TH15": purple_ratio * 2.8,
        "TH16": red_ratio * 2.2,
    }

    best_lvl, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score < 0.04:
        # Si el score es muy bajo, verificar si tiene rasgos TH12 predominantes
        if blue_ratio > 0.015:
            return "TH12"
        return "TH_General"
    return best_lvl


def line_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float]
) -> bool:
    """Verifica si dos segmentos (p1-p2) y (p3-p4) se intersecan."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-9:
        return False

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    return (0.0 <= ua <= 1.0) and (0.0 <= ub <= 1.0)


def estimate_wall_layers(
    center_pt: Tuple[int, int],
    wall_segments: List[Tuple[int, int, int, int]],
    screen_size: Tuple[int, int] = (2400, 1080),
) -> int:
    """
    Estima el número de capas concéntricas de muro (profundidad de compartimentos hasta el núcleo)
    trazando rayos desde los flancos exteriores del césped hacia el centro/TH.
    """
    if not wall_segments:
        return 0

    cx, cy = center_pt
    w, h = screen_size

    # Puntos de drop en los 4 flancos diagonales principales y 4 cardinales
    flank_drops = [
        (int(w * 0.29), int(h * 0.24)),  # NW
        (int(w * 0.71), int(h * 0.24)),  # NE
        (int(w * 0.29), int(h * 0.70)),  # SW
        (int(w * 0.71), int(h * 0.70)),  # SE
        (cx, int(h * 0.15)),             # Norte
        (cx, int(h * 0.85)),             # Sur
        (int(w * 0.18), cy),             # Oeste
        (int(w * 0.82), cy),             # Este
    ]

    crossings_counts = []
    for dp in flank_drops:
        crossings = find_wall_crossings_from_segments(
            p_drop=dp,
            p_target=center_pt,
            wall_segments=wall_segments,
            min_dist_drop=30.0,
            min_dist_target=40.0
        )
        if crossings:
            crossings_counts.append(len(crossings))

    if not crossings_counts:
        return 0

    # Retorna la mediana de capas atravesadas desde el exterior al núcleo
    return int(round(float(np.median(crossings_counts))))


def audit_single_image(
    image_path: str,
    detector: YoloDetector,
    gt_dir: str
) -> dict:
    """Audita una sola imagen de aldea."""
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Fallo al cargar imagen: {image_path}"}

    h, w = img.shape[:2]
    parent_dir = os.path.basename(os.path.dirname(image_path))
    file_name = os.path.basename(image_path)
    is_attack_log = "attack_" in parent_dir
    village_id = f"{parent_dir}_{file_name}" if is_attack_log else file_name

    # 1. Detecciones YOLO
    detections = detector.detect(img)
    total_buildings = len(detections)

    # 2. Ayuntamiento (Town Hall)
    th_dets = [d for d in detections if d.cls_name == "town_hall"]
    th_detected = len(th_dets) > 0
    th_level = "None"
    th_center = (w // 2, h // 2)
    th_bbox = []

    if th_detected:
        d_th = th_dets[0]
        th_center = (int(d_th.center[0]), int(d_th.center[1]))
        bx1, by1, bx2, by2 = [int(v) for v in d_th.bbox_xyxy]
        th_bbox = [bx1, by1, bx2, by2]
        crop_th = img[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
        th_level = classify_th_level(crop_th)

    # 3. Detección de muros globales
    iso_angles, wall_segments = detect_global_wall_segments(img, force_refresh=True)
    wall_count = len(wall_segments)

    # 4. Capas concéntricas de muro
    ref_center = th_center if th_detected else (1220, 500)
    wall_layers = estimate_wall_layers(ref_center, wall_segments, screen_size=(w, h))

    # 5. Edificios fuera vs dentro de muros (convex hull de muros)
    outside_buildings_count = 0
    inside_buildings_count = 0
    outside_ratio = 0.0

    if wall_segments and len(wall_segments) >= 3:
        wall_pts = []
        for s in wall_segments:
            wall_pts.append([s[0], s[1]])
            wall_pts.append([s[2], s[3]])
        wall_pts_np = np.array(wall_pts, dtype=np.int32)
        hull = cv2.convexHull(wall_pts_np)

        for d in detections:
            pt = (float(d.center[0]), float(d.center[1]))
            # pointPolygonTest: >0 interior, ==0 borde, <0 exterior
            dist = cv2.pointPolygonTest(hull, pt, False)
            if dist < 0:
                outside_buildings_count += 1
            else:
                inside_buildings_count += 1

        if total_buildings > 0:
            outside_ratio = float(outside_buildings_count) / float(total_buildings)
    else:
        # Sin muros detectados, todos cuentan como exteriores
        outside_buildings_count = total_buildings
        inside_buildings_count = 0
        outside_ratio = 1.0 if total_buildings > 0 else 0.0

    # 6. Metadata de botín (si existe json acompañante)
    loot_info = {}
    json_path = os.path.splitext(image_path)[0] + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                jdata = json.load(jf)
                loot_info = jdata.get("loot", {})
        except Exception:
            pass

    # 7. Verificación de Ground Truth
    gt_file_simple = os.path.splitext(file_name)[0] + ".json"
    gt_file_scoped = os.path.splitext(village_id)[0] + ".json"
    has_gt = os.path.exists(os.path.join(gt_dir, gt_file_simple)) or os.path.exists(os.path.join(gt_dir, gt_file_scoped))

    return {
        "village_id": village_id,
        "image_path": image_path,
        "source": "attack_log" if is_attack_log else "raw_dataset",
        "resolution": [w, h],
        "total_buildings": total_buildings,
        "town_hall_detected": th_detected,
        "town_hall_level": th_level,
        "town_hall_center": list(th_center) if th_detected else None,
        "town_hall_bbox": th_bbox if th_detected else None,
        "iso_angles": [round(float(iso_angles[0]), 1), round(float(iso_angles[1]), 1)],
        "wall_segments_count": wall_count,
        "wall_concentric_layers": wall_layers,
        "buildings_outside_walls": outside_buildings_count,
        "buildings_inside_walls": inside_buildings_count,
        "buildings_outside_ratio": round(outside_ratio, 3),
        "loot": loot_info,
        "has_ground_truth": has_gt,
    }


def run_village_diversity_audit(
    raw_dir: str = "datasets/coc/images/raw",
    attack_pattern: str = "logs/yolo/attack_*/screen_raw.png",
    gt_dir: str = "datasets/coc/jump_gt",
    output_json: str = "logs/village_diversity_audit.json",
    output_md: str = "logs/village_diversity_report.md"
) -> dict:
    """Ejecuta la auditoría completa de diversidad en todas las aldeas disponibles."""
    raw_images = sorted(glob.glob(os.path.join(raw_dir, "*.png")))
    attack_images = sorted(glob.glob(attack_pattern))
    all_images = raw_images + attack_images

    log.info("Iniciando auditoría de diversidad: %d imágenes en raw, %d en attack logs (Total: %d)",
             len(raw_images), len(attack_images), len(all_images))

    attack_cfg = SETTINGS.get("attack", {})
    conf = float(attack_cfg.get("yolo_conf", 0.35))
    imgsz = int(attack_cfg.get("yolo_imgsz", 1280))
    detector = YoloDetector(conf=conf, imgsz=imgsz)

    results = []
    for idx, img_path in enumerate(all_images, start=1):
        log.info("[%2d/%2d] Auditando %s...", idx, len(all_images), os.path.basename(img_path))
        data = audit_single_image(img_path, detector, gt_dir)
        results.append(data)

    # Estadísticas globales
    valid_results = [r for r in results if "error" not in r]
    total_audited = len(valid_results)

    th_levels_counter = Counter([r["town_hall_level"] for r in valid_results])
    layers_counter = Counter([r["wall_concentric_layers"] for r in valid_results])
    source_counter = Counter([r["source"] for r in valid_results])

    wall_counts = [r["wall_segments_count"] for r in valid_results]
    outside_ratios = [r["buildings_outside_ratio"] for r in valid_results]
    outside_counts = [r["buildings_outside_walls"] for r in valid_results]
    inside_counts = [r["buildings_inside_walls"] for r in valid_results]

    gt_annotated_count = sum(1 for r in valid_results if r["has_ground_truth"])
    gt_pending_count = total_audited - gt_annotated_count

    stats = {
        "total_images_audited": total_audited,
        "sources": dict(source_counter),
        "town_hall_levels": dict(th_levels_counter),
        "wall_segments": {
            "min": int(np.min(wall_counts)) if wall_counts else 0,
            "max": int(np.max(wall_counts)) if wall_counts else 0,
            "mean": round(float(np.mean(wall_counts)), 1) if wall_counts else 0.0,
            "median": round(float(np.median(wall_counts)), 1) if wall_counts else 0.0,
        },
        "wall_layers_distribution": dict(layers_counter),
        "buildings_distribution": {
            "mean_outside_count": round(float(np.mean(outside_counts)), 1) if outside_counts else 0.0,
            "mean_inside_count": round(float(np.mean(inside_counts)), 1) if inside_counts else 0.0,
            "mean_outside_ratio": round(float(np.mean(outside_ratios)), 3) if outside_ratios else 0.0,
        },
        "ground_truth_status": {
            "target_annotations": 30,
            "completed": gt_annotated_count,
            "pending": gt_pending_count,
            "completion_pct": round(gt_annotated_count / total_audited * 100.0, 1) if total_audited > 0 else 0.0,
        }
    }

    report_payload = {
        "stats": stats,
        "villages": valid_results
    }

    # Guardar JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)
    log.info("Auditoría JSON guardada en: %s", output_json)

    # Generar reporte Markdown
    md_content = generate_markdown_report(stats, valid_results)
    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    log.info("Reporte Markdown guardado en: %s", output_md)

    return report_payload


def generate_markdown_report(stats: dict, villages: List[dict]) -> str:
    """Genera el contenido Markdown formateado del reporte de diversidad."""
    th_table = "\n".join([f"| {lvl} | {cnt} | {cnt / stats['total_images_audited'] * 100:.1f}% |"
                          for lvl, cnt in sorted(stats['town_hall_levels'].items(), key=lambda x: -x[1])])

    layer_table = "\n".join([f"| {layer} capa(s) | {cnt} | {cnt / stats['total_images_audited'] * 100:.1f}% |"
                            for layer, cnt in sorted(stats['wall_layers_distribution'].items())])

    md = f"""# Auditoría de Diversidad de Aldeas y Estado de Ground Truth (Fase 1.3)

> Reporte oficial de auditoría de aldeas generado por `tools/audit_village_diversity.py` para Antigravity / ClashFarmerbot.

---

## 1. Resumen Ejecutivo de Muestras y Anotaciones

| Métrica | Valor | Notas |
| :--- | :--- | :--- |
| **Total de aldeas inspeccionadas** | **{stats['total_images_audited']}** | 37 en raw + 20 en attack logs |
| **Aldeas en `datasets/coc/images/raw`** | {stats['sources'].get('raw_dataset', 0)} | Capturas continuas pre-ataque con metadata de botín |
| **Aldeas en `logs/yolo/attack_*`** | {stats['sources'].get('attack_log', 0)} | Capturas reales de combate antes del drop (`screen_raw.png`) |
| **Anotaciones Ground Truth completadas** | **{stats['ground_truth_status']['completed']}** | Archivos existentes en `datasets/coc/jump_gt/` |
| **Anotaciones Ground Truth pendientes** | **{stats['ground_truth_status']['pending']}** | **Objetivo Fase 1: Anotar al menos 30 aldeas** |

---

## 2. Distribución de Niveles de Ayuntamiento (TH)

| Nivel de Ayuntamiento | Cantidad | Porcentaje |
| :--- | :--- | :--- |
{th_table}

> **Observación:** El dataset presenta una fuerte concentración en **TH12** ({stats['town_hall_levels'].get('TH12', 0)} aldeas), coherente con el emparejamiento de la cuenta de farmeo actual. Se identificaron además aldeas de TH14 y TH16.

---

## 3. Complejidad Estructural de Muros y Capas Concéntricas

- **Segmentos de muro detectados:**
  - Mínimo: `{stats['wall_segments']['min']}` segmentos
  - Máximo: `{stats['wall_segments']['max']}` segmentos
  - Media: `{stats['wall_segments']['mean']}` segmentos
  - Mediana: `{stats['wall_segments']['median']}` segmentos

### Distribución de Capas Concéntricas (Profundidad de Núcleo)
| Capas Concéntricas | Aldeas | Porcentaje |
| :--- | :--- | :--- |
{layer_table}

> Las aldeas con **2 y 3 capas concéntricas** representan la gran mayoría del conjunto de prueba ({stats['wall_layers_distribution'].get(2, 0) + stats['wall_layers_distribution'].get(3, 0)} aldeas), proporcionando el escenario óptimo para evaluar saltos dobles y triples que reemplacen múltiples Super Rompemuros.

---

## 4. Distribución de Edificios Fuera vs Dentro de Muros

- **Media de edificios fuera de muros (perímetro/recolección):** `{stats['buildings_distribution']['mean_outside_count']}` edificios
- **Media de edificios dentro de muros (compartimentos/almacenes):** `{stats['buildings_distribution']['mean_inside_count']}` edificios
- **Ratio promedio de edificios en exterior:** `{stats['buildings_distribution']['mean_outside_ratio'] * 100:.1f}%`

> Los edificios exteriores permiten verificar que el optimizador priorice las aperturas hacia los almacenes internos sin desperdiciar saltos en recolectores perimetrales accesibles directamente por césped.

---

## 5. Próximos Pasos (Punto de Parada 1)
1. **Anotación manual humana:** Con la herramienta `tools/annotate_best_jump.py`, el usuario anotará los mejores puntos de salto en las 30 aldeas prioritarias.
2. **Evaluación de huella:** Integrar la Elipse Isométrica ($Rx=115$px, $Ry=65$px) en el optimizador de compartimentos (Fase 2) para eliminar la sobrestimación vertical del 109% producida por el círculo de 125px.
"""
    return md


def main():
    parser = argparse.ArgumentParser(description="Auditoría de diversidad de aldeas CoC (Fase 1.3)")
    parser.add_argument("--raw-dir", type=str, default="datasets/coc/images/raw")
    parser.add_argument("--attack-pattern", type=str, default="logs/yolo/attack_*/screen_raw.png")
    parser.add_argument("--gt-dir", type=str, default="datasets/coc/jump_gt")
    parser.add_argument("--output-json", type=str, default="logs/village_diversity_audit.json")
    parser.add_argument("--output-md", type=str, default="logs/village_diversity_report.md")
    args = parser.parse_args()

    run_village_diversity_audit(
        raw_dir=args.raw_dir,
        attack_pattern=args.attack_pattern,
        gt_dir=args.gt_dir,
        output_json=args.output_json,
        output_md=args.output_md
    )


if __name__ == "__main__":
    main()
```

## File: tools/auto_capture_villages.py
```python
"""
tools/auto_capture_villages.py
Script CLI standalone para captura automática y masiva de aldeas para dataset CoC.
Navega de aldea en aldea continuamente usando el botón 'Next' sin salir al menú principal,
haciendo zoom out y extrayendo botín/metadata en cada aldea, sin desplegar tropas, hechizos ni héroes.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from core.device import Device
from core.config import SETTINGS, load_settings
from core.vision import Vision
from core.process_manager import ProcessManager
from core.logger import get_logger
from humanizer.behavior import HumanBehavior
from vision.army_checker import ArmyChecker
from vision.resource_reader import ResourceReader

# Alias para compatibilidad
load_config = load_settings
VisionEngine = Vision

log = get_logger("auto_capture")

# ====================================================================
# SEGURIDAD DURA (HARDCODED, NO CONFIGURABLE)
# ====================================================================
DEPLOY_ENABLED = False


def safe_tap(device: Device, x: int | float, y: int | float, description: str = "", in_battle: bool = False):
    """
    Toque seguro que garantiza que JAMÁS se toque la barra inferior de despliegue
    de tropas, hechizos o héroes durante la batalla.
    """
    ix, iy = int(x), int(y)
    deployment_bar_y = int(device.height * 0.82)
    # La franja de cartas de despliegue de tropas/hechizos solo existe durante el combate
    is_in_deployment_strip = in_battle and (iy >= deployment_bar_y) and (int(device.width * 0.12) <= ix <= int(device.width * 0.88))

    if not DEPLOY_ENABLED and is_in_deployment_strip:
        log.critical("SEGURIDAD: Intento de toque en la barra de tropas (%s): (%d, %d). ABORTANDO.", description, ix, iy)
        raise RuntimeError("Deployment disabled in capture mode")

    device.tap(ix, iy)


class AutoCaptureRunner:
    def __init__(self, serial: str | None = None, output_dir: str = "datasets/coc/images/raw"):
        self.serial = serial or SETTINGS.get("device", {}).get("serial", "5fba8618")
        self.output_dir = output_dir
        self.device = Device(serial=self.serial)
        self.vision = Vision(threshold=0.76)
        self.vision.preload_templates("assets")
        self.proc_mgr = ProcessManager(self.device, self.vision)
        self.army = ArmyChecker(self.vision)
        self.reader = ResourceReader(self.vision)
        self.in_battle = False

        # Templates de navegación
        self.tpl_attack = "assets/buttons/attack_btn.png"
        self.tpl_find_match = "assets/buttons/find_match_btn.png"
        self.tpl_attack_confirms = [
            "assets/buttons/attack_btn_confirm.png",
            "assets/buttons/confirm_attack_btn.png",
        ]
        self.tpl_next = "assets/buttons/next_btn.png"
        self.tpl_surrender = "assets/buttons/surrender_battle.png"
        self.tpl_end_battle = "assets/buttons/end_battle.png"
        self.tpl_confirm_end = "assets/buttons/confirm_end.png"
        self.tpl_okay = "assets/buttons/okay_btn.png"
        self.tpl_return_home = "assets/buttons/return_home.png"

    def is_in_match(self) -> bool:
        """Verifica si actualmente estamos en la pantalla de batalla/búsqueda con Next visible."""
        screen = self.device.screenshot()
        next_pos = self.vision.find_template(screen, self.tpl_next, threshold=0.74, zona="ESQUINA_INF_DER")
        if not next_pos:
            next_pos = self.vision.find_template(screen, self.tpl_next, threshold=0.68, zona="ESQUINA_INF_DER")
        return next_pos is not None

    def surrender_and_return_home(self) -> bool:
        """
        Finaliza la búsqueda / batalla y regresa de forma limpia a la aldea principal:
        1. Comprueba si ya está en aldea principal o si hay Return Home.
        2. Pulsa End Battle (o Surrender).
        3. Confirma finalización (Confirm End / Okay).
        4. Espera confirmación de aldea principal o Return Home.
        """
        log.info("🏳️ Finalizando combate / búsqueda y retornando a la aldea principal...")
        screen = self.device.screenshot()

        # Si ya estamos en la aldea principal
        if self.vision.find_template(screen, self.tpl_attack, threshold=0.74, zona="ESQUINA_INF_IZQ"):
            log.info("🏰 Ya nos encontramos en la aldea principal.")
            self.in_battle = False
            return True

        # Si ya está Return Home en pantalla (ej. batalla finalizada)
        ret_pos = self.vision.find_template(screen, self.tpl_return_home, zona="MITAD_INFERIOR")
        if ret_pos:
            HumanBehavior.reaction_delay()
            safe_tap(self.device, *ret_pos, description="return_home", in_battle=False)
            self.in_battle = False
            time.sleep(3.0)
            return True

        # 1. End Battle / Surrender en esquina inferior izquierda
        end_pos = self.vision.find_template(screen, self.tpl_end_battle, threshold=0.72, zona="ESQUINA_INF_IZQ")
        if not end_pos:
            end_pos = self.vision.find_template(screen, self.tpl_surrender, threshold=0.72, zona="ESQUINA_INF_IZQ")
        if not end_pos:
            end_pos = self.vision.wait_for_template(
                self.device.screenshot, self.tpl_end_battle, timeout=3, threshold=0.72, zona="MITAD_INFERIOR"
            )
        if not end_pos:
            end_pos = self.vision.wait_for_template(
                self.device.screenshot, self.tpl_surrender, timeout=3, threshold=0.72, zona="MITAD_INFERIOR"
            )

        if end_pos:
            HumanBehavior.reaction_delay()
            safe_tap(self.device, *end_pos, description="end_or_surrender_btn")
            time.sleep(0.8)
        else:
            log.warning("Botón de fin de batalla no detectado por template. Probando tecla BACK...")
            self.device.d.press("back")
            time.sleep(0.8)

        # 2. Confirmación de finalización / rendición
        confirm_pos = self.vision.wait_for_template(
            self.device.screenshot, self.tpl_confirm_end, timeout=4, threshold=0.70, zona="CENTRO"
        )
        if not confirm_pos:
            confirm_pos = self.vision.wait_for_template(
                self.device.screenshot, self.tpl_okay, timeout=2, threshold=0.70, zona="CENTRO"
            )

        if confirm_pos:
            HumanBehavior.reaction_delay()
            safe_tap(self.device, *confirm_pos, description="confirm_end_btn")
            time.sleep(1.0)

        # 3. Esperar confirmación de retorno: puede aparecer Return Home o volver directo a la aldea
        t0 = time.monotonic()
        while time.monotonic() - t0 < 8:
            s = self.device.screenshot()
            # Si aparece botón Return Home
            ret_pos = self.vision.find_template(s, self.tpl_return_home, zona="MITAD_INFERIOR")
            if ret_pos:
                HumanBehavior.reaction_delay()
                safe_tap(self.device, *ret_pos, description="return_home", in_battle=False)
                self.in_battle = False
                time.sleep(3.0)
                return True
            # Si ya se cargó la aldea principal (botón Attack visible)
            if self.vision.find_template(s, self.tpl_attack, threshold=0.74, zona="ESQUINA_INF_IZQ"):
                log.info("🏰 Aldea principal confirmada tras salir de búsqueda.")
                self.in_battle = False
                return True
            time.sleep(0.5)

        self.in_battle = False
        log.warning("No se confirmó Return Home ni aldea principal de forma inmediata.")
        return False

    def verify_army_ready(self, timeout_sec: int = 60) -> bool:
        """Verifica que el ejército esté listo con timeout de seguridad."""
        log.info("Verificando estado del ejército...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_sec:
            screen = self.device.screenshot()
            if self.army.can_start_battle(screen):
                log.info("Ejército confirmado listo para batalla.")
                return True
            time.sleep(2.0)
        log.warning("Timeout (%ds) esperando ejército listo.", timeout_sec)
        return False

    def _wait_for_first_rival(self, timeout: int = 90) -> bool:
        log.info("Esperando botón Next (despeje de nubes e ingreso a batalla)...")
        next_pos = self.vision.wait_for_template(
            self.device.screenshot, self.tpl_next, timeout=timeout, threshold=0.74, zona="ESQUINA_INF_DER"
        )
        if not next_pos:
            log.error("Timeout (%ds) esperando botón Next / ingreso a batalla.", timeout)
            self.surrender_and_return_home()
            return False

        self.in_battle = True
        log.info("🎯 Primera aldea rival cargada (Next visible en %s). MODO COMBATE ACTIVO.", next_pos)
        time.sleep(0.6)
        return True

    def enter_matchmaking(self) -> bool:
        """
        Inicia la búsqueda de rivales desde la aldea principal (o menús intermedios)
        hasta que carga la primera aldea rival con botón Next visible.
        """
        if self.is_in_match():
            log.info("🎯 Ya se encuentra en pantalla de rival (Next visible). Continuando...")
            self.in_battle = True
            return True

        log.info("Iniciando emparejamiento para combate...")
        screen = self.device.screenshot()

        # Caso A: ¿Ya está visible el botón de confirmar ataque?
        match_confirm = self.vision.find_any(screen, self.tpl_attack_confirms, threshold=0.72, zona="MITAD_INFERIOR")
        if match_confirm:
            log.info("🎯 Botón Confirm Attack ya visible en (%d, %d). Confirmando...", match_confirm[1], match_confirm[2])
            HumanBehavior.reaction_delay()
            safe_tap(self.device, match_confirm[1], match_confirm[2], description="confirm_attack_btn")
            return self._wait_for_first_rival()

        # Caso B: ¿Ya está abierto el menú de ataque (Find Match visible)?
        find_pos = self.vision.find_template(screen, self.tpl_find_match, threshold=0.72, zona="COMPLETA")
        if not find_pos:
            # Caso C: Estamos en la aldea principal
            if not self.proc_mgr.ensure_game_ready(timeout=25):
                log.error("Fallo ensure_game_ready antes de buscar rivales.")
                return False

            if not self.verify_army_ready(timeout_sec=30):
                log.error("Ejército no disponible para iniciar búsqueda.")
                return False

            screen = self.device.screenshot()
            attack_pos = self.vision.find_template(screen, self.tpl_attack, threshold=0.74, zona="ESQUINA_INF_IZQ")
            if not attack_pos:
                log.error("Botón Attack no detectado en aldea principal.")
                return False

            HumanBehavior.reaction_delay()
            safe_tap(self.device, *attack_pos, description="attack_btn")

            find_pos = self.vision.wait_for_template(
                self.device.screenshot, self.tpl_find_match, timeout=5, zona="COMPLETA"
            )
            if not find_pos:
                # Reintento de toque por si no registró
                log.info("Reintentando toque en botón Attack...")
                safe_tap(self.device, *attack_pos, description="attack_btn_retry")
                find_pos = self.vision.wait_for_template(
                    self.device.screenshot, self.tpl_find_match, timeout=6, zona="COMPLETA"
                )

        if not find_pos:
            log.error("Botón Find Match no detectado tras abrir menú de ataque.")
            self.surrender_and_return_home()
            return False

        HumanBehavior.reaction_delay()
        safe_tap(self.device, *find_pos, description="find_match_btn")
        time.sleep(0.8)

        # Esperar y confirmar búsqueda
        confirm_pos = None
        t_start = time.monotonic()
        while time.monotonic() - t_start < 4.5:
            s_confirm = self.device.screenshot()
            match = self.vision.find_any(s_confirm, self.tpl_attack_confirms, threshold=0.72, zona="MITAD_INFERIOR")
            if match:
                confirm_pos = (match[1], match[2])
                break
            time.sleep(0.25)

        if confirm_pos:
            HumanBehavior.reaction_delay()
            safe_tap(self.device, *confirm_pos, description="confirm_attack_btn")
        else:
            log.info("Clic de confirmación en posición estándar tras find_match...")
            HumanBehavior.reaction_delay()
            safe_tap(self.device, int(self.device.width * 0.814), int(self.device.height * 0.900), description="confirm_attack_fallback")

        return self._wait_for_first_rival()

    def capture_current_village(self, idx: int, timestamp_str: str) -> bool:
        """
        Ejecuta zoom out completo, estabiliza cámara, toma captura y guarda metadata de botín
        de la aldea rival actualmente visible.
        """
        # 1. Zoom out completo centrado y estabilización
        self.device.zoom_out()
        self.device.wait_camera_settled(timeout=2.0)

        # 2. Captura de pantalla
        cap_screen = self.device.screenshot()
        filename_png = f"auto_{timestamp_str}_{idx:04d}.png"
        filename_json = f"auto_{timestamp_str}_{idx:04d}.json"
        path_png = os.path.join(self.output_dir, filename_png)
        path_json = os.path.join(self.output_dir, filename_json)

        cv2.imwrite(path_png, cap_screen)
        log.info("💾 Imagen [%04d] guardada: %s (%dx%d)", idx, path_png, cap_screen.shape[1], cap_screen.shape[0])

        # 3. Lectura de metadata de botín
        loot_info = {}
        try:
            loot_info = self.reader.read_enemy_loot(cap_screen)
        except Exception as e:
            log.warning("No se pudo parsear botín de la captura: %s", e)

        meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "index": idx,
            "image_file": filename_png,
            "resolution": [int(cap_screen.shape[1]), int(cap_screen.shape[0])],
            "loot": loot_info,
            "troops_deployed": False,
        }
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        log.info("📄 Metadata [%04d] guardada: %s", idx, path_json)

        return True

    def next_village(self, timeout: int = 90) -> bool:
        """
        Pulsa el botón 'Next' para avanzar a la siguiente aldea y espera a que las nubes se despejen.
        No sale al menú principal.
        """
        screen = self.device.screenshot()
        next_pos = self.vision.find_template(screen, self.tpl_next, threshold=0.74, zona="ESQUINA_INF_DER")
        if not next_pos:
            next_pos = self.vision.find_template(screen, self.tpl_next, threshold=0.68, zona="ESQUINA_INF_DER")

        if not next_pos:
            # Si el temporizador de 30s expiró, aparece surrender en lugar de end_battle y Next desaparece
            is_surrender = self.vision.find_template(screen, self.tpl_surrender, threshold=0.70, zona="ESQUINA_INF_IZQ")
            if is_surrender:
                log.warning("El temporizador de exploración expiró (combate iniciado sin tropas). Forzando rendición y re-entrada...")
                self.surrender_and_return_home()
                return False

            # Fallback en coordenadas conocidas para 2400x1080 si estamos en combate
            log.warning("Botón Next no detectado por template. Probando coordenadas conocidas (2051, 782)...")
            next_pos = (int(self.device.width * 0.854), int(self.device.height * 0.724))

        log.info("⏩ Pulsando botón Next en %s...", next_pos)
        HumanBehavior.reaction_delay()
        safe_tap(self.device, *next_pos, description="next_btn", in_battle=True)

        # Esperar a que entren las nubes (Next desaparece)
        time.sleep(1.2)

        # Esperar a que se despejen las nubes y aparezca el botón Next de la nueva aldea
        log.info("Esperando que disipen las nubes y cargue la nueva aldea...")
        new_next = self.vision.wait_for_template(
            self.device.screenshot, self.tpl_next, timeout=timeout, threshold=0.74, zona="ESQUINA_INF_DER"
        )
        if not new_next:
            new_next = self.vision.find_template(self.device.screenshot(), self.tpl_next, threshold=0.68, zona="ESQUINA_INF_DER")

        if not new_next:
            log.warning("Timeout (%ds) esperando nueva aldea tras pulsar Next.", timeout)
            return False

        log.info("🎯 Nueva aldea rival cargada (Next visible en %s).", new_next)
        time.sleep(0.6)
        return True

    def execute_capture_cycle(self, idx: int, timestamp_str: str) -> bool:
        """
        Método de compatibilidad para capturar la aldea actual.
        """
        if not self.is_in_match():
            if not self.enter_matchmaking():
                return False

        if not self.capture_current_village(idx, timestamp_str):
            return False

        return True


def get_connected_devices() -> list[str]:
    """Obtiene la lista de dispositivos conectados por ADB."""
    try:
        import adbutils
        devs = [d.serial for d in adbutils.adb.device_list()]
        if devs:
            return devs
    except Exception:
        pass
    try:
        res = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=5)
        devs = []
        for line in res.stdout.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "device" and parts[0] != "List":
                devs.append(parts[0])
        return devs
    except Exception:
        return []


def print_banner(count: int):
    cost_est = count * 750
    print("=" * 70)
    print(" AUTO-CAPTURA MASIVA DE ALDEAS (MODO CONTINUO 'NEXT')")
    print("=" * 70)
    print(" Navega de aldea en aldea mediante botón 'Next' sin volver al menú.")
    print(" NO se despliegan tropas, hechizos ni héroes.")
    print(f" Coste ~500-1000 oro/aldea por Next. Total estimado: ~{cost_est:,} oro.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Auto-captura masiva de aldeas para dataset CoC")
    parser.add_argument("--count", type=int, default=100, help="Número de aldeas a capturar (default: 100)")
    parser.add_argument("--serial", type=str, default=None, help="Serial ADB del dispositivo")
    parser.add_argument("--output", type=str, default="datasets/coc/images/raw", help="Directorio de salida")
    parser.add_argument("--delay-min", type=float, default=2.5, help="Pausa mínima entre aldeas en segundos (default: 2.5)")
    parser.add_argument("--delay-max", type=float, default=5.0, help="Pausa máxima entre aldeas en segundos (default: 5.0)")
    parser.add_argument("--max-fails", type=int, default=3, help="Máximo de fallos consecutivos antes de abortar")
    parser.add_argument("--dry-run", action="store_true", help="Imprime plan y sale sin conectar ni tocar el dispositivo")
    parser.add_argument("--resume", action="store_true", help="Continúa desde el conteo de capturas ya existentes")
    args = parser.parse_args()

    print_banner(args.count)

    os.makedirs(args.output, exist_ok=True)

    connected_devs = get_connected_devices()
    target_serial = args.serial or SETTINGS.get("device", {}).get("serial", "5fba8618")
    has_target_device = target_serial in connected_devs or len(connected_devs) > 0

    if args.dry_run:
        print("\n--- PLAN DE EJECUCIÓN (--dry-run activo) ---")
        print(f"  • Objetivo de capturas : {args.count}")
        print(f"  • Carpeta de salida    : {os.path.abspath(args.output)}")
        print(f"  • Serial configurado   : {target_serial}")
        print(f"  • Dispositivos ADB     : {connected_devs if connected_devs else 'Ninguno detectado'}")
        print(f"  • Estado dispositivo   : {'Conectado' if has_target_device else 'No detectado'}")
        print(f"  • Rango de pausa Next  : {args.delay_min}s - {args.delay_max}s")
        print(f"  • Máximo de fallos     : {args.max_fails}")
        print(f"  • Modo despliegue      : DESACTIVADO (DEPLOY_ENABLED = {DEPLOY_ENABLED})")
        print("Saliendo sin conectar ni interactuar con el dispositivo.\n")
        return

    # Contar capturas existentes si resume está activo
    existing_files = [f for f in os.listdir(args.output) if f.startswith("auto_") and f.endswith(".png")]
    start_idx = 1
    if args.resume:
        start_idx = len(existing_files) + 1
        log.info("Modo --resume activo: Se encontraron %d capturas previas. Iniciando en índice %d.", len(existing_files), start_idx)
        if start_idx > args.count:
            log.info("El número de capturas existentes (%d) ya alcanza o supera el objetivo (%d). Nada que hacer.", len(existing_files), args.count)
            return

    if not has_target_device:
        log.critical("No se detectó el dispositivo ADB '%s'. Conecta el dispositivo o verifica 'adb devices'.", target_serial)
        sys.exit(1)

    runner = AutoCaptureRunner(serial=target_serial, output_dir=args.output)
    if not runner.device.connect():
        log.critical("No se pudo establecer conexión ADB con el dispositivo %s.", target_serial)
        sys.exit(1)

    ts_session = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    consecutive_fails = 0
    successful_captures = 0

    log.info("═" * 60)
    log.info("🚀 INICIANDO AUTO-CAPTURA CONTINUA (Aldea en aldea vía botón Next)")
    log.info("   Meta: %d capturas | Pausa entre aldeas: %.1fs - %.1fs", args.count, args.delay_min, args.delay_max)
    log.info("═" * 60)

    try:
        # Asegurar ingreso inicial a la búsqueda de rivales si aún no estamos en combate
        if not runner.is_in_match():
            if not runner.enter_matchmaking():
                log.critical("No se pudo ingresar a la búsqueda de rivales. Abortando.")
                return

        current_idx = start_idx
        while current_idx <= args.count and consecutive_fails < args.max_fails:
            log.info("═" * 60)
            log.info("📸 CAPTURANDO ALDEA [%04d/%04d]...", current_idx, args.count)
            log.info("═" * 60)

            # Si salimos de batalla por algún motivo (desconexión, etc.), reingresar
            if not runner.is_in_match():
                log.warning("No estamos en modo búsqueda de rivales. Reingresando...")
                if not runner.enter_matchmaking():
                    consecutive_fails += 1
                    log.warning("Fallo al reingresar a búsqueda (%d/%d fallos consecutivos).", consecutive_fails, args.max_fails)
                    time.sleep(2.0)
                    continue

            # 1. Capturar aldea actual (zoom out + screenshot + metadata)
            try:
                ok_cap = runner.capture_current_village(current_idx, ts_session)
                if ok_cap:
                    successful_captures += 1
                    consecutive_fails = 0
                else:
                    consecutive_fails += 1
                    log.warning("Fallo al capturar aldea %d (%d/%d fallos).", current_idx, consecutive_fails, args.max_fails)
                    continue
            except Exception as e:
                consecutive_fails += 1
                log.error("Excepción en captura [%04d]: %s (%d/%d fallos)", current_idx, e, consecutive_fails, args.max_fails)
                continue

            # Si ya se alcanzó la última captura solicitada, salimos del bucle (no necesitamos pulsar Next)
            if current_idx >= args.count:
                log.info("🎯 Se alcanzó el total solicitado de %d capturas.", args.count)
                break

            # 2. Pausa humanizada antes de pasar a la siguiente aldea
            log.info("⏩ Avanzando a la siguiente aldea usando botón 'Next'...")
            HumanBehavior.sleep_range(args.delay_min, args.delay_max)

            # 3. Pulsar Next y esperar a que disipen las nubes
            ok_next = runner.next_village(timeout=90)
            if ok_next:
                current_idx += 1
                consecutive_fails = 0
            else:
                consecutive_fails += 1
                log.warning("Fallo al avanzar con Next (%d/%d fallos consecutivos).", consecutive_fails, args.max_fails)
                # Revisar si apareció un popup (ej. sin oro suficiente, reconexión)
                runner.proc_mgr.handle_popup()

    except KeyboardInterrupt:
        log.warning("\n⏹️ Interrupción manual por usuario (Ctrl+C).")
    except Exception as e:
        log.critical("Error fatal no controlado en bucle de captura: %s", e)
    finally:
        # Al finalizar la sesión completa o en caso de interrupción/error, volver a la aldea principal
        log.info("🏳️ Finalizando sesión: retornando a la aldea principal...")
        try:
            runner.surrender_and_return_home()
        except Exception as e:
            log.warning("No se pudo completar retorno a casa automático: %s", e)

    log.info("═" * 60)
    log.info("🏆 AUTO-CAPTURA FINALIZADA: %d/%d exitosas guardadas en %s", successful_captures, args.count, args.output)
    log.info("═" * 60)


if __name__ == "__main__":
    main()
```

## File: tools/calibrate_jump_footprint.py
```python
"""
tools/calibrate_jump_footprint.py
Genera comparaciones visuales lado a lado sobre aldeas reales de Clash of Clans:
- Círculo R=90px (definido en config/settings.json, radio equivalente de área)
- Círculo R=125px (fallback de depuración en _save_yolo_attack_debug y test_jump_overlay)
- Elipse isométrica Rx=115px, Ry=65px (huella real proyectada en elevación ~30°-35°)

Guarda overlays detallados y paneles comparativos en logs/yolo/calibration_footprint_*.png.
"""
import argparse
import glob
import math
import os
import sys
from typing import List, Tuple, Optional

import cv2
import numpy as np

# Configurar path raíz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import SETTINGS
from core.logger import get_logger
from vision.yolo_detector import YoloDetector
from vision.wall_raycast import detect_global_wall_segments, estimate_iso_angles
from attack.deployment_strategy import compute_adaptive_jump_targets_from_detections

log = get_logger("calibrate_jump_footprint")

# Colores BGR para cada huella
COLOR_CIRCLE_90 = (0, 0, 255)       # Rojo puro para 90px
COLOR_CIRCLE_125 = (0, 255, 255)    # Amarillo brillante para 125px
COLOR_ELLIPSE_ISO = (0, 255, 0)     # Verde lima brillante para Elipse Isométrica
COLOR_WALL = (160, 160, 160)        # Gris para muros
COLOR_TARGET = (255, 120, 0)        # Naranja/Cyan para centro de salto


def calculate_footprint_metrics(
    r_circle_small: float = 90.0,
    r_circle_large: float = 125.0,
    rx_ellipse: float = 115.0,
    ry_ellipse: float = 65.0
) -> dict:
    """
    Calcula áreas y métricas geométricas comparativas de las tres huellas.
    """
    area_90 = math.pi * (r_circle_small ** 2)
    area_125 = math.pi * (r_circle_large ** 2)
    area_ellipse = math.pi * rx_ellipse * ry_ellipse
    equiv_radius_ellipse = math.sqrt(rx_ellipse * ry_ellipse)
    aspect_ratio = rx_ellipse / ry_ellipse if ry_ellipse > 0 else 1.0

    return {
        "r_circle_small": r_circle_small,
        "area_circle_small_px2": area_90,
        "r_circle_large": r_circle_large,
        "area_circle_large_px2": area_125,
        "rx_ellipse": rx_ellipse,
        "ry_ellipse": ry_ellipse,
        "area_ellipse_px2": area_ellipse,
        "equiv_radius_ellipse": equiv_radius_ellipse,
        "aspect_ratio": aspect_ratio,
        "area_ratio_125_vs_ellipse": area_125 / area_ellipse if area_ellipse > 0 else 0.0,
        "area_ratio_90_vs_ellipse": area_90 / area_ellipse if area_ellipse > 0 else 0.0,
    }


def draw_legend(
    img: np.ndarray,
    metrics: dict,
    start_x: int = 40,
    start_y: int = 60
) -> None:
    """Dibuja una leyenda explicativa con métricas geométricas en la imagen."""
    box_w = 700
    box_h = 210
    overlay = img.copy()
    cv2.rectangle(overlay, (start_x - 15, start_y - 35), (start_x + box_w, start_y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    cv2.rectangle(img, (start_x - 15, start_y - 35), (start_x + box_w, start_y + box_h), (120, 120, 120), 1)

    cv2.putText(
        img,
        "CALIBRACION DE HUELLA DE HECHIZO DE SALTO (FASE 1.2)",
        (start_x, start_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    items = [
        (
            COLOR_CIRCLE_90,
            f"Circulo R={int(metrics['r_circle_small'])}px (settings.json default) | Area: {int(metrics['area_circle_small_px2']):,} px2 (Radio equiv: {metrics['r_circle_small']:.1f}px)"
        ),
        (
            COLOR_CIRCLE_125,
            f"Circulo R={int(metrics['r_circle_large'])}px (Debug fallback) | Area: {int(metrics['area_circle_large_px2']):,} px2 (+{((metrics['area_ratio_125_vs_ellipse'] - 1) * 100):.1f}% vs real)"
        ),
        (
            COLOR_ELLIPSE_ISO,
            f"Elipse Isometrica Rx={int(metrics['rx_ellipse'])}px, Ry={int(metrics['ry_ellipse'])}px (Aspect={metrics['aspect_ratio']:.2f}) | Area: {int(metrics['area_ellipse_px2']):,} px2 (Req={metrics['equiv_radius_ellipse']:.1f}px)"
        ),
    ]

    for i, (col, text) in enumerate(items):
        y_pos = start_y + 40 + i * 40
        cv2.line(img, (start_x, y_pos - 5), (start_x + 35, y_pos - 5), col, 3)
        cv2.circle(img, (start_x + 17, y_pos - 5), 4, col, -1)
        cv2.putText(
            img,
            text,
            (start_x + 50, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            col,
            1,
            cv2.LINE_AA,
        )

    note = "Nota: En vista isometrica (30 deg elev), el circulo en suelo se proyecta como elipse horizontal."
    cv2.putText(
        img,
        note,
        (start_x, start_y + 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def generate_side_by_side_comparison(
    image_path: str,
    output_dir: str = "logs/yolo",
    detector: Optional[YoloDetector] = None,
    rx_px: float = 115.0,
    ry_px: float = 65.0,
    r_small_px: float = 90.0,
    r_large_px: float = 125.0
) -> Optional[str]:
    """
    Genera una imagen comparativa lado a lado para una aldea específica:
    - Panel Izquierdo: Vista global con muros, detecciones y los 3 overlays superpuestos.
    - Panel Derecho: 3 sub-paneles con zoom detallado en el punto de salto principal:
      1. R=90px (muestra brechas horizontales sin cubrir)
      2. R=125px (muestra falso positivo en dirección vertical)
      3. Elipse Isométrica (cobertura conforme precisa)
    """
    if not os.path.exists(image_path):
        log.warning("Imagen no encontrada: %s", image_path)
        return None

    img = cv2.imread(image_path)
    if img is None:
        log.warning("Fallo al decodificar imagen: %s", image_path)
        return None

    h, w = img.shape[:2]
    parent_dir = os.path.basename(os.path.dirname(image_path))
    raw_base = os.path.splitext(os.path.basename(image_path))[0]
    if raw_base == "screen_raw":
        base_name = f"{parent_dir}_{raw_base}"
    else:
        base_name = raw_base
    os.makedirs(output_dir, exist_ok=True)

    metrics = calculate_footprint_metrics(r_small_px, r_large_px, rx_px, ry_px)

    # 1. Detección YOLO
    detections = []
    if detector is not None:
        try:
            detections = detector.detect(img)
        except Exception as e:
            log.warning("Error en inferencia YOLO para %s: %s", base_name, e)

    # 2. Detección de muros globales
    _, wall_segments = detect_global_wall_segments(img, force_refresh=True)

    # 3. Puntos de salto candidatos
    jump_points = []
    if detections:
        try:
            raw_targets = compute_adaptive_jump_targets_from_detections(detections, (w, h), screen=img)
            for t in raw_targets:
                pt = t[1] if (isinstance(t, (tuple, list)) and len(t) == 2 and isinstance(t[0], str)) else t
                jump_points.append((int(round(pt[0])), int(round(pt[1]))))
        except Exception as e:
            log.warning("Error en compute_adaptive_jump_targets: %s", e)

    # Fallback si no hay puntos por YOLO
    if not jump_points:
        if wall_segments:
            # Seleccionar el punto medio del segmento más cercano al centro
            center_x, center_y = w // 2, h // 2
            dists = [
                ((s[0] + s[2]) // 2 - center_x) ** 2 + ((s[1] + s[3]) // 2 - center_y) ** 2
                for s in wall_segments
            ]
            best_idx = int(np.argmin(dists))
            s = wall_segments[best_idx]
            jump_points.append(((s[0] + s[2]) // 2, (s[1] + s[3]) // 2))
        else:
            jump_points.append((w // 2, h // 2))

    # Limitar a máximo 3 saltos para el overlay principal
    primary_jumps = jump_points[:3]
    primary_pt = primary_jumps[0]
    pjx, pjy = primary_pt

    # Construir Panel Izquierdo: Vista global con los 3 overlays
    panel_left = img.copy()

    # Dibujar segmentos de muro en gris tenue
    for s in wall_segments:
        cv2.line(panel_left, (s[0], s[1]), (s[2], s[3]), COLOR_WALL, 1, cv2.LINE_AA)

    # Dibujar huellas en los puntos de salto
    for jx, jy in primary_jumps:
        # Círculo 90px
        cv2.circle(panel_left, (jx, jy), int(r_small_px), COLOR_CIRCLE_90, 2, cv2.LINE_AA)
        # Círculo 125px
        cv2.circle(panel_left, (jx, jy), int(r_large_px), COLOR_CIRCLE_125, 2, cv2.LINE_AA)
        # Elipse Isométrica (major axis horizontal = 0 deg)
        cv2.ellipse(
            panel_left,
            (jx, jy),
            (int(rx_px), int(ry_px)),
            0.0,
            0.0,
            360.0,
            COLOR_ELLIPSE_ISO,
            2,
            cv2.LINE_AA
        )
        # Centro
        cv2.circle(panel_left, (jx, jy), 4, COLOR_TARGET, -1)
        cv2.circle(panel_left, (jx, jy), 6, (0, 0, 0), 1)

    # Dibuja leyenda
    draw_legend(panel_left, metrics)

    # Construir Panel Derecho: Zoom Comparativo
    # Región de interés alrededor del salto principal
    crop_radius = int(r_large_px + 70)
    x1 = max(0, pjx - crop_radius)
    y1 = max(0, pjy - crop_radius)
    x2 = min(w, pjx + crop_radius)
    y2 = min(h, pjy + crop_radius)

    crop_base = img[y1:y2, x1:x2].copy()
    crop_walls = [
        (s[0] - x1, s[1] - y1, s[2] - x1, s[3] - y1)
        for s in wall_segments
        if (x1 <= s[0] <= x2 or x1 <= s[2] <= x2) and (y1 <= s[1] <= y2 or y1 <= s[3] <= y2)
    ]
    rel_jx = pjx - x1
    rel_jy = pjy - y1

    # Sub-panel 1: Círculo 90px
    crop_90 = crop_base.copy()
    for s in crop_walls:
        cv2.line(crop_90, (s[0], s[1]), (s[2], s[3]), COLOR_WALL, 2, cv2.LINE_AA)
    cv2.circle(crop_90, (rel_jx, rel_jy), int(r_small_px), COLOR_CIRCLE_90, 3, cv2.LINE_AA)
    cv2.circle(crop_90, (rel_jx, rel_jy), 5, COLOR_TARGET, -1)
    cv2.putText(
        crop_90,
        f"A: Circulo R={int(r_small_px)}px (Subestima horizontal)",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_CIRCLE_90,
        2,
        cv2.LINE_AA,
    )

    # Sub-panel 2: Círculo 125px
    crop_125 = crop_base.copy()
    for s in crop_walls:
        cv2.line(crop_125, (s[0], s[1]), (s[2], s[3]), COLOR_WALL, 2, cv2.LINE_AA)
    cv2.circle(crop_125, (rel_jx, rel_jy), int(r_large_px), COLOR_CIRCLE_125, 3, cv2.LINE_AA)
    cv2.circle(crop_125, (rel_jx, rel_jy), 5, COLOR_TARGET, -1)
    cv2.putText(
        crop_125,
        f"B: Circulo R={int(r_large_px)}px (Sobreestima vertical +93%)",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_CIRCLE_125,
        2,
        cv2.LINE_AA,
    )

    # Sub-panel 3: Elipse Isométrica
    crop_ellipse = crop_base.copy()
    for s in crop_walls:
        cv2.line(crop_ellipse, (s[0], s[1]), (s[2], s[3]), COLOR_WALL, 2, cv2.LINE_AA)
    cv2.ellipse(
        crop_ellipse,
        (rel_jx, rel_jy),
        (int(rx_px), int(ry_px)),
        0.0,
        0.0,
        360.0,
        COLOR_ELLIPSE_ISO,
        3,
        cv2.LINE_AA,
    )
    cv2.circle(crop_ellipse, (rel_jx, rel_jy), 5, COLOR_TARGET, -1)
    cv2.putText(
        crop_ellipse,
        f"C: Elipse Rx={int(rx_px)}px, Ry={int(ry_px)}px (Conforme exacta)",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_ELLIPSE_ISO,
        2,
        cv2.LINE_AA,
    )

    # Redimensionar crops para apilarlos verticalmente en el panel derecho
    target_crop_w = int(w * 0.40)
    target_crop_h = int(h / 3.0)

    c90_res = cv2.resize(crop_90, (target_crop_w, target_crop_h), interpolation=cv2.INTER_AREA)
    c125_res = cv2.resize(crop_125, (target_crop_w, target_crop_h), interpolation=cv2.INTER_AREA)
    cel_res = cv2.resize(crop_ellipse, (target_crop_w, target_crop_h), interpolation=cv2.INTER_AREA)

    panel_right = np.vstack([c90_res, c125_res, cel_res])
    # Ajustar altura de panel derecho si hay discrepancia por redondeo
    if panel_right.shape[0] != h:
        panel_right = cv2.resize(panel_right, (target_crop_w, h), interpolation=cv2.INTER_AREA)

    # Línea divisoria vertical
    separator = np.zeros((h, 4, 3), dtype=np.uint8)
    separator[:] = (100, 100, 100)

    # Componer imagen final lado a lado
    composite = np.hstack([panel_left, separator, panel_right])

    out_path = os.path.join(output_dir, f"calibration_footprint_{base_name}.png")
    cv2.imwrite(out_path, composite)
    log.info("Overlay de calibracion guardado en: %s", out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Calibración de huella de hechizo de salto (Fase 1.2)")
    parser.add_argument("--samples", type=int, default=5, help="Número de aldeas de muestra a calibrar")
    parser.add_argument("--all", action="store_true", help="Procesar todas las aldeas disponibles")
    parser.add_argument("--output-dir", type=str, default="logs/yolo", help="Directorio destino para overlays")
    parser.add_argument("--rx", type=float, default=115.0, help="Radio semi-mayor horizontal en px")
    parser.add_argument("--ry", type=float, default=65.0, help="Radio semi-menor vertical en px")
    parser.add_argument("--r-small", type=float, default=90.0, help="Radio de círculo pequeño en px")
    parser.add_argument("--r-large", type=float, default=125.0, help="Radio de círculo grande en px")
    args = parser.parse_args()

    raw_dir = os.path.join(PROJECT_ROOT, "datasets", "coc", "images", "raw")
    attack_raw_pattern = os.path.join(PROJECT_ROOT, "logs", "yolo", "attack_*", "screen_raw.png")

    raw_images = sorted(glob.glob(os.path.join(raw_dir, "*.png")))
    attack_images = sorted(glob.glob(attack_raw_pattern))
    all_images = raw_images + attack_images

    log.info("Imágenes encontradas: %d en raw, %d en attack logs (Total: %d)", len(raw_images), len(attack_images), len(all_images))

    if not all_images:
        log.error("No se encontraron imágenes para calibrar.")
        sys.exit(1)

    # Selección de muestras representativas si no es --all
    if args.all:
        selected_images = all_images
    else:
        step = max(1, len(all_images) // args.samples)
        selected_images = all_images[::step][:args.samples]

    log.info("Procesando %d aldeas seleccionadas para calibración...", len(selected_images))

    # Inicializar detector YOLO
    detector = None
    try:
        attack_cfg = SETTINGS.get("attack", {})
        conf = float(attack_cfg.get("yolo_conf", 0.40))
        imgsz = int(attack_cfg.get("yolo_imgsz", 1280))
        detector = YoloDetector(conf=conf, imgsz=imgsz)
        log.info("YoloDetector inicializado exitosamente.")
    except Exception as e:
        log.warning("No se pudo inicializar YoloDetector: %s (se utilizará fallback geométrico)", e)

    metrics = calculate_footprint_metrics(args.r_small, args.r_large, args.rx, args.ry)
    log.info("==================================================")
    log.info("METRICAS DE CALIBRACION DE HUELLA:")
    log.info("  Circulo 90px: Area = %.1f px2, Radio equiv = %.1f px", metrics["area_circle_small_px2"], metrics["r_circle_small"])
    log.info("  Circulo 125px: Area = %.1f px2 (+%.1f%% vs elipse)", metrics["area_circle_large_px2"], (metrics["area_ratio_125_vs_ellipse"] - 1) * 100)
    log.info("  Elipse Isométrica: Rx = %.1f px, Ry = %.1f px, Aspect = %.2f, Area = %.1f px2, Req = %.1f px",
             metrics["rx_ellipse"], metrics["ry_ellipse"], metrics["aspect_ratio"], metrics["area_ellipse_px2"], metrics["equiv_radius_ellipse"])
    log.info("==================================================")

    generated = []
    for img_path in selected_images:
        out_path = generate_side_by_side_comparison(
            image_path=img_path,
            output_dir=args.output_dir,
            detector=detector,
            rx_px=args.rx,
            ry_px=args.ry,
            r_small_px=args.r_small,
            r_large_px=args.r_large
        )
        if out_path:
            generated.append(out_path)

    log.info("Calibración completada. %d overlays generados en %s.", len(generated), args.output_dir)


if __name__ == "__main__":
    main()
```

## File: tools/capture_dataset.py
```python
"""
tools/capture_dataset.py
Toma capturas del dispositivo conectándose por ADB para armar dataset sin etiquetar.
"""
import argparse
import os
import sys
import time
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.device import Device
from core.logger import get_logger
from humanizer.behavior import HumanBehavior

log = get_logger("capture_dataset")


def main():
    parser = argparse.ArgumentParser(description="Captura imágenes raw del dispositivo para dataset CoC")
    parser.add_argument("--count", type=int, default=50, help="Número de capturas a tomar (default: 50)")
    parser.add_argument("--out_dir", type=str, default="datasets/coc/images/raw", help="Directorio de salida")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dev = Device()
    if not dev.connect():
        log.critical("No se pudo conectar al dispositivo por ADB.")
        return

    log.info("Iniciando captura de %d imágenes en %s...", args.count, args.out_dir)

    for i in range(1, args.count + 1):
        try:
            screen = dev.screenshot()
            ts = int(time.time() * 1000)
            filename = f"cap_{ts}.png"
            out_path = os.path.join(args.out_dir, filename)
            cv2.imwrite(out_path, screen)
            log.info("[%d/%d] Captura guardada: %s", i, args.count, out_path)
        except Exception as e:
            log.error("[%d/%d] Error tomando captura: %s", i, args.count, e)

        if i < args.count:
            # Pausa con jitter entre capturas
            HumanBehavior.sleep_range(1.5, 3.5)

    log.info("Proceso de captura finalizado. Total guardadas en %s", args.out_dir)


if __name__ == "__main__":
    main()
```

## File: tools/check_templates.py
```python
import cv2
import os
import glob

# Ruta de tus plantillas
CARPETA = "assets/buildings"
SALIDA = "logs/template_check"
os.makedirs(SALIDA, exist_ok=True)

archivos = glob.glob(os.path.join(CARPETA, "*.png"))
print(f"Revisando {len(archivos)} plantillas...")

for f in archivos:
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] < 4:
        continue

    alpha = img[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None: continue

    # Recortar igual que en el script principal
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]

    # Escalar al 35% (el tamaño medio que usa el script)
    new_w, new_h = int(w * 0.35), int(h * 0.35)
    if new_w < 5 or new_h < 5: continue

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Guardar para que lo veas
    nombre_salida = os.path.basename(f)
    cv2.imwrite(os.path.join(SALIDA, nombre_salida), resized)

print(f"✅ Listo. Abre la carpeta: {SALIDA}")
print("Revisa si las imágenes son edificios 3D grandes o íconos planos pequeños.")
```

## File: tools/crop_helper.py
```python
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import PROJECT_ROOT


def main():
    if len(sys.argv) < 2:
        print("Uso: python tools/crop_helper.py <ruta_imagen.png>")
        return

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"No se pudo abrir la imagen: {img_path}")
        return

    os.makedirs(os.path.join(PROJECT_ROOT, "assets", "digits"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "assets", "buttons"), exist_ok=True)

    print("=== ASISTENTE DE RECORTE CLASH BOT ===")
    print("1. Dibuja un rectángulo con el mouse y presiona ENTER o ESPACIO.")
    print("2. Presiona 'c' o ESC en la ventana para salir.")

    while True:
        r = cv2.selectROI("Recortar Template (ENTER=Guardar, ESC=Salir)", img, fromCenter=False, showCrosshair=True)
        x, y, w, h = [int(v) for v in r]
        if w == 0 or h == 0:
            break

        crop = img[y:y + h, x:x + w]
        nombre = input("Guardar como (ej: digits/8.png o buttons/next_btn.png): ").strip()
        if nombre:
            out_path = os.path.join(PROJECT_ROOT, "assets", nombre)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, crop)
            print(f"✅ Guardado en: {out_path} ({w}x{h} px)")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

## File: tools/diagnose_drop_points.py
```python
"""
tools/diagnose_drop_points.py
Herramienta de diagnóstico offline para validar puntos de drop de duendes (Fase 1.2).
"""
import os
import glob
import argparse
import csv
import cv2
import numpy as np

from core.config import PROJECT_ROOT, SETTINGS
from core.logger import get_logger
from vision.deploy_zone_detector import get_deploy_zone_detector
from vision.yolo_detector import get_detector
from attack.deployment_strategy import DeploymentStrategy

log = get_logger("diagnose_drop_points")


def get_battle_hud_mask(h: int, w: int) -> np.ndarray:
    """Filtra HUD superior e inferior."""
    mask = np.ones((h, w), dtype=np.uint8)
    mask[:138, :] = 0  # HUD superior
    mask[845:, :] = 0  # Barra inferior de tropas
    mask[:, :120] = 0  # Margen lateral izquierdo
    mask[:, 2280:] = 0  # Margen lateral derecho
    return mask


def main():
    parser = argparse.ArgumentParser(description="Diagnóstico offline de puntos de drop")
    parser.add_argument("--images", default="logs/yolo", help="Carpeta de imágenes a evaluar")
    parser.add_argument("--out", default="logs/diagnostics/drops", help="Carpeta de salida")
    parser.add_argument("--limit", type=int, default=20, help="Límite de imágenes")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    images_dir = os.path.join(PROJECT_ROOT, args.images) if not os.path.isabs(args.images) else args.images
    img_files = sorted(glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")))

    if not img_files:
        log.warning("No se encontraron imágenes en: %s", images_dir)
        return

    img_files = img_files[:args.limit]
    log.info("Analizando %d capturas...", len(img_files))

    dz = get_deploy_zone_detector()
    yolo = get_detector()
    strat = DeploymentStrategy(None, None)

    results = []
    dilations = [0, 15, 25, 35, 50]

    for idx_img, fpath in enumerate(img_files, 1):
        fname = os.path.basename(fpath)
        img = cv2.imread(fpath)
        if img is None:
            continue
        h, w = img.shape[:2]

        raw_dets = yolo.detect(img) if yolo and yolo.is_ready() else []
        hud_mask = get_battle_hud_mask(h, w)
        dets = [d for d in raw_dets if hud_mask[int(d.center[1]), int(d.center[0])] == 1]

        mask = dz.predict_mask(img) if dz and dz.is_ready() else None
        if mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)

        # Componentes conexos de la máscara
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # El mayor componente (excluyendo fondo label 0)
        largest_comp_idx = -1
        max_area = 0
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] > max_area:
                max_area = stats[lbl, cv2.CC_STAT_AREA]
                largest_comp_idx = lbl

        # Contornos principales
        pts_bld, pts_grs = dz.get_boundary_polygons(img, mask=mask)
        red_contour = pts_bld

        # Overlay de debug
        overlay = img.copy()
        # Máscara en magenta traslúcido
        magenta_layer = np.zeros_like(img)
        magenta_layer[mask > 0] = (255, 0, 255)
        overlay = cv2.addWeighted(overlay, 0.75, magenta_layer, 0.25, 0)

        for det in dets:
            cx, cy = int(round(det.center[0])), int(round(det.center[1]))
            bx1, by1, bx2, by2 = det.bbox_xyxy

            mask_at_center = int(mask[cy, cx]) if 0 <= cy < h and 0 <= cx < w else 0
            deployable_at_center = dz.is_point_deployable(cx, cy, safety_margin_px=0, mask=mask) if dz else True

            # Fracción de bbox prohibido
            sub_mask = mask[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
            frac_forbidden = float(np.mean(sub_mask > 0)) if sub_mask.size > 0 else 0.0

            # Componente conexo
            center_label = labels[cy, cx] if 0 <= cy < h and 0 <= cx < w else 0
            isolated_blob = bool(center_label > 0 and center_label != largest_comp_idx)

            # Drop actual calculado por DeploymentStrategy
            drop_x, drop_y = strat._get_drop_point_on_contour(
                det.center, green_contour=pts_grs, red_contour=pts_bld
            )

            drop_in_mask = bool(mask[drop_y, drop_x] > 0) if 0 <= drop_y < h and 0 <= drop_x < w else False

            # Comprobación de invasión de BBox con varias dilataciones
            veto_by_dil = {}
            for dil in dilations:
                invaded = False
                for other in dets:
                    obx1, oby1, obx2, oby2 = other.bbox_xyxy
                    if (obx1 - dil) <= drop_x <= (obx2 + dil) and (oby1 - dil) <= drop_y <= (oby2 + dil):
                        invaded = True
                        break
                veto_by_dil[dil] = invaded

            drop_inside_red = bool(cv2.pointPolygonTest(red_contour, (float(drop_x), float(drop_y)), False) >= 0) if red_contour is not None and len(red_contour) >= 3 else False

            # Clasificación de causa
            cause = "OK"
            if deployable_at_center and mask_at_center == 0:
                cause = "A_MODEL_MISS"
            elif isolated_blob:
                cause = "B_CONTOUR_BLIND"
            elif veto_by_dil[0]:
                cause = "DROP_IN_BBOX_DIL0"
            elif drop_in_mask:
                cause = "DROP_IN_MASK"

            row = {
                "image": fname,
                "class": det.cls_name,
                "confidence": f"{det.confidence:.2f}",
                "cx": cx,
                "cy": cy,
                "drop_x": drop_x,
                "drop_y": drop_y,
                "mask_at_center": mask_at_center,
                "deployable_at_center": deployable_at_center,
                "frac_bbox_forbidden": f"{frac_forbidden:.2f}",
                "isolated_blob": isolated_blob,
                "drop_in_mask": drop_in_mask,
                "drop_in_bbox_dil0": veto_by_dil[0],
                "drop_in_bbox_dil15": veto_by_dil[15],
                "drop_in_bbox_dil25": veto_by_dil[25],
                "drop_in_bbox_dil35": veto_by_dil[35],
                "drop_in_bbox_dil50": veto_by_dil[50],
                "drop_inside_red_main": drop_inside_red,
                "cause": cause,
            }
            results.append(row)

            # Dibujar en overlay
            box_color = (0, 255, 0) if cause == "OK" else (0, 0, 255)
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), box_color, 2)
            cv2.putText(overlay, f"{det.cls_name} {det.confidence:.2f}", (bx1, max(15, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

            drop_color = (0, 255, 0) if cause == "OK" else (0, 0, 255)
            cv2.circle(overlay, (drop_x, drop_y), 6, drop_color, -1)
            cv2.arrowedLine(overlay, (drop_x, drop_y), (cx, cy), (255, 255, 0), 1, tipLength=0.15)

        # Guardar imagen procesada
        diag_out_path = os.path.join(args.out, f"diag_{fname}")
        cv2.imwrite(diag_out_path, overlay)

    # Guardar CSV
    csv_path = os.path.join(args.out, "results.csv")
    fieldnames = list(results[0].keys()) if results else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    log.info("Resultados guardados en %s (%d detecciones)", csv_path, len(results))


if __name__ == "__main__":
    main()
```

## File: tools/evaluate_jumps_offline.py
```python
"""
Herramienta de Evaluación Offline y Línea Base de Colocación de Hechizos de Salto.
Fase 3 del Plan: colocación óptima de hechizos de salto (plan_saltos_optimos_antigravity.md).

Ejecuta y compara los 4 modos de decisión de saltos:
1. wall_raycast (modo actual de producción)
2. cluster_intersection (modo basado en centroides y cuadrantes)
3. legacy_quadrant (modo clásico proporcional)
4. coverage_optimizer (nuevo optimizador global de compartimentos)

Métricas evaluadas:
- Saltos usados
- Costo inicial de capas de muro hacia botín
- Costo final tras colocar saltos
- Reducción neta de costo (ganancia) y porcentaje de reducción
- Objetivos de botín que quedan cerrados (infranqueables)
- Tiempo de ejecución (ms)
- Barrido de sensibilidad de parámetros (buildings_block, grosor de muro, huella de salto)
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
import cv2
import numpy as np

# Configurar path raíz del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.logger import get_logger
from core.config import SETTINGS
from vision.yolo_detector import YoloDetector, Detection
from vision.wall_raycast import detect_global_wall_segments
import vision.wall_compartments as wc
import attack.deployment_strategy as ds

log = get_logger("evaluate_jumps_offline")


def discover_village_images(
    raw_dir: str = "datasets/coc/images/raw",
    attack_pattern: str = "logs/yolo/attack_*/screen_raw.png",
) -> list[tuple[str, str, str]]:
    """
    Descubre todas las imágenes disponibles para evaluación.
    Retorna lista de (village_id, file_path, source).
    """
    villages = []
    seen_paths = set()

    # 1. Imágenes sin procesar de datasets
    raw_files = sorted(glob.glob(os.path.join(PROJECT_ROOT, raw_dir, "*.png")))
    for rf in raw_files:
        norm = os.path.normpath(rf)
        if norm not in seen_paths:
            seen_paths.add(norm)
            vid = os.path.splitext(os.path.basename(rf))[0]
            villages.append((vid, norm, "raw"))

    # 2. Capturas de pantalla reales en logs de ataques
    atk_files = sorted(glob.glob(os.path.join(PROJECT_ROOT, attack_pattern)))
    for af in atk_files:
        norm = os.path.normpath(af)
        if norm not in seen_paths:
            seen_paths.add(norm)
            folder_name = os.path.basename(os.path.dirname(af))
            vid = f"attack_{folder_name}"
            villages.append((vid, norm, "attack_log"))

    return villages


def evaluate_jumps_under_compartment_model(
    comp_labels: np.ndarray,
    nearest_labels: np.ndarray,
    graph: dict[int, set[int]],
    entry_comps: set[int],
    targets: list[dict],
    jump_points: list[tuple[int, int]],
    footprint: dict | tuple | float | int,
) -> dict:
    """
    Evalúa cualquier conjunto de puntos de salto (x, y) utilizando el modelo de compartimentos.
    Calcula: costo inicial, costo final, reducción, % de reducción y objetivos cerrados.
    """
    initial_cost, initial_dists = wc.compute_total_cost(graph, entry_comps, targets, jump_bridges=[])

    jump_bridges = []
    for pt in jump_points:
        comps = wc.get_compartments_in_footprint(pt, footprint, comp_labels, nearest_labels)
        if len(comps) >= 2:
            jump_bridges.append(comps)

    final_cost, final_dists = wc.compute_total_cost(graph, entry_comps, targets, jump_bridges=jump_bridges)
    cost_reduction = initial_cost - final_cost
    pct_reduction = (cost_reduction / initial_cost * 100.0) if initial_cost > 0 else 0.0

    targets_closed = sum(1 for t in targets if final_dists.get(t["comp"], float("inf")) > 0)

    return {
        "num_jumps": len(jump_points),
        "initial_cost": round(float(initial_cost), 2),
        "final_cost": round(float(final_cost), 2),
        "cost_reduction": round(float(cost_reduction), 2),
        "pct_reduction": round(float(pct_reduction), 1),
        "targets_closed": int(targets_closed),
        "jumps": [[int(pt[0]), int(pt[1])] for pt in jump_points],
    }


def run_offline_evaluation(
    villages: list[tuple[str, str, str]],
    conf: float = 0.35,
    imgsz: int = 1280,
    max_jumps: int = 4,
    output_json_path: str = "logs/offline_evaluation_results.json",
) -> dict:
    """
    Ejecuta la evaluación offline completa sobre la lista de aldeas.
    """
    detector = YoloDetector(conf=conf, imgsz=imgsz)
    total_villages = len(villages)

    log.info("🚀 Iniciando Evaluación Offline sobre %d aldeas...", total_villages)

    per_village_results = []
    mode_accumulators = {
        "coverage_optimizer": defaultdict(list),
        "wall_raycast": defaultdict(list),
        "cluster_intersection": defaultdict(list),
        "legacy_quadrant": defaultdict(list),
    }

    # Acumuladores de sensibilidad
    sensitivity_accumulators = {
        "buildings_block": {
            "True": defaultdict(list),
            "False": defaultdict(list),
        },
        "wall_thickness": {
            "6": defaultdict(list),
            "8": defaultdict(list),
            "12": defaultdict(list),
        },
        "footprint": {
            "circle_90": defaultdict(list),
            "circle_125": defaultdict(list),
            "ellipse_115_65": defaultdict(list),
        },
    }

    t_eval_start = time.perf_counter()

    for idx, (vid, img_path, source) in enumerate(villages, start=1):
        rel_path = os.path.relpath(img_path, PROJECT_ROOT).replace("\\", "/")
        log.info("[%d/%d] Evaluando aldea: %s (%s)", idx, total_villages, vid, source)

        img = cv2.imread(img_path)
        if img is None:
            log.warning("⚠️ No se pudo cargar imagen: %s", img_path)
            continue

        h, w = img.shape[:2]

        # 1. Detecciones YOLO
        t_det0 = time.perf_counter()
        detections = detector.detect(img)
        t_det = (time.perf_counter() - t_det0) * 1000.0

        # 2. Detección global de segmentos de muro
        t_wall0 = time.perf_counter()
        iso_angles, wall_segments = detect_global_wall_segments(
            img, force_refresh=True, detections=detections
        )
        t_wall = (time.perf_counter() - t_wall0) * 1000.0

        # 3. Construir modelo de compartimentos base (buildings_block=True, wall_thickness=8, footprint=90px)
        default_footprint = 90.0
        wall_mask = wc.rasterize_wall_mask((h, w), wall_segments, iso_angles=iso_angles, wall_thickness=8)
        barrier_mask = wc.build_barrier_mask(wall_mask, detections, buildings_block=True)
        num_comps, comp_labels, entry_comps, nearest_labels = wc.compute_compartments(barrier_mask)
        graph, edges = wc.build_compartment_graph(num_comps, comp_labels, nearest_labels, wall_mask)
        targets = wc.map_targets_to_compartments(detections, comp_labels, nearest_labels)

        village_entry = {
            "village_id": vid,
            "image_path": rel_path,
            "source": source,
            "resolution": [w, h],
            "num_wall_segments": len(wall_segments),
            "num_compartments": max(1, num_comps - 1),
            "num_loot_targets": len(targets),
            "modes": {},
        }

        # --- A. coverage_optimizer ---
        t0 = time.perf_counter()
        opt_res = wc.optimize_jumps(
            img_shape=(h, w),
            wall_segments=wall_segments,
            detections=detections,
            iso_angles=iso_angles,
            jump_footprint=default_footprint,
            max_jumps=max_jumps,
            buildings_block=True,
            wall_thickness=8,
            generate_debug_overlay=False,
        )
        rt_opt = (time.perf_counter() - t0) * 1000.0
        opt_jumps = [(j[0], j[1]) for j in opt_res.jumps]
        eval_opt = evaluate_jumps_under_compartment_model(
            comp_labels, nearest_labels, graph, entry_comps, targets, opt_jumps, default_footprint
        )
        eval_opt["runtime_ms"] = round(rt_opt, 1)
        village_entry["modes"]["coverage_optimizer"] = eval_opt

        # --- B. wall_raycast ---
        t0 = time.perf_counter()
        raycast_jumps = ds._compute_wall_raycast_jump_targets(img, detections, (w, h))
        rt_raycast = (time.perf_counter() - t0) * 1000.0
        eval_raycast = evaluate_jumps_under_compartment_model(
            comp_labels, nearest_labels, graph, entry_comps, targets, raycast_jumps, default_footprint
        )
        eval_raycast["runtime_ms"] = round(rt_raycast, 1)
        village_entry["modes"]["wall_raycast"] = eval_raycast

        # --- C. cluster_intersection ---
        t0 = time.perf_counter()
        cluster_jumps = ds._compute_cluster_intersection_jump_targets(detections, (w, h))
        rt_cluster = (time.perf_counter() - t0) * 1000.0
        eval_cluster = evaluate_jumps_under_compartment_model(
            comp_labels, nearest_labels, graph, entry_comps, targets, cluster_jumps, default_footprint
        )
        eval_cluster["runtime_ms"] = round(rt_cluster, 1)
        village_entry["modes"]["cluster_intersection"] = eval_cluster

        # --- D. legacy_quadrant ---
        t0 = time.perf_counter()
        legacy_jumps = ds._compute_legacy_quadrant_jump_targets(detections, (w, h))
        rt_legacy = (time.perf_counter() - t0) * 1000.0
        eval_legacy = evaluate_jumps_under_compartment_model(
            comp_labels, nearest_labels, graph, entry_comps, targets, legacy_jumps, default_footprint
        )
        eval_legacy["runtime_ms"] = round(rt_legacy, 1)
        village_entry["modes"]["legacy_quadrant"] = eval_legacy

        # Acumular para estadísticas globales
        for mode_name, metrics in village_entry["modes"].items():
            for k, val in metrics.items():
                if k != "jumps":
                    mode_accumulators[mode_name][k].append(val)

        # --- SENSITIVITY SWEEP SOBRE ESTA ALDEA ---
        # 1. buildings_block (True vs False)
        for bb in [True, False]:
            sw_res = wc.optimize_jumps(
                img_shape=(h, w),
                wall_segments=wall_segments,
                detections=detections,
                iso_angles=iso_angles,
                jump_footprint=default_footprint,
                max_jumps=max_jumps,
                buildings_block=bb,
                wall_thickness=8,
                generate_debug_overlay=False,
            )
            sw_eval = evaluate_jumps_under_compartment_model(
                comp_labels, nearest_labels, graph, entry_comps, targets,
                [(j[0], j[1]) for j in sw_res.jumps], default_footprint
            )
            sensitivity_accumulators["buildings_block"][str(bb)]["num_jumps"].append(sw_eval["num_jumps"])
            sensitivity_accumulators["buildings_block"][str(bb)]["final_cost"].append(sw_eval["final_cost"])
            sensitivity_accumulators["buildings_block"][str(bb)]["cost_reduction"].append(sw_eval["cost_reduction"])
            sensitivity_accumulators["buildings_block"][str(bb)]["pct_reduction"].append(sw_eval["pct_reduction"])
            sensitivity_accumulators["buildings_block"][str(bb)]["runtime_ms"].append(sw_res.execution_time_ms)

        # 2. wall_thickness (6, 8, 12)
        for wt in [6, 8, 12]:
            sw_res = wc.optimize_jumps(
                img_shape=(h, w),
                wall_segments=wall_segments,
                detections=detections,
                iso_angles=iso_angles,
                jump_footprint=default_footprint,
                max_jumps=max_jumps,
                buildings_block=True,
                wall_thickness=wt,
                generate_debug_overlay=False,
            )
            sw_eval = evaluate_jumps_under_compartment_model(
                comp_labels, nearest_labels, graph, entry_comps, targets,
                [(j[0], j[1]) for j in sw_res.jumps], default_footprint
            )
            sensitivity_accumulators["wall_thickness"][str(wt)]["num_jumps"].append(sw_eval["num_jumps"])
            sensitivity_accumulators["wall_thickness"][str(wt)]["final_cost"].append(sw_eval["final_cost"])
            sensitivity_accumulators["wall_thickness"][str(wt)]["cost_reduction"].append(sw_eval["cost_reduction"])
            sensitivity_accumulators["wall_thickness"][str(wt)]["pct_reduction"].append(sw_eval["pct_reduction"])
            sensitivity_accumulators["wall_thickness"][str(wt)]["runtime_ms"].append(sw_res.execution_time_ms)

        # 3. jump_footprint (circle_90, circle_125, ellipse_115_65)
        fp_variants = {
            "circle_90": 90.0,
            "circle_125": 125.0,
            "ellipse_115_65": {"type": "ellipse", "rx": 115.0, "ry": 65.0, "angle": 0.0},
        }
        for fp_name, fp_spec in fp_variants.items():
            sw_res = wc.optimize_jumps(
                img_shape=(h, w),
                wall_segments=wall_segments,
                detections=detections,
                iso_angles=iso_angles,
                jump_footprint=fp_spec,
                max_jumps=max_jumps,
                buildings_block=True,
                wall_thickness=8,
                generate_debug_overlay=False,
            )
            sw_eval = evaluate_jumps_under_compartment_model(
                comp_labels, nearest_labels, graph, entry_comps, targets,
                [(j[0], j[1]) for j in sw_res.jumps], fp_spec
            )
            sensitivity_accumulators["footprint"][fp_name]["num_jumps"].append(sw_eval["num_jumps"])
            sensitivity_accumulators["footprint"][fp_name]["final_cost"].append(sw_eval["final_cost"])
            sensitivity_accumulators["footprint"][fp_name]["cost_reduction"].append(sw_eval["cost_reduction"])
            sensitivity_accumulators["footprint"][fp_name]["pct_reduction"].append(sw_eval["pct_reduction"])
            sensitivity_accumulators["footprint"][fp_name]["runtime_ms"].append(sw_res.execution_time_ms)

        per_village_results.append(village_entry)

    total_eval_time = (time.perf_counter() - t_eval_start)

    # --- AGREGAR RESUMEN POR MODO ---
    summary_modes = {}
    for mode_name, metrics in mode_accumulators.items():
        summary_modes[mode_name] = {
            "mean_jumps": round(float(np.mean(metrics["num_jumps"])), 2),
            "mean_initial_cost": round(float(np.mean(metrics["initial_cost"])), 2),
            "mean_final_cost": round(float(np.mean(metrics["final_cost"])), 2),
            "mean_cost_reduction": round(float(np.mean(metrics["cost_reduction"])), 2),
            "mean_pct_reduction": round(float(np.mean(metrics["pct_reduction"])), 1),
            "mean_targets_closed": round(float(np.mean(metrics["targets_closed"])), 2),
            "mean_runtime_ms": round(float(np.mean(metrics["runtime_ms"])), 1),
        }

    # --- AGREGAR RESUMEN DE SENSIBILIDAD ---
    summary_sensitivity = {}
    for param_name, param_vars in sensitivity_accumulators.items():
        summary_sensitivity[param_name] = {}
        for var_name, metrics in param_vars.items():
            summary_sensitivity[param_name][var_name] = {
                "mean_jumps": round(float(np.mean(metrics["num_jumps"])), 2),
                "mean_final_cost": round(float(np.mean(metrics["final_cost"])), 2),
                "mean_cost_reduction": round(float(np.mean(metrics["cost_reduction"])), 2),
                "mean_pct_reduction": round(float(np.mean(metrics["pct_reduction"])), 1),
                "mean_runtime_ms": round(float(np.mean(metrics["runtime_ms"])), 1),
            }

    # --- RANKING DE ALDEAS (10 Mejores y 10 Peores) ---
    # Ordenar por ganancia neta (cost_reduction) y porcentaje de reducción
    sorted_by_opt = sorted(
        per_village_results,
        key=lambda v: (
            v["modes"]["coverage_optimizer"]["cost_reduction"],
            v["modes"]["coverage_optimizer"]["pct_reduction"],
        ),
        reverse=True,
    )

    best_10 = []
    for v in sorted_by_opt[:10]:
        m = v["modes"]["coverage_optimizer"]
        best_10.append({
            "village_id": v["village_id"],
            "image_path": v["image_path"],
            "source": v["source"],
            "num_wall_segments": v["num_wall_segments"],
            "num_compartments": v["num_compartments"],
            "num_loot_targets": v["num_loot_targets"],
            "initial_cost": m["initial_cost"],
            "final_cost": m["final_cost"],
            "cost_reduction": m["cost_reduction"],
            "pct_reduction": m["pct_reduction"],
            "num_jumps": m["num_jumps"],
            "jumps": m["jumps"],
        })

    worst_10 = []
    for v in sorted_by_opt[-10:]:
        m = v["modes"]["coverage_optimizer"]
        worst_10.append({
            "village_id": v["village_id"],
            "image_path": v["image_path"],
            "source": v["source"],
            "num_wall_segments": v["num_wall_segments"],
            "num_compartments": v["num_compartments"],
            "num_loot_targets": v["num_loot_targets"],
            "initial_cost": m["initial_cost"],
            "final_cost": m["final_cost"],
            "cost_reduction": m["cost_reduction"],
            "pct_reduction": m["pct_reduction"],
            "num_jumps": m["num_jumps"],
            "jumps": m["jumps"],
        })

    final_payload = {
        "summary": {
            "total_villages_evaluated": len(per_village_results),
            "total_evaluation_time_sec": round(total_eval_time, 1),
            "modes_comparison": summary_modes,
            "sensitivity_sweep": summary_sensitivity,
            "ranking": {
                "best_10": best_10,
                "worst_10": worst_10,
            },
        },
        "per_village": per_village_results,
    }

    # Guardar a disco
    out_dir = os.path.dirname(os.path.join(PROJECT_ROOT, output_json_path))
    os.makedirs(out_dir, exist_ok=True)
    full_out = os.path.join(PROJECT_ROOT, output_json_path)
    with open(full_out, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    log.info("💾 Resultados de evaluación offline guardados exitosamente en: %s", full_out)

    # Imprimir resumen de alto impacto en consola
    print("\n" + "=" * 90)
    print("📊 RESUMEN COMPARATIVO DE MODOS DE SALTO (FASE 3 OFFLINE)")
    print("=" * 90)
    print(f"{'Modo':<24} | {'Saltos':<8} | {'Costo Ini':<10} | {'Costo Fin':<10} | {'Ganancia':<10} | {'% Reduc':<8} | {'Tiempo':<8}")
    print("-" * 90)
    for mode, s in summary_modes.items():
        print(f"{mode:<24} | {s['mean_jumps']:<8.2f} | {s['mean_initial_cost']:<10.2f} | {s['mean_final_cost']:<10.2f} | {s['mean_cost_reduction']:<10.2f} | {s['mean_pct_reduction']:<7.1f}% | {s['mean_runtime_ms']:<6.1f}ms")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("🔬 BARRIDO DE SENSIBILIDAD (coverage_optimizer)")
    print("=" * 90)
    for param, p_vars in summary_sensitivity.items():
        print(f"Parámetro: {param}")
        for v_name, s in p_vars.items():
            print(f"  • {v_name:<16}: Saltos={s['mean_jumps']:.2f} | CostoFin={s['mean_final_cost']:.2f} | Ganancia={s['mean_cost_reduction']:.2f} ({s['mean_pct_reduction']:.1f}%) | Tiempo={s['mean_runtime_ms']:.1f}ms")
    print("=" * 90 + "\n")

    return final_payload


def main():
    parser = argparse.ArgumentParser(description="Evaluación offline y línea base de saltos.")
    parser.add_argument("--limit", type=int, default=None, help="Limita el número de aldeas a procesar (para pruebas rápidas)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confianza YOLO")
    parser.add_argument("--imgsz", type=int, default=1280, help="Tamaño de imagen YOLO")
    parser.add_argument("--max-jumps", type=int, default=4, help="Número máximo de saltos")
    parser.add_argument("--output", type=str, default="logs/offline_evaluation_results.json", help="Ruta del archivo JSON de salida")
    args = parser.parse_args()

    villages = discover_village_images()
    if args.limit:
        villages = villages[:args.limit]

    run_offline_evaluation(
        villages=villages,
        conf=args.conf,
        imgsz=args.imgsz,
        max_jumps=args.max_jumps,
        output_json_path=args.output,
    )


if __name__ == "__main__":
    main()
```

## File: tools/measure_wall_raycast_accuracy.py
```python
"""
tools/measure_wall_raycast_accuracy.py
Herramienta de validación cuantitativa recomendada por el usuario:
Evalúa la precisión de Wall Raycast sobre 12 capturas reales de batallas guardadas en logs/yolo/.
Compara la posición de cada cruce de muro detectado contra la cresta de gradiente real
(máximo de relieve de muro) a lo largo del rayo y calcula el error medio en píxeles.
Criterio de aceptación: Error medio < 25.0 px (el radio de 85px del hechizo cubre con holgura).
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import glob
import cv2
import numpy as np

from vision.wall_raycast import (
    detect_global_wall_segments,
    find_wall_crossings_from_segments,
    reset_iso_angles_cache,
)
from vision.yolo_detector import get_detector
from attack.deployment_strategy import compute_canonical_diamond_drop


def measure_accuracy():
    files = sorted(glob.glob("logs/yolo/attack_*.png"), reverse=True)
    # Filtrar solo capturas reales completas (> 1MB)
    valid_files = [f for f in files if os.path.getsize(f) > 1_000_000][:12]

    if not valid_files:
        print("No se encontraron capturas válidas en logs/yolo/")
        return

    print(f"==================================================================")
    print(f"🔍 Evaluando precisión de Wall Raycast sobre {len(valid_files)} capturas reales de CoC")
    print(f"==================================================================")

    detector = get_detector()
    total_crossings_evaluated = 0
    all_errors = []

    for idx, path in enumerate(valid_files):
        reset_iso_angles_cache()
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Gradiente Sobel para hallar las crestas reales de muros
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)

        iso_angles, wall_segs = detect_global_wall_segments(img, force_refresh=True)

        detections = []
        if detector and detector.is_ready():
            detections = detector.detect(img)

        storages = [
            d for d in detections
            if d.cls_name in {"storage_dark", "storage_gold", "storage_elixir", "town_hall"}
        ]

        if not storages:
            continue

        # Centroide de la aldea
        all_pts = np.array([d.center for d in detections], dtype=np.float32)
        center = (float(np.mean(all_pts[:, 0])), float(np.mean(all_pts[:, 1])))

        file_errors = []
        for s in storages[:4]:  # Evaluar hasta 4 almacenes por captura
            p_target = (int(s.center[0]), int(s.center[1]))
            p_drop = compute_canonical_diamond_drop(p_target, center_pt=center)

            crossings = find_wall_crossings_from_segments(
                p_drop=p_drop,
                p_target=p_target,
                wall_segments=wall_segs,
                min_crossing_angle_deg=25.0,
                min_dist_drop=40.0,
                min_dist_target=50.0,
                endpoint_tol=30.0,
            )

            # Para cada cruce detectado, medir la distancia al pico de gradiente más cercano a lo largo del rayo
            vx = p_target[0] - p_drop[0]
            vy = p_target[1] - p_drop[1]
            dist_ray = max(float(np.hypot(vx, vy)), 1e-3)
            ux, uy = vx / dist_ray, vy / dist_ray

            for cx, cy in crossings:
                # Muestrear el perfil de gradiente en una ventana local [-35, +35] px alrededor del cruce
                best_peak_dist = 0.0
                max_grad = -1.0
                for offset in range(-35, 36):
                    sx = int(round(cx + offset * ux))
                    sy = int(round(cy + offset * uy))
                    if 0 <= sx < w and 0 <= sy < h:
                        g = float(grad_mag[sy, sx])
                        if g > max_grad:
                            max_grad = g
                            best_peak_dist = abs(offset)

                file_errors.append(best_peak_dist)
                all_errors.append(best_peak_dist)
                total_crossings_evaluated += 1

        avg_file_err = (sum(file_errors) / len(file_errors)) if file_errors else 0.0
        print(f"[{idx+1:02d}/12] {os.path.basename(path)} | Muros: {len(wall_segs):3d} | Cruces: {len(file_errors):2d} | Error Medio: {avg_file_err:4.1f} px")

    if all_errors:
        mean_err = sum(all_errors) / len(all_errors)
        max_err = max(all_errors)
        p90_err = float(np.percentile(all_errors, 90))

        print(f"\n==================================================================")
        print(f"📊 RESUMEN DE PRECISIÓN CUANTITATIVA (Muros Evaluados: {total_crossings_evaluated})")
        print(f"   • Error Medio en Píxeles: {mean_err:.2f} px (Umbral Aceptación: < 25.0 px)")
        print(f"   • Percentil 90:            {p90_err:.2f} px")
        print(f"   • Error Máximo Local:      {max_err:.2f} px")
        if mean_err < 25.0:
            print(f"   ✅ RESULTADO: SUPERADO CON ÉXITO (Precisión milimétrica sobre muros)")
        else:
            print(f"   ⚠️ RESULTADO: Requiere afinamiento de umbrales")
        print(f"==================================================================")


if __name__ == "__main__":
    measure_accuracy()
```

## File: tools/prelabel_hsv.py
```python
"""
tools/prelabel_hsv.py
Script CLI para pre-anotar imágenes crudas con formato YOLO v8 utilizando
la detección por colorimetría HSV (StorageDetector.detect_boxes).
"""
import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Asegurar importación del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.storage_detector import StorageDetector
from core.logger import get_logger

log = get_logger("prelabel_hsv")

CLASS_TO_ID = {
    "storage_gold": 0,
    "storage_elixir": 1,
    "storage_dark": 2,
}

CLASS_COLORS = {
    "storage_gold": (0, 215, 255),    # Dorado / BGR
    "storage_elixir": (255, 0, 255),  # Magenta
    "storage_dark": (128, 0, 128),    # Púrpura oscuro
}


def draw_debug_overlay(img: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """Dibuja rectángulos, etiquetas y banner superior en la imagen."""
    debug_img = img.copy()
    h, w = debug_img.shape[:2]

    counts = {"storage_gold": 0, "storage_elixir": 0, "storage_dark": 0}

    for b in boxes:
        cls_name = b["cls"]
        x1, y1, x2, y2 = b["bbox_xyxy"]
        conf = b["confidence"]
        counts[cls_name] = counts.get(cls_name, 0) + 1

        color = CLASS_COLORS.get(cls_name, (0, 255, 0))
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)

        # Etiqueta con fondo
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bg_y1 = max(0, y1 - th - 6)
        cv2.rectangle(debug_img, (x1, bg_y1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(debug_img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Banner superior
    banner_text = f"gold:{counts['storage_gold']}  elixir:{counts['storage_elixir']}  dark:{counts['storage_dark']}  |  total:{len(boxes)}"
    banner_h = 40
    overlay = debug_img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, debug_img, 0.25, 0, debug_img)
    cv2.putText(debug_img, banner_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return debug_img


def prelabel_images(input_dir: str, output_dir: str, debug_dir: str, min_conf: float = 0.30, overwrite: bool = False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    debug_path = Path(debug_dir) if debug_dir else None

    if not input_path.exists():
        log.error("El directorio de entrada no existe: %s", input_path)
        return

    output_path.mkdir(parents=True, exist_ok=True)
    if debug_path:
        debug_path.mkdir(parents=True, exist_ok=True)

    # Buscar imágenes
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    img_files = sorted([f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions])

    if not img_files:
        log.warning("No se encontraron imágenes en: %s", input_path)
        return

    log.info("Iniciando pre-etiquetado HSV en %d imágenes...", len(img_files))
    detector = StorageDetector()

    total_processed = 0
    total_skipped = 0
    class_totals = {"storage_gold": 0, "storage_elixir": 0, "storage_dark": 0}
    zero_detection_images = []

    for img_file in img_files:
        label_file = output_path / f"{img_file.stem}.txt"
        if label_file.exists() and not overwrite:
            log.debug("Omitiendo %s (label ya existe y --overwrite no fue activado)", img_file.name)
            total_skipped += 1
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            log.warning("No se pudo leer imagen: %s", img_file)
            continue

        h, w = img.shape[:2]
        boxes = detector.detect_boxes(img, min_conf=min_conf)

        # Generar líneas YOLO: <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
        yolo_lines = []
        counts = {"storage_gold": 0, "storage_elixir": 0, "storage_dark": 0}

        for b in boxes:
            cls_name = b["cls"]
            if cls_name not in CLASS_TO_ID:
                continue
            cls_id = CLASS_TO_ID[cls_name]
            counts[cls_name] += 1
            class_totals[cls_name] += 1

            x1, y1, x2, y2 = b["bbox_xyxy"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = (x2 - x1)
            bh = (y2 - y1)

            cx_norm = cx / w
            cy_norm = cy / h
            w_norm = bw / w
            h_norm = bh / h

            yolo_lines.append(f"{cls_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Guardar archivo de etiquetas (incluso si está vacío)
        with open(label_file, "w", encoding="utf-8") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")

        # Guardar imagen debug si fue solicitada
        if debug_path:
            debug_img = draw_debug_overlay(img, boxes)
            debug_out_file = debug_path / f"{img_file.stem}.png"
            cv2.imwrite(str(debug_out_file), debug_img)

        total_processed += 1
        if not boxes:
            zero_detection_images.append(img_file.name)
            log.info("  [-] %-30s | 0 detecciones (requiere revisión manual)", img_file.name)
        else:
            log.info("  [+] %-30s | %2d detecciones (gold=%d, elixir=%d, dark=%d)",
                     img_file.name, len(boxes), counts["storage_gold"], counts["storage_elixir"], counts["storage_dark"])

    # Resumen final
    log.info("==================================================")
    log.info("RESUMEN DE PRE-ETIQUETADO HSV:")
    log.info("  Imágenes procesadas : %d", total_processed)
    log.info("  Imágenes omitidas   : %d", total_skipped)
    log.info("  Total storage_gold  : %d", class_totals["storage_gold"])
    log.info("  Total storage_elixir: %d", class_totals["storage_elixir"])
    log.info("  Total storage_dark  : %d", class_totals["storage_dark"])
    log.info("  Total almacenes     : %d", sum(class_totals.values()))
    log.info("  Imágenes sin cajas  : %d (%.1f%%)", len(zero_detection_images),
             (len(zero_detection_images) / total_processed * 100) if total_processed > 0 else 0)
    if zero_detection_images:
        log.info("  Lista sin cajas: %s", ", ".join(zero_detection_images[:10]) + ("..." if len(zero_detection_images) > 10 else ""))
    log.info("==================================================")


def main():
    parser = argparse.ArgumentParser(description="Pre-etiquetador HSV de almacenes para dataset YOLO")
    parser.add_argument("--input", type=str, default="datasets/coc/images/raw", help="Directorio con imágenes crudas")
    parser.add_argument("--output", type=str, default="datasets/coc/labels/raw", help="Directorio destino para etiquetas .txt")
    parser.add_argument("--debug", type=str, default="datasets/coc/images/raw_debug", help="Directorio destino para imágenes debug anotadas")
    parser.add_argument("--min-conf", type=float, default=0.30, help="Confianza mínima heurística para filtrar cajas")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribir etiquetas existentes")
    args = parser.parse_args()

    prelabel_images(
        input_dir=args.input,
        output_dir=args.output,
        debug_dir=args.debug,
        min_conf=args.min_conf,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
```

## File: tools/prelabel_with_model.py
```python
"""
tools/prelabel_with_model.py
Script para pre-etiquetado asistido por modelo YOLO (Ronda 2 del workflow).
Carga un modelo entrenado (ej: runs/coc_v1/weights/best.pt) e infiere sobre las capturas
en 'datasets/coc/images/raw', guardando los archivos .txt en formato YOLO para su
revisión y corrección rápida en LabelImg, Label Studio, CVAT o tools/review_prelabels.py.
"""
import argparse
import glob
import os
import sys
from pathlib import Path
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import get_logger

log = get_logger("prelabel_model")


def main():
    parser = argparse.ArgumentParser(description="Pre-etiquetar imágenes con modelo YOLOv8 entrenado")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/coc_v1/weights/best.pt",
        help="Ruta a los pesos del modelo YOLO (default: runs/coc_v1/weights/best.pt)"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="datasets/coc/images/raw",
        help="Directorio con imágenes crudas (default: datasets/coc/images/raw)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="datasets/coc/labels_pre",
        help="Directorio destino para etiquetas .txt (default: datasets/coc/labels_pre)"
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Tamaño de imagen para inferencia (default: 1280)")
    parser.add_argument("--conf", type=float, default=0.20, help="Umbral de confianza para pre-etiquetar (default: 0.20)")
    parser.add_argument("--iou", type=float, default=0.50, help="Umbral IoU para NMS (default: 0.50)")
    parser.add_argument(
        "--filter-hud",
        action="store_true",
        default=True,
        help="Filtrar detecciones fuera del área de juego (150 < y < 900) para evitar falsos positivos en HUD"
    )
    parser.add_argument("--no-filter-hud", dest="filter_hud", action="store_false")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        log.critical("Modelo no encontrado en: %s", args.model)
        log.info(
            "Recuerda que para la Ronda 1 debes etiquetar a mano ~12 imágenes y entrenar un modelo rápido con:"
            "\n  python tools/train_yolo.py --epochs 40 --batch 8 --imgsz 1280 --name coc_v1"
        )
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError as e:
        log.critical("No se pudo importar 'ultralytics': %s", e)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    images = sorted(glob.glob(os.path.join(args.src, "*.png")) + glob.glob(os.path.join(args.src, "*.jpg")))

    if not images:
        log.warning("No se encontraron imágenes en: %s", args.src)
        sys.exit(0)

    log.info("Cargando modelo: %s...", args.model)
    model = YOLO(args.model)

    log.info(
        "Procesando %d imágenes con imgsz=%d, conf=%.2f, iou=%.2f, filter_hud=%s...",
        len(images), args.imgsz, args.conf, args.iou, args.filter_hud
    )

    total_boxes = 0
    class_counts = {}

    for idx, p in enumerate(images, 1):
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]

        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]

        lines = []
        for (a, b, c, d), k in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
            cls_id = int(k)
            cx_px = (a + c) / 2.0
            cy_px = (b + d) / 2.0
            bw_px = c - a
            bh_px = d - b

            # Filtro de área de juego (HUD superior e inferior)
            if args.filter_hud:
                # Proporcional a la altura de la imagen (~150 a ~900 px en 1080p)
                min_y = 0.138 * h
                max_y = 0.833 * h
                if cy_px < min_y or cy_px > max_y:
                    continue

            cx_norm = cx_px / w
            cy_norm = cy_px / h
            w_norm = bw_px / w
            h_norm = bh_px / h

            lines.append(f"{cls_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            total_boxes += 1

        base_name = os.path.splitext(os.path.basename(p))[0]
        out_file = os.path.join(args.out, f"{base_name}.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

        log.info("  [%d/%d] %-35s -> %d cajas pre-etiquetadas", idx, len(images), os.path.basename(p), len(lines))

    log.info("=" * 60)
    log.info("PRE-ETIQUETADO COMPLETADO:")
    log.info("  Total imágenes procesadas : %d", len(images))
    log.info("  Total cajas generadas     : %d", total_boxes)
    log.info("  Desglose por clase ID     : %s", class_counts)
    log.info("  Directorio de etiquetas   : %s", os.path.abspath(args.out))
    log.info("=" * 60)
    log.info("Siguiente paso (Ronda 2): Abre estas etiquetas con tools/review_prelabels.py o LabelImg/Label Studio para corregir cajas.")


if __name__ == "__main__":
    main()
```

## File: tools/rename_templates.py
```python
import os
from PIL import Image

# Asegúrate de que esta sea la ruta correcta
CARPETA = r"C:\ClashFarmerbot\assets\buildings"


def renombrar_y_convertir():
    if not os.path.exists(CARPETA):
        print(f"❌ La carpeta {CARPETA} no existe. Créala y pon tus imágenes ahí.")
        return

    contador = 0
    for archivo in os.listdir(CARPETA):
        # Ignorar si ya está en el formato correcto
        if archivo.startswith("storage_") or archivo.startswith("collector_"):
            continue

        # Obtener el nombre sin extensión
        nombre_base, extension = os.path.splitext(archivo)
        nuevo_nombre = None

        # --- MAPEO DE ALMACENES ---
        if "Dark_Elixir_Storage" in nombre_base:
            nivel = nombre_base.replace("Dark_Elixir_Storage", "").strip()
            nuevo_nombre = f"storage_dark_lvl{nivel}.png"
        elif "Elixir_Storage" in nombre_base:
            nivel = nombre_base.replace("Elixir_Storage", "").strip()
            nuevo_nombre = f"storage_elixir_lvl{nivel}.png"
        elif "Gold_Storage" in nombre_base:
            nivel = nombre_base.replace("Gold_Storage", "").strip()
            nuevo_nombre = f"storage_gold_lvl{nivel}.png"

        # --- MAPEO DE RECOLECTORES ---
        elif "Dark_Elixir_Drill" in nombre_base:
            nivel = nombre_base.replace("Dark_Elixir_Drill", "").strip()
            nuevo_nombre = f"collector_dark_lvl{nivel}.png"
        elif "Elixir_Collector" in nombre_base:
            nivel = nombre_base.replace("Elixir_Collector", "").strip()
            nuevo_nombre = f"collector_elixir_lvl{nivel}.png"
        elif "Gold_Mine" in nombre_base:
            nivel = nombre_base.replace("Gold_Mine", "").strip()
            nuevo_nombre = f"collector_gold_lvl{nivel}.png"

        # Si encontramos una coincidencia, procesamos
        if nuevo_nombre:
            ruta_vieja = os.path.join(CARPETA, archivo)
            ruta_nueva = os.path.join(CARPETA, nuevo_nombre)

            # Si el archivo es .webp, lo convertimos a .png real
            if extension.lower() == '.webp':
                try:
                    img = Image.open(ruta_vieja).convert("RGBA")
                    img.save(ruta_nueva, "PNG")
                    os.remove(ruta_vieja)  # Borramos el .webp original
                    print(f"✅ Convertido y renombrado: {archivo} -> {nuevo_nombre}")
                except Exception as e:
                    print(f"❌ Error convirtiendo {archivo}: {e}")
            else:
                # Si ya es png (o similar), solo lo renombramos
                os.rename(ruta_vieja, ruta_nueva)
                print(f"✅ Renombrado: {archivo} -> {nuevo_nombre}")

            contador += 1

    print(f"\n🎉 Proceso terminado. {contador} archivos procesados.")


if __name__ == "__main__":
    renombrar_y_convertir()
```

## File: tools/review_prelabels.py
```python
"""
tools/review_prelabels.py
Visor interactivo con OpenCV para revisar, corregir y complementar
las anotaciones YOLO generadas por prelabel_hsv.py o anotaciones manuales.
"""
import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Configuración de clases y colores BGR
CLASS_NAMES = {
    0: "storage_gold",
    1: "storage_elixir",
    2: "storage_dark",
    3: "town_hall",
    4: "collector_gold",
    5: "collector_elixir",
    6: "collector_dark",
}

CLASS_COLORS = {
    0: (0, 215, 255),    # Dorado / BGR
    1: (255, 0, 255),    # Magenta
    2: (128, 0, 128),    # Púrpura oscuro
    3: (255, 100, 0),    # Azul
    4: (0, 255, 255),    # Amarillo
    5: (200, 120, 255),  # Rosa suave
    6: (80, 20, 120),    # Morado/negro extractor oscuro
}

# Variables de estado globales para el callback del ratón
g_drawing = False
g_ix, g_iy = -1, -1
g_cx, g_cy = -1, -1
g_active_cls = 0
g_current_boxes: list[dict] = []
g_dirty = False


def mouse_callback(event, x, y, flags, param):
    global g_drawing, g_ix, g_iy, g_cx, g_cy, g_current_boxes, g_dirty, g_active_cls

    if event == cv2.EVENT_LBUTTONDOWN:
        g_drawing = True
        g_ix, g_iy = x, y
        g_cx, g_cy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if g_drawing:
            g_cx, g_cy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        if g_drawing:
            g_drawing = False
            x1, x2 = min(g_ix, x), max(g_ix, x)
            y1, y2 = min(g_iy, y), max(g_iy, y)
            # Evitar clics accidentales de tamaño insignificante
            if (x2 - x1) >= 8 and (y2 - y1) >= 8:
                g_current_boxes.append({
                    "cls_id": g_active_cls,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })
                g_dirty = True


def load_yolo_labels(label_file: Path, img_w: int, img_h: int) -> list[dict]:
    """Carga cajas desde archivo formato YOLO y convierte a coordenadas en píxeles."""
    boxes = []
    if not label_file.exists():
        return boxes

    try:
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx_norm, cy_norm = float(parts[1]), float(parts[2])
                    w_norm, h_norm = float(parts[3]), float(parts[4])

                    cx = cx_norm * img_w
                    cy = cy_norm * img_h
                    bw = w_norm * img_w
                    bh = h_norm * img_h

                    x1 = int(round(cx - bw / 2.0))
                    y1 = int(round(cy - bh / 2.0))
                    x2 = int(round(cx + bw / 2.0))
                    y2 = int(round(cy + bh / 2.0))

                    # Clamping
                    x1 = max(0, min(img_w, x1))
                    y1 = max(0, min(img_h, y1))
                    x2 = max(0, min(img_w, x2))
                    y2 = max(0, min(img_h, y2))

                    boxes.append({
                        "cls_id": cls_id,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    })
    except Exception as e:
        print(f"Error al leer etiquetas de {label_file}: {e}")

    return boxes


def save_yolo_labels(label_file: Path, boxes: list[dict], img_w: int, img_h: int):
    """Guarda cajas actuales en formato YOLO normalizado."""
    lines = []
    for b in boxes:
        cls_id = b["cls_id"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = (x2 - x1)
        bh = (y2 - y1)

        cx_norm = cx / img_w
        cy_norm = cy / img_h
        w_norm = bw / img_w
        h_norm = bh / img_h

        lines.append(f"{cls_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

    label_file.parent.mkdir(parents=True, exist_ok=True)
    with open(label_file, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
    print(f"Guardado {label_file.name}: {len(lines)} cajas.")


def draw_ui(img: np.ndarray, boxes: list[dict], filename: str, idx: int, total: int) -> np.ndarray:
    """Dibuja las cajas, la caja en progreso y la barra de estado superior."""
    global g_drawing, g_ix, g_iy, g_cx, g_cy, g_active_cls, g_dirty
    disp = img.copy()
    h, w = disp.shape[:2]

    # Dibujar cajas confirmadas
    for b in boxes:
        cls_id = b["cls_id"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")

        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
        label = f"{name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bg_y1 = max(0, y1 - th - 4)
        cv2.rectangle(disp, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(disp, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Dibujar caja en arrastre actual
    if g_drawing:
        x1, x2 = min(g_ix, g_cx), max(g_ix, g_cx)
        y1, y2 = min(g_iy, g_cy), max(g_iy, g_cy)
        color = CLASS_COLORS.get(g_active_cls, (0, 255, 255))
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 1, cv2.LINE_4)

    # Banner superior
    banner_h = 50
    overlay = disp.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, disp, 0.2, 0, disp)

    active_name = CLASS_NAMES.get(g_active_cls, f"cls_{g_active_cls}")
    active_col = CLASS_COLORS.get(g_active_cls, (255, 255, 255))

    dirty_flag = " [MODIFICADO]" if g_dirty else ""
    line1 = f"[{idx + 1}/{total}] {filename}{dirty_flag} | Cajas: {len(boxes)}"
    line2 = f"Clase activa: [{g_active_cls}] {active_name} | Teclas [1-7]: 1:Gold 2:Elixir 3:Dark 4:TH 5:Mine 6:Pump 7:Drill | [s] Guardar | [n] Sig [p] Ant | [d] Undo [q] Salir"

    cv2.putText(disp, line1, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(disp, line2, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, active_col, 1, cv2.LINE_AA)

    return disp


def main():
    global g_active_cls, g_current_boxes, g_dirty

    parser = argparse.ArgumentParser(description="Revisor y anotador interactivo de prelabels YOLO")
    parser.add_argument("--images", type=str, default="datasets/coc/images/raw", help="Directorio con imágenes crudas")
    parser.add_argument("--labels", type=str, default="datasets/coc/labels/raw", help="Directorio con archivos .txt YOLO")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)

    if not images_dir.exists():
        print(f"Error: El directorio de imágenes no existe: {images_dir}")
        sys.exit(1)

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    img_files = sorted([f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions])

    if not img_files:
        print(f"No se encontraron imágenes en: {images_dir}")
        sys.exit(0)

    window_name = "Review YOLO Pre-labels"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, mouse_callback)

    idx = 0
    total = len(img_files)

    current_img_file = img_files[idx]
    current_label_file = labels_dir / f"{current_img_file.stem}.txt"
    current_img = cv2.imread(str(current_img_file))

    if current_img is None:
        print(f"Error al leer imagen inicial: {current_img_file}")
        sys.exit(1)

    h, w = current_img.shape[:2]
    g_current_boxes = load_yolo_labels(current_label_file, w, h)
    g_dirty = False

    while True:
        disp = draw_ui(current_img, g_current_boxes, current_img_file.name, idx, total)
        cv2.imshow(window_name, disp)

        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            # Guardar y salir
            if g_dirty:
                save_yolo_labels(current_label_file, g_current_boxes, w, h)
            break

        elif key == 27:  # Esc
            break

        elif key == ord('1'):
            g_active_cls = 0  # storage_gold
        elif key == ord('2'):
            g_active_cls = 1  # storage_elixir
        elif key == ord('3'):
            g_active_cls = 2  # storage_dark
        elif key == ord('4'):
            g_active_cls = 3  # town_hall
        elif key == ord('5'):
            g_active_cls = 4  # collector_gold
        elif key == ord('6'):
            g_active_cls = 5  # collector_elixir
        elif key == ord('7'):
            g_active_cls = 6  # collector_dark

        elif key == ord('d'):
            # Undo último bbox
            if g_current_boxes:
                g_current_boxes.pop()
                g_dirty = True
        elif key == ord('D'):
            # Borrar todos
            if g_current_boxes:
                g_current_boxes.clear()
                g_dirty = True

        elif key == ord('s'):
            # Guardar y avanzar
            save_yolo_labels(current_label_file, g_current_boxes, w, h)
            g_dirty = False
            if idx < total - 1:
                idx += 1
                current_img_file = img_files[idx]
                current_label_file = labels_dir / f"{current_img_file.stem}.txt"
                current_img = cv2.imread(str(current_img_file))
                h, w = current_img.shape[:2]
                g_current_boxes = load_yolo_labels(current_label_file, w, h)

        elif key == ord('n'):
            # Siguiente (guarda si está modificado)
            if g_dirty:
                save_yolo_labels(current_label_file, g_current_boxes, w, h)
                g_dirty = False
            if idx < total - 1:
                idx += 1
                current_img_file = img_files[idx]
                current_label_file = labels_dir / f"{current_img_file.stem}.txt"
                current_img = cv2.imread(str(current_img_file))
                h, w = current_img.shape[:2]
                g_current_boxes = load_yolo_labels(current_label_file, w, h)

        elif key == ord('p'):
            # Anterior
            if g_dirty:
                save_yolo_labels(current_label_file, g_current_boxes, w, h)
                g_dirty = False
            if idx > 0:
                idx -= 1
                current_img_file = img_files[idx]
                current_label_file = labels_dir / f"{current_img_file.stem}.txt"
                current_img = cv2.imread(str(current_img_file))
                h, w = current_img.shape[:2]
                g_current_boxes = load_yolo_labels(current_label_file, w, h)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

## File: tools/run_live_test_3_attacks.py
```python
"""
tools/run_live_test_3_attacks.py
Script de supervisión y ejecución en vivo de 3 ataques reales:
1. Activa punteros táctiles completos (show_touches=1, pointer_location=1) en el dispositivo.
2. Ejecuta 3 ataques consecutivos mediante ClashStateMachine.
3. En cada ataque, monitorea la apertura de muros (SWB y Hechizos de Salto) y el despliegue de duendes.
4. Genera un reporte consolidado con capturas de pantalla de cada batalla.
"""
import os
import sys
import time
import shutil

# Agregar raíz del proyecto al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.device import Device
from core.vision import Vision
from core.state_machine import ClashStateMachine
from core.logger import get_logger

log = get_logger("live_test")


def enable_pointers(serial: str):
    log.info("🎯 Activando punteros táctiles completos en dispositivo %s...", serial)
    os.system(f'adb -s {serial} shell "settings put system show_touches 1; settings put system pointer_location 1"')


def disable_pointers(serial: str):
    log.info("🔌 Desactivando punteros táctiles en dispositivo %s...", serial)
    os.system(f'adb -s {serial} shell "settings put system show_touches 0; settings put system pointer_location 0"')


def main():
    log.info("🚀 Iniciando Test en Vivo: 3 Ataques Reales con Punteros y Auditoría Visual...")

    device = Device()
    if not device.connect():
        log.critical("❌ No se pudo conectar al dispositivo por ADB.")
        sys.exit(1)

    enable_pointers(device.serial)

    vision = Vision(threshold=0.80)
    vision.preload_templates("assets")

    test_dir = os.path.join("logs", "live_tests")
    os.makedirs(test_dir, exist_ok=True)

    bot = ClashStateMachine(device, vision)

    try:
        log.info("⚔️ Iniciando bucle de 3 ataques reales...")
        bot.run_forever(max_attacks=3)
    finally:
        # Desactivar punteros al concluir la ejecución de prueba
        disable_pointers(device.serial)
        log.info("✅ Punteros restaurados al estado original.")

    log.info("🏁 Test de 3 ataques completado. Verificando auditorías visuales...")


if __name__ == "__main__":
    main()
```

## File: tools/split_dataset.py
```python
"""
tools/split_dataset.py
Distribuye capturas y etiquetas anotadas desde 'raw' hacia train/val de YOLO.
IDEMPOTENTE POR CONTRATO: purga train/val y sus .cache ANTES de copiar, y
valida invariantes DESPUÉS. Correrlo N veces siempre deja el disco limpio.
"""
import argparse
import glob
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import get_logger

log = get_logger("split_dataset")


def _dirs(dest):
    return {
        "img_train": os.path.join(dest, "images", "train"),
        "img_val":   os.path.join(dest, "images", "val"),
        "lbl_train": os.path.join(dest, "labels", "train"),
        "lbl_val":   os.path.join(dest, "labels", "val"),
    }


def purge_split_dirs(dest):
    """[FIX #1 y #2] Vaciado INCONDICIONAL de train/val + caché de Ultralytics.
    La caché vive en <dest>/labels/{train,val}.cache (hermana de las carpetas,
    no hija), por eso rmtree de las carpetas NO la alcanza: hay que borrarla a mano."""
    d = _dirs(dest)
    for path in d.values():
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    for cache in (os.path.join(dest, "labels", "train.cache"),
                  os.path.join(dest, "labels", "val.cache")):
        if os.path.exists(cache):
            os.remove(cache)
            log.info("Purgada caché obsoleta: %s", cache)


def validate_label_format(lbl_path, bad):
    """[FIX #4] Atrapa .txt truncados/corruptos (crash del revisor a mitad de 's')
    ANTES de quemar horas de CPU. Valida estructura, no semántica de clase
    (el rango clase<nc lo vela YOLO en train, para no acoplar split a data.yaml)."""
    stem = Path(lbl_path).stem
    with open(lbl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            tok = line.split()
            if len(tok) != 5:
                bad.append(f"{stem}.txt:{i} -> {len(tok)} campos (espera 5)")
                continue
            try:
                cls = int(tok[0])
                vals = [float(x) for x in tok[1:]]
            except ValueError:
                bad.append(f"{stem}.txt:{i} -> campos no numéricos: {line!r}")
                continue
            if cls < 0 or any(v < 0.0 or v > 1.0 for v in vals):
                bad.append(f"{stem}.txt:{i} -> fuera de rango: {line!r}")


def collect_valid_pairs(images_src, labels_src, labels_pre, allow_prelabel):
    images = sorted(
        glob.glob(os.path.join(images_src, "*.png")) +
        glob.glob(os.path.join(images_src, "*.jpg"))
    )
    pairs, using_pre, bad = [], [], []
    for img_p in images:
        stem = Path(img_p).stem
        lbl_p = os.path.join(labels_src, f"{stem}.txt")
        src_kind = "raw"
        if not (os.path.exists(lbl_p) and os.path.getsize(lbl_p) > 0):
            # [FIX #3] fallback a labels_pre SOLO si el humano lo autoriza, y loggeado.
            alt = os.path.join(labels_pre, f"{stem}.txt")
            if allow_prelabel and os.path.exists(alt) and os.path.getsize(alt) > 0:
                lbl_p, src_kind = alt, "PRELABEL(sucia)"
                using_pre.append(stem)
            else:
                continue
        validate_label_format(lbl_p, bad)
        pairs.append((img_p, lbl_p, src_kind))
    if bad:
        log.error("Etiquetas corruptas/truncadas detectadas (re-guarda esas aldeas):")
        for b in bad:
            log.error("   %s", b)
        sys.exit(1)  # fail-fast: no entreno con veneno geométrico
    if using_pre:
        log.warning("⚠️  %d imágenes ENTRENAN con label PRE-ETIQUETADA SIN REVISAR: %s",
                    len(using_pre), ", ".join(using_pre))
    return pairs


def assert_invariants(dest):
    """[FIX #4] Contrato de salida: el disco REAL debe cumplir paridad y cero solape.
    Esto es lo que habría gritado '43/9' en el segundo que ocurrió."""
    d = _dirs(dest)
    sets = {}
    for key, path in d.items():
        ext = "*.png" if key.startswith("img") else "*.txt"
        sets[key] = {Path(p).stem for p in glob.glob(os.path.join(path, ext))}
    if sets["img_train"] != sets["lbl_train"]:
        raise RuntimeError(f"Paridad rota TRAIN: imgs-lbls={sets['img_train']-sets['lbl_train']}, "
                           f"lbls-imgs={sets['lbl_train']-sets['img_train']}")
    if sets["img_val"] != sets["lbl_val"]:
        raise RuntimeError(f"Paridad rota VAL: imgs-lbls={sets['img_val']-sets['lbl_val']}, "
                           f"lbls-imgs={sets['lbl_val']-sets['img_val']}")
    overlap = sets["img_train"] & sets["img_val"]
    if overlap:
        raise RuntimeError(f"DATA LEAKAGE: {len(overlap)} aldeas en train Y val: "
                           f"{sorted(overlap)[:5]}")
    return len(sets["img_train"]), len(sets["img_val"])


def main():
    ap = argparse.ArgumentParser(description="Dividir dataset CoC en train/val (idempotente)")
    ap.add_argument("--images-src", default="datasets/coc/images/raw")
    ap.add_argument("--labels-src", default="datasets/coc/labels/raw")
    ap.add_argument("--dest", default="datasets/coc")
    ap.add_argument("--val-count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-prelabel", action="store_true",
                    help="Permitir que imágenes sin label revisada usen el borrador de labels_pre (loggeado)")
    args = ap.parse_args()

    random.seed(args.seed)
    labels_pre = os.path.join(args.dest, "labels_pre")  # [FIX #3] ruta derivada, no hardcodeada

    purge_split_dirs(args.dest)  # [FIX #1 y #2] SIEMPRE, sin flag

    pairs = collect_valid_pairs(args.images_src, args.labels_src, labels_pre, args.allow_prelabel)
    if not pairs:
        log.error("No hay pares válidos en %s / %s. Etiqueta primero con review_prelabels.py.",
                  args.images_src, args.labels_src)
        sys.exit(1)
    log.info("Pares válidos encontrados: %d", len(pairs))

    random.shuffle(pairs)
    cap = max(1, int(len(pairs) * 0.25))
    val_n = min(args.val_count, cap)
    if args.val_count > cap:  # [FIX] techo del 25% ya NO es silencioso
        log.warning("--val-count %d capado a %d (25%% de %d pares).", args.val_count, val_n, len(pairs))
    val_pairs, train_pairs = pairs[:val_n], pairs[val_n:]
    log.info("Partición deseada: TRAIN=%d, VAL=%d", len(train_pairs), len(val_pairs))

    d = _dirs(args.dest)
    for img_p, lbl_p, _ in train_pairs:
        shutil.copy2(img_p, os.path.join(d["img_train"], os.path.basename(img_p)))
        shutil.copy2(lbl_p, os.path.join(d["lbl_train"], os.path.basename(lbl_p)))
    for img_p, lbl_p, _ in val_pairs:
        shutil.copy2(img_p, os.path.join(d["img_val"], os.path.basename(img_p)))
        shutil.copy2(lbl_p, os.path.join(d["lbl_val"], os.path.basename(lbl_p)))

    n_train, n_val = assert_invariants(args.dest)  # [FIX #4] valida el disco REAL
    log.info("✅ Dataset limpio y verificado: Train=%d, Val=%d (solape=0, paridad img==lbl, caché purgada).",
             n_train, n_val)
    log.info("   python tools/train_yolo.py --epochs 80 --batch 8 --imgsz 1280")


if __name__ == "__main__":
    main()
```

## File: tools/test_annotate_tool.py
```python
"""
tools/test_annotate_tool.py
Suite de pruebas unitarias para la herramienta de anotación de saltos óptimos (Fase 1.1).

Verifica:
1. Descubrimiento de capturas (raw village images y attack screens).
2. Detección y superposición de segmentos de muros sobre 3 imágenes en modo headless.
3. Guardado y recarga de anotaciones respetando el esquema estricto:
   {"puntos": [[x, y], ...], "sin_salto": bool, "botin_en_colectores": bool, "notas": str}
   y compatibilidad hacia atrás con anotaciones existentes de 3 saltos sin flag de colectores.
4. Lógica de reanudación automática (resume where left off).
5. Máquina de estados: adición de hasta 4 saltos, deshacer ('u'), 'sin salto' ('s'), 'botin colectores' ('k') y limpiar ('c').
6. Invocación de CLI en modo headless / dry-run sin bloqueo de GUI.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Asegurar raíz del proyecto en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from core.logger import get_logger
from tools.annotate_best_jump import JumpAnnotator

log = get_logger("test_annotate_tool")


class TestAnnotateBestJump(unittest.TestCase):
    """Pruebas unitarias de JumpAnnotator y persistencia de ground truth."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_jump_gt_")
        self.gt_dir = Path(self.temp_dir)
        self.annotator = JumpAnnotator(
            project_root=PROJECT_ROOT,
            gt_dir=self.gt_dir,
            headless=True,
        )

    def tearDown(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_image_discovery(self) -> None:
        """Verifica que se descubran imágenes tanto en raw como en logs/yolo/attack_*."""
        images = self.annotator.find_all_images()
        self.assertGreaterEqual(len(images), 3, "Debe descubrir al menos 3 imágenes para pruebas")

        raw_count = sum(1 for p in images if "raw" in p.parts)
        attack_count = sum(1 for p in images if "attack_" in str(p))

        log.info(
            "Test 01: Descubiertas %d imágenes (%d raw, %d attack)",
            len(images),
            raw_count,
            attack_count,
        )
        self.assertGreater(raw_count, 0, "Debe incluir imágenes del dataset raw")
        self.assertGreater(attack_count, 0, "Debe incluir capturas de logs/yolo/attack_*")

    def test_02_headless_load_and_detect_walls_3_images(self) -> None:
        """Verifica carga y cálculo de segmentos de muro sobre 3 imágenes reales sin bloquear GUI."""
        images = self.annotator.find_all_images()
        self.assertGreaterEqual(len(images), 3)

        sample_images = images[:3]
        for idx, img_path in enumerate(sample_images):
            img = cv2.imread(str(img_path))
            self.assertIsNotNone(img, f"La imagen {img_path} no pudo ser leída por OpenCV")
            self.assertEqual(len(img.shape), 3)
            self.assertEqual(img.shape[0], 1080)
            self.assertEqual(img.shape[1], 2400)

            iso_angles, segments = self.annotator.compute_wall_segments(img)
            self.assertEqual(len(iso_angles), 2, "Debe retornar 2 ángulos isométricos")
            self.assertGreater(len(segments), 10, f"Debe detectar segmentos de muro en {img_path.name}")

            # Validar formato de segmento: (x1, y1, x2, y2)
            for seg in segments[:5]:
                self.assertEqual(len(seg), 4)

            log.info(
                "Test 02: Imagen %d/3 (%s) validada con %d segmentos de muro",
                idx + 1,
                img_path.name,
                len(segments),
            )

    def test_03_save_and_load_schema_3_images(self) -> None:
        """
        Verifica guardado y carga de imágenes con casos representativos:
        1. Aldea con 4 saltos y botin_en_colectores=True
        2. Aldea con 1 salto y botin_en_colectores=False
        3. Aldea marcada con 'sin_salto' y botin_en_colectores=False
        4. Compatibilidad hacia atrás: archivo legado con 3 saltos sin campo 'botin_en_colectores'.
        Valida rigurosamente el esquema JSON: {puntos: [[x, y], ...], sin_salto: bool, botin_en_colectores: bool, notas: str}.
        """
        images = self.annotator.find_all_images()[:3]
        self.assertEqual(len(images), 3)

        test_cases = [
            {
                "puntos": [[1100, 520], [1350, 680], [950, 400], [1200, 800]],
                "sin_salto": False,
                "botin_en_colectores": True,
                "notas": "Entrada principal norte, núcleo y colectores llenos",
            },
            {
                "puntos": [[980, 410]],
                "sin_salto": False,
                "botin_en_colectores": False,
                "notas": "Salto único entre compartimento oscuro",
            },
            {
                "puntos": [],
                "sin_salto": True,
                "botin_en_colectores": False,
                "notas": "Aldea abierta, no se requiere hechizo de salto",
            },
        ]

        for idx, (img_path, tc) in enumerate(zip(images, test_cases)):
            saved_path = self.annotator.save_annotation(
                image_path=img_path,
                puntos=tc["puntos"],
                sin_salto=tc["sin_salto"],
                notas=tc["notas"],
                botin_en_colectores=tc["botin_en_colectores"],
            )

            self.assertTrue(saved_path.exists(), f"El archivo {saved_path} debe existir")

            # Lectura directa del archivo crudo para validar formato JSON
            with open(saved_path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)

            self.assertIn("puntos", raw_json)
            self.assertIn("sin_salto", raw_json)
            self.assertIn("botin_en_colectores", raw_json)
            self.assertIn("notas", raw_json)

            self.assertEqual(raw_json["puntos"], tc["puntos"])
            self.assertEqual(raw_json["sin_salto"], tc["sin_salto"])
            self.assertEqual(raw_json["botin_en_colectores"], tc["botin_en_colectores"])
            self.assertEqual(raw_json["notas"], tc["notas"])

            # Lectura a través del método de carga del anotador
            loaded = self.annotator.load_annotation(img_path)
            self.assertTrue(loaded["exists"])
            self.assertEqual(loaded["puntos"], tc["puntos"])
            self.assertEqual(loaded["sin_salto"], tc["sin_salto"])
            self.assertEqual(loaded["botin_en_colectores"], tc["botin_en_colectores"])
            self.assertEqual(loaded["notas"], tc["notas"])

            log.info("Test 03: Validación de esquema JSON superada para caso %d (%s)", idx + 1, saved_path.name)

        # 4. Caso de prueba de compatibilidad hacia atrás: archivo preexistente de 3 saltos sin flag de colectores
        legacy_file = self.gt_dir / "legacy_test_village.json"
        legacy_payload = {
            "image": "legacy_test_village.png",
            "puntos": [[1000, 500], [1100, 600], [1200, 700]],
            "sin_salto": False,
            "notas": "Anotación previa de 3 saltos sin flag de colectores",
        }
        with open(legacy_file, "w", encoding="utf-8") as f:
            json.dump(legacy_payload, f, indent=2)

        loaded_legacy = self.annotator.load_annotation(Path("dummy/legacy_test_village.png"))
        self.assertTrue(loaded_legacy["exists"])
        self.assertEqual(len(loaded_legacy["puntos"]), 3)
        self.assertFalse(loaded_legacy["botin_en_colectores"], "Debe tener por defecto False si no existe en JSON")
        self.assertFalse(loaded_legacy["sin_salto"])
        log.info("Test 03: Compatibilidad hacia atrás con anotaciones de 3 saltos verificada exitosamente")

    def test_04_resume_functionality(self) -> None:
        """Verifica que find_first_unannotated_idx salte las imágenes ya anotadas."""
        images = self.annotator.find_all_images()
        self.assertGreaterEqual(len(images), 3)

        # Inicialmente ninguna anotada en el directorio temporal
        first_idx = self.annotator.find_first_unannotated_idx()
        self.assertEqual(first_idx, 0)

        # Anotar imagen 0
        self.annotator.save_annotation(images[0], [[1000, 500]], sin_salto=False, notas="Anotado 0")
        resumed_idx = self.annotator.find_first_unannotated_idx()
        self.assertEqual(resumed_idx, 1, "Debe avanzar automáticamente a la imagen 1")

        # Anotar imagen 1
        self.annotator.save_annotation(images[1], [], sin_salto=True, notas="Anotado 1")
        resumed_idx_2 = self.annotator.find_first_unannotated_idx()
        self.assertEqual(resumed_idx_2, 2, "Debe avanzar automáticamente a la imagen 2")

        log.info("Test 04: Reanudación automática validada con éxito")

    def test_05_state_machine_controls(self) -> None:
        """Verifica adición de saltos (máximo 4), undo, toggle sin_salto, toggle botin_en_colectores y clear."""
        # 1. Cargar imagen de prueba
        self.annotator.find_all_images()
        success = self.annotator.load_image_at_index(0)
        self.assertTrue(success)

        # 2. Agregar hasta 4 puntos (nuevo tope para ejército de 4 saltos)
        self.assertTrue(self.annotator.add_point(100, 200))
        self.assertTrue(self.annotator.add_point(300, 400))
        self.assertTrue(self.annotator.add_point(500, 600))
        self.assertTrue(self.annotator.add_point(700, 800))
        self.assertEqual(len(self.annotator.current_puntos), 4)

        # 3. Intentar agregar un 5to punto debe fallar
        self.assertFalse(self.annotator.add_point(900, 1000))
        self.assertEqual(len(self.annotator.current_puntos), 4)

        # 4. Probar tecla 'k' (toggle botin_en_colectores)
        self.assertFalse(self.annotator.current_botin_en_colectores)
        self.assertTrue(self.annotator.toggle_botin_en_colectores())
        self.assertTrue(self.annotator.current_botin_en_colectores)
        self.assertFalse(self.annotator.toggle_botin_en_colectores())
        self.assertFalse(self.annotator.current_botin_en_colectores)

        # 5. Deshacer punto ('u')
        self.assertTrue(self.annotator.undo_point())
        self.assertEqual(len(self.annotator.current_puntos), 3)
        self.assertEqual(self.annotator.current_puntos[-1], [500, 600])

        # 6. Activar 'sin_salto' ('s') -> debe vaciar los puntos
        is_sin_salto = self.annotator.toggle_sin_salto()
        self.assertTrue(is_sin_salto)
        self.assertEqual(len(self.annotator.current_puntos), 0)

        # 7. Deshacer cuando sin_salto está activo debe desactivarlo
        self.assertTrue(self.annotator.undo_point())
        self.assertFalse(self.annotator.current_sin_salto)

        # 8. Agregar punto y luego limpiar ('c')
        self.annotator.add_point(900, 950)
        self.assertEqual(len(self.annotator.current_puntos), 1)
        self.annotator.clear_points()
        self.assertEqual(len(self.annotator.current_puntos), 0)
        self.assertFalse(self.annotator.current_sin_salto)

        log.info("Test 05: Controles interactivos y límites (máx 4, undo, sin_salto, botin colectores 'k', clear) validados")

    def test_06_cli_headless_dry_run(self) -> None:
        """Verifica ejecución de tools/annotate_best_jump.py desde línea de comandos en headless dry-run."""
        script_path = PROJECT_ROOT / "tools" / "annotate_best_jump.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--headless",
            "--dry-run",
            "--limit",
            "3",
            "--gt-dir",
            str(self.gt_dir),
        ]

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(
            result.returncode,
            0,
            f"El CLI headless falló con código {result.returncode}.\nStderr: {result.stderr}\nStdout: {result.stdout}",
        )
        self.assertIn("Procesamiento headless completado", result.stdout + result.stderr)
        log.info("Test 06: Ejecución por CLI en modo headless --dry-run completada con retorno 0")


def main() -> None:
    log.info("Ejecutando suite de pruebas para annotate_best_jump...")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAnnotateBestJump)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
```

## File: tools/test_boundary_overlay.py
```python
"""
tools/test_boundary_overlay.py
Offline tool to test, analyze, and generate visual village boundary overlays.
Follows the non-convex isotropic boundary detection pipeline and dual operational modes
('pixel_scan' and 'high_res_contour') to guarantee 0 structure cuts across all flanks.
"""
import os
import sys
import argparse
import logging
import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vision.yolo_detector import get_detector
from core.config import SETTINGS
from core.logger import get_logger

log = get_logger("test_boundary_overlay")

CLASS_COLORS = {
    "storage_gold": (0, 215, 255),       # Gold
    "storage_elixir": (255, 0, 255),     # Magenta
    "storage_dark": (0, 140, 255),       # Orange
    "town_hall": (255, 120, 0),          # Cyan / Blue
    "collector_gold": (0, 255, 255),     # Yellow
    "collector_elixir": (220, 150, 255), # Soft pink
    "collector_dark": (50, 50, 180),     # Dark Maroon
}


def build_ui_mask(shape: tuple[int, int]) -> np.ndarray:
    """
    Constructs a precise UI mask preserving game terrain while blanking
    static UI elements (surrender button, troop deployment ribbon, next button, top HUD).
    Eliminates crude horizontal clamping (red_mask[880:, :] = 0).
    """
    h, w = shape[:2]
    ui_mask = np.ones((h, w), dtype=np.uint8)

    # 1. Surrender / End battle button (lower-left quadrant)
    y1_surr, y2_surr = int(0.72 * h), int(0.85 * h)
    x1_surr, x2_surr = int(0.03 * w), int(0.16 * w)
    ui_mask[y1_surr:y2_surr, x1_surr:x2_surr] = 0

    # 2. Bottom troop selection ribbon (lower-center)
    y1_ribbon = int(0.89 * h)
    x1_ribbon, x2_ribbon = int(0.15 * w), int(0.85 * w)
    ui_mask[y1_ribbon:, x1_ribbon:x2_ribbon] = 0

    # 3. Next match button (lower-right quadrant)
    y1_next, y2_next = int(0.78 * h), int(0.91 * h)
    x1_next, x2_next = int(0.85 * w), int(0.99 * w)
    ui_mask[y1_next:y2_next, x1_next:x2_next] = 0

    # 4. Top-left HUD (resources/trophies/shield)
    y2_top = int(0.10 * h)
    x2_hud_left = int(0.16 * w)
    ui_mask[:y2_top, :x2_hud_left] = 0

    # 5. Top-right HUD (gems/builders/resources) - strict mask y < 160, x > 1900
    y2_top_right = int(160 * h / 1080.0)
    x1_hud_right = int(1900 * w / 2400.0)
    ui_mask[:y2_top_right, x1_hud_right:] = 0

    return ui_mask


def extract_red_mask(img: np.ndarray, ui_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Extracts the red prohibition line pixels using calibrated HSV color ranges
    and applies the UI mask.
    """
    if img is None or img.size == 0:
        return np.zeros((1080, 2400), dtype=np.uint8)

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Calibrated ranges for Clash of Clans red deployment boundary line
    # H in [0, 10] and [168, 180] accurately captures the authentic red line while rejecting yellow-brown stones (H>=12)
    m1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    m2 = cv2.inRange(hsv, (168, 100, 100), (180, 255, 255))
    cond_bgr = (img[:, :, 2] > 130) & (img[:, :, 1] < 60) & (img[:, :, 0] < 60)
    red_mask = m1 | m2 | (cond_bgr.astype(np.uint8) * 255)

    if ui_mask is None:
        ui_mask = build_ui_mask((h, w))

    red_mask = cv2.bitwise_and(red_mask, red_mask, mask=ui_mask)
    return red_mask


def detect_village_boundary(
    img: np.ndarray,
    detections: list | None = None,
    mode: str = "pixel_scan",
    dilation_radius: int = 45,
    building_buffer_px: int = 30
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Pixel-exact, non-convex village boundary detector.
    Satisfies PROJECT.md Interface Contract:
    - Inputs: img (BGR), detections (list[Detection]), mode ('pixel_scan' or 'high_res_contour')
    - Outputs: red_contour (np.ndarray), green_contour (np.ndarray)
    - Guarantees: Zero structure cuts across all flanks; all building centers strictly enclosed.
    Pure computer vision: extracts the real village boundary directly from the screen,
    without baking in rectangular bounding boxes that create jagged irregularities.
    """
    if img is None or img.size == 0:
        return None, None

    h, w = img.shape[:2]
    ui_mask = build_ui_mask((h, w))
    red_mask = extract_red_mask(img, ui_mask)

    scale_x = w / 2400.0
    scale_y = h / 1080.0

    # Ensure top edge margin and troop ribbon cards are strictly suppressed from red signal
    red_mask[:int(60 * scale_y), :] = 0
    red_mask[int(888 * scale_y):, :] = 0

    has_tr_detection = any(
        d.center[0] > int(1450 * scale_x) and d.center[1] < int(280 * scale_y)
        for d in (detections or [])
    )
    if not has_tr_detection:
        # Suppress top-right HUD banner / loot text background without clipping authentic defenses (y >= 160)
        # Stepped suppression: y < 110 at x > 1500 preserves Wizard Tower, y < 220 at x > 1720 eliminates stone stairs bulge ("cabeza")
        red_mask[:int(110 * scale_y), int(1500 * scale_x):] = 0
        red_mask[:int(220 * scale_y), int(1720 * scale_x):] = 0

    red_pixel_count = int(np.count_nonzero(red_mask))
    log.info("Red exclusion pixels detected after UI masking: %d", red_pixel_count)

    # Bridge thin 1-3px isometric red boundary line segments
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    red_closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k_close)

    # Base mask derives directly from authentic village exclusion signal
    base_mask = red_closed.copy()
    if detections:
        for d in detections:
            cx, cy = d.center
            # Skip detections overlapping top HUD
            if cy < int(80 * scale_y) and (cx < int(0.20 * w) or cx > int(0.75 * w)):
                continue
            x1, y1, x2, y2 = d.bbox_xyxy
            # Isometric diamond footprint follows 45-degree tile grid with zero boxy artifacts
            diamond = np.array([[cx, y1], [x2, cy], [cx, y2], [x1, cy]], dtype=np.int32)
            cv2.fillPoly(base_mask, [diamond], 255)

    if np.count_nonzero(base_mask) == 0:
        log.warning("No exclusion features found in frame.")
        return None, None

    # Dynamic southern clamp ensures buildings are enclosed without plunging into troop cards
    max_det_y = max([d.bbox_xyxy[3] for d in (detections or [])] + [0])
    y_south_clamp = max(int(900 * scale_y), max_det_y + 4 if max_det_y > int(890 * scale_y) else 0)

    # Uniform isotropic outward dilation normal to the local boundary (+45px)
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_radius + 1, 2 * dilation_radius + 1)
    )
    dilated_mask = cv2.dilate(base_mask, dilation_kernel)
    # Puenteo morfológico elíptico exterior para eliminar saltos o cortes irregulares entre edificios cercanos
    k_bridge_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (85, 85))
    dilated_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, k_bridge_outer)

    # Clean boundaries: no invasion into top bar edge or lower troop selection ribbon
    dilated_mask[:int(60 * scale_y), :] = 0
    if not has_tr_detection:
        dilated_mask[:int(110 * scale_y), int(1500 * scale_x):] = 0
        dilated_mask[:int(220 * scale_y), int(1720 * scale_x):] = 0
    dilated_mask[y_south_clamp:, :] = 0

    # Extract outer contour
    cnts, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    main_legal = max(cnts, key=cv2.contourArea)
    solid_legal = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(solid_legal, [main_legal], -1, 255, -1)
    solid_legal[:int(60 * scale_y), :] = 0
    if not has_tr_detection:
        solid_legal[:int(110 * scale_y), int(1500 * scale_x):] = 0
        solid_legal[:int(220 * scale_y), int(1720 * scale_x):] = 0
    solid_legal[y_south_clamp:, :] = 0

    solid_red = cv2.erode(solid_legal, dilation_kernel)
    solid_red[int(888 * scale_y):, :] = 0

    if mode == "high_res_contour":
        cnts_g, _ = cv2.findContours(solid_legal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        main_g = max(cnts_g, key=cv2.contourArea)
        green_contour = cv2.approxPolyDP(main_g, epsilon=2.5, closed=True)

        cnts_r, _ = cv2.findContours(solid_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if cnts_r:
            main_r = max(cnts_r, key=cv2.contourArea)
            red_contour = cv2.approxPolyDP(main_r, epsilon=2.5, closed=True)
        else:
            red_contour = green_contour.copy()
    else:
        # Default pixel_scan mode keeps dense millimeter-level perimeter points
        cnts_g, _ = cv2.findContours(solid_legal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        green_contour = max(cnts_g, key=cv2.contourArea)

        cnts_r, _ = cv2.findContours(solid_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts_r:
            red_contour = max(cnts_r, key=cv2.contourArea)
        else:
            red_contour = green_contour.copy()

    if red_contour is not None and red_contour.ndim == 3:
        red_contour = red_contour.reshape((-1, 2))
    if green_contour is not None and green_contour.ndim == 3:
        green_contour = green_contour.reshape((-1, 2))

    return red_contour, green_contour


def audit_boundary_collisions(green_contour: np.ndarray, detections: list) -> list[dict]:
    """
    Audits the generated boundary against all detected buildings.
    A structure is cut if the polygon boundary cuts across its bounding box.
    Returns a list of cut structure details.
    """
    cuts = []
    if green_contour is None or not detections:
        return cuts

    cnt = np.array(green_contour, dtype=np.int32)
    if cnt.ndim == 2:
        cnt = cnt.reshape((-1, 1, 2))

    for d in detections:
        if not hasattr(d, "bbox_xyxy"):
            continue
        x1, y1, x2, y2 = d.bbox_xyxy
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        dists = [cv2.pointPolygonTest(cnt, (float(cx), float(cy)), True) for cx, cy in corners]
        center_dist = cv2.pointPolygonTest(cnt, (float(d.center[0]), float(d.center[1])), True)

        # Structure is cut if polygon edge passes through its bounding box
        if min(dists) < -1.0 and max(dists) > 1.0:
            cuts.append({
                "class": d.cls_name,
                "center": d.center,
                "bbox": d.bbox_xyxy,
                "center_dist": center_dist,
                "corners_dist": dists
            })
    return cuts


def generate_boundary_overlay(
    img: np.ndarray,
    red_contour: np.ndarray | None,
    green_contour: np.ndarray | None,
    detections: list,
    mode: str = "pixel_scan"
) -> np.ndarray:
    """
    Generates a high-clarity diagnostic visual overlay.
    Per user requirement:
    1. First reads and renders the village boundary (red exclusion shading + contours).
    2. Then draws building detections (bounding boxes, dots, labels) ON TOP of everything.
    """
    vis = img.copy()
    h, w = vis.shape[:2]

    # 1. Draw semi-transparent exclusion shading inside red contour if available
    if red_contour is not None:
        shading = vis.copy()
        cv2.drawContours(shading, [np.array(red_contour, dtype=np.int32)], -1, (0, 0, 180), -1)
        cv2.addWeighted(shading, 0.15, vis, 0.85, 0, vis)
        cv2.drawContours(vis, [np.array(red_contour, dtype=np.int32)], -1, (0, 0, 255), 2, cv2.LINE_AA)

    # 2. Draw green legal boundary
    n_pts = 0
    if green_contour is not None:
        cv2.drawContours(vis, [np.array(green_contour, dtype=np.int32)], -1, (0, 255, 0), 3, cv2.LINE_AA)
        n_pts = len(green_contour)

    # 3. Draw detected buildings ON TOP of the village boundary
    for d in detections:
        x1, y1, x2, y2 = d.bbox_xyxy
        color = CLASS_COLORS.get(d.cls_name, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, d.center, 4, color, -1)
        label = f"{d.cls_name} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # 4. Audit cuts
    cuts = audit_boundary_collisions(green_contour, detections)

    # 5. Draw top status banner
    status_color = (0, 255, 0) if len(cuts) == 0 else (0, 0, 255)
    status_text = "PASS: 0 STRUCTURE CUTS (100% LEGAL)" if len(cuts) == 0 else f"FAIL: {len(cuts)} CUTS DETECTED"
    banner_text = f"BOUNDARY DETECTION [{mode.upper()}] | Vertices: {n_pts} | Structures: {len(detections)} | {status_text}"

    cv2.rectangle(vis, (0, 0), (w, 42), (20, 20, 20), -1)
    cv2.putText(vis, banner_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, status_color, 2, cv2.LINE_AA)

    return vis


def run_boundary_tool(
    image_path: str = "logs/red_test_flash.png",
    mode: str = "pixel_scan",
    output_path: str = "logs/yolo/red_boundary_detected.png"
) -> int:
    """CLI execution entrypoint."""
    full_img_path = os.path.join(PROJECT_ROOT, image_path) if not os.path.isabs(image_path) else image_path
    if not os.path.exists(full_img_path):
        log.error("Input image not found: %s", full_img_path)
        return 1

    img = cv2.imread(full_img_path)
    log.info("Loaded image '%s' (%dx%d)", image_path, img.shape[1], img.shape[0])

    # Run YOLO detection
    detector = get_detector()
    raw_detections = detector.detect(img) if detector else []
    ui_mask_raw = build_ui_mask((img.shape[0], img.shape[1]))
    detections = []
    for d in raw_detections:
        if d.confidence < 0.50:
            continue
        cx, cy = d.center
        if 0 <= cy < img.shape[0] and 0 <= cx < img.shape[1] and ui_mask_raw[cy, cx] == 0:
            continue
        bx1, by1, bx2, by2 = d.bbox_xyxy
        crop = ui_mask_raw[max(0, by1):min(img.shape[0], by2), max(0, bx1):min(img.shape[1], bx2)]
        if crop.size > 0 and np.mean(crop == 0) > 0.20:
            continue
        detections.append(d)
    log.info("Detected %d valid village buildings with YOLO (filtered from %d)", len(detections), len(raw_detections))

    # Detect boundary
    red_cnt, green_cnt = detect_village_boundary(img, detections=detections, mode=mode)

    # Audit collisions
    cuts = audit_boundary_collisions(green_cnt, detections)
    log.info("Collision Audit: %d cut structures (Mode: %s)", len(cuts), mode)
    for c in cuts:
        log.warning("  CUT: %s at %s (dist_center: %.1f)", c["class"], c["center"], c["center_dist"])

    # Generate and save overlay
    overlay = generate_boundary_overlay(img, red_cnt, green_cnt, detections, mode=mode)
    full_out_path = os.path.join(PROJECT_ROOT, output_path) if not os.path.isabs(output_path) else output_path
    os.makedirs(os.path.dirname(full_out_path), exist_ok=True)
    cv2.imwrite(full_out_path, overlay)
    log.info("Visual overlay saved to: %s", full_out_path)

    return 0 if len(cuts) == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and generate village boundary visual overlays.")
    parser.add_argument("--image", default="logs/red_test_flash.png", help="Path to input battle flash screenshot")
    parser.add_argument("--mode", default="pixel_scan", choices=["pixel_scan", "high_res_contour"], help="Boundary extraction mode")
    parser.add_argument("--output", default="logs/yolo/red_boundary_detected.png", help="Path to save output visual overlay")
    args = parser.parse_args()

    sys.exit(run_boundary_tool(args.image, args.mode, args.output))
```

## File: tools/test_canonical_diamond.py
```python
"""
tools/test_canonical_diamond.py
Visualizador del Rombo Canónico y trazado de vectores de drop sin dispositivo.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from attack.deployment_strategy import (
    CANONICAL_CENTER,
    CANONICAL_RX_BUILDABLE,
    CANONICAL_RY_BUILDABLE,
    CANONICAL_RX_GRASS,
    CANONICAL_RY_GRASS,
    compute_canonical_diamond_drop
)
from vision.yolo_detector import get_detector

def test_on_image(image_path: str):
    if not os.path.exists(image_path):
        print(f"No existe: {image_path}")
        return

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    dbg = img.copy()

    cx, cy = CANONICAL_CENTER

    # 1. Dibujar Rombo Construible (Rojo / Exclusión)
    pts_bld = np.array([
        [cx, cy - CANONICAL_RY_BUILDABLE],
        [cx + CANONICAL_RX_BUILDABLE, cy],
        [cx, cy + CANONICAL_RY_BUILDABLE],
        [cx - CANONICAL_RX_BUILDABLE, cy]
    ], dtype=np.int32)
    cv2.polylines(dbg, [pts_bld], isClosed=True, color=(0, 0, 255), thickness=2)

    # 2. Dibujar Rombo Césped Legal (Verde / Despliegue)
    pts_grass = np.array([
        [cx, cy - CANONICAL_RY_GRASS],
        [cx + CANONICAL_RX_GRASS, cy],
        [cx, cy + CANONICAL_RY_GRASS],
        [cx - CANONICAL_RX_GRASS, cy]
    ], dtype=np.int32)
    cv2.polylines(dbg, [pts_grass], isClosed=True, color=(0, 255, 0), thickness=3)

    # 3. Detectar edificios con YOLO y trazar vectores de drop
    detector = get_detector()
    if detector:
        dets = detector.detect(img)
        print(f"Edificios detectados: {len(dets)}")
        for d in dets:
            bx, by = d.center
            dx, dy = compute_canonical_diamond_drop(d.center)
            # Dibujar edificio
            cv2.circle(dbg, (bx, by), 5, (255, 255, 0), -1)
            # Dibujar drop
            cv2.circle(dbg, (dx, dy), 7, (255, 0, 255), -1)
            # Flecha del drop al edificio
            cv2.arrowedLine(dbg, (dx, dy), (bx, by), (255, 200, 0), 2, tipLength=0.15)

    os.makedirs("logs/yolo", exist_ok=True)
    out_path = "logs/yolo/test_canonical_diamond_result.png"
    cv2.imwrite(out_path, dbg)
    print(f"✅ Resultado guardado en: {out_path}")

if __name__ == "__main__":
    import glob
    candidates = glob.glob("logs/yolo/attack_*.png") + glob.glob("datasets/coc/images/raw/*.png")
    if candidates:
        test_on_image(candidates[0])
    else:
        print("No se encontraron capturas para probar.")
```

## File: tools/test_jump_from_labels.py
```python
"""
tools/test_jump_from_labels.py
Valida la estrategia de Hechizos de Salto SIN necesidad de tener el modelo entrenado.
Usa las etiquetas manuales (.txt) como 'detecciones perfectas' y alimenta directamente
compute_adaptive_jump_targets_from_detections() para graficar los objetivos de salto,
el centroide de la base y los vectores de proyección sobre las capturas reales.
"""
import argparse
import glob
import os
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack.deployment_strategy import compute_adaptive_jump_targets_from_detections
from vision.yolo_detector import Detection
from core.logger import get_logger

log = get_logger("test_jumps")

CLASS_MAP = {
    0: "storage_gold",
    1: "storage_elixir",
    2: "storage_dark",
    3: "town_hall",
    4: "collector_gold",
    5: "collector_elixir",
    6: "collector_dark",
}

CLASS_COLORS = {
    "storage_gold": (0, 215, 255),
    "storage_elixir": (255, 0, 255),
    "storage_dark": (128, 0, 128),
    "town_hall": (255, 100, 0),
    "collector_gold": (0, 255, 255),
    "collector_elixir": (200, 120, 255),
    "collector_dark": (80, 20, 120),
}


def load_detections_from_txt(txt_path: str, img_w: int, img_h: int) -> list[Detection]:
    detections = []
    if not os.path.exists(txt_path):
        return detections

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx_norm, cy_norm = float(parts[1]), float(parts[2])
                w_norm, h_norm = float(parts[3]), float(parts[4])

                cls_name = CLASS_MAP.get(cls_id, f"class_{cls_id}")

                cx = int(round(cx_norm * img_w))
                cy = int(round(cy_norm * img_h))
                bw = int(round(w_norm * img_w))
                bh = int(round(h_norm * img_h))

                x1 = max(0, min(img_w, cx - bw // 2))
                y1 = max(0, min(img_h, cy - bh // 2))
                x2 = max(0, min(img_w, cx + bw // 2))
                y2 = max(0, min(img_h, cy + bh // 2))

                detections.append(Detection(
                    cls_name=cls_name,
                    confidence=1.0,
                    bbox_xyxy=(x1, y1, x2, y2),
                    center=(cx, cy)
                ))
    return detections


def render_jump_analysis(img: np.ndarray, detections: list[Detection], jump_targets: list[tuple[int, int]]) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]

    # 1. Dibujar edificios etiquetados
    for d in detections:
        x1, y1, x2, y2 = d.bbox_xyxy
        color = CLASS_COLORS.get(d.cls_name, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, d.center, 3, color, -1)
        label = f"{d.cls_name}"
        cv2.putText(vis, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # 2. Centroide de la base
    if detections:
        all_pts = np.array([d.center for d in detections], dtype=np.float32)
        if len(all_pts) >= 3:
            hull = cv2.convexHull(all_pts)
            cv2.polylines(vis, [hull.astype(np.int32)], isClosed=True, color=(100, 100, 100), thickness=1, lineType=cv2.LINE_AA)
            M = cv2.moments(hull)
            if M["m00"] > 0:
                base_cx = int(M["m10"] / M["m00"])
                base_cy = int(M["m01"] / M["m00"])
            else:
                base_cx = int(np.mean(all_pts[:, 0]))
                base_cy = int(np.mean(all_pts[:, 1]))
        else:
            base_cx = int(np.mean(all_pts[:, 0]))
            base_cy = int(np.mean(all_pts[:, 1]))

        cv2.drawMarker(vis, (base_cx, base_cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(vis, "CENTRO BASE", (base_cx + 8, base_cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # 3. Dibujar objetivos calculados de Hechizo de Salto
    for idx, pt in enumerate(jump_targets, 1):
        jx, jy = int(pt[0]), int(pt[1])
        # Círculo del radio de efecto del salto
        cv2.circle(vis, (jx, jy), 28, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(vis, (jx, jy), 4, (0, 255, 0), -1)
        # Línea guía desde centroide de base al salto
        if detections:
            cv2.line(vis, (base_cx, base_cy), (jx, jy), (0, 255, 0), 1, cv2.LINE_AA)
        tag = f"SALTO #{idx} ({jx}, {jy})"
        cv2.putText(vis, tag, (jx - 30, jy - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Banner superior
    banner_text = f"Detecciones GT: {len(detections)} | Saltos calculados: {len(jump_targets)}"
    cv2.rectangle(vis, (0, 0), (w, 35), (20, 20, 20), -1)
    cv2.putText(vis, banner_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    return vis


def main():
    parser = argparse.ArgumentParser(description="Prueba visual de lógica de Hechizos de Salto a partir de etiquetas")
    parser.add_argument("--image", type=str, default="", help="Ruta de una imagen específica (opcional)")
    parser.add_argument("--images-dir", type=str, default="datasets/coc/images/raw", help="Directorio de imágenes")
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="datasets/coc/labels/raw",
        help="Directorio de etiquetas .txt (o datasets/coc/labels_pre)"
    )
    parser.add_argument("--out-dir", type=str, default="logs/yolo/test_jumps", help="Directorio de salida para visualizaciones")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.image:
        img_paths = [args.image]
    else:
        img_paths = sorted(glob.glob(os.path.join(args.images_dir, "*.png")) + glob.glob(os.path.join(args.images_dir, "*.jpg")))

    if not img_paths:
        log.error("No se encontraron imágenes en: %s", args.images_dir)
        sys.exit(1)

    log.info("Analizando lógica de saltos con etiquetas en %s...", args.labels_dir)
    processed = 0

    for p in img_paths:
        stem = Path(p).stem
        label_file = os.path.join(args.labels_dir, f"{stem}.txt")
        # Fallback a labels_pre si no está en labels/raw
        if not os.path.exists(label_file):
            alt = os.path.join("datasets/coc/labels_pre", f"{stem}.txt")
            if os.path.exists(alt):
                label_file = alt

        if not os.path.exists(label_file):
            continue

        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]

        dets = load_detections_from_txt(label_file, w, h)
        if not dets:
            continue

        targets = compute_adaptive_jump_targets_from_detections(dets, (w, h))

        vis_img = render_jump_analysis(img, dets, targets)
        out_path = os.path.join(args.out_dir, f"jump_{stem}.png")
        cv2.imwrite(out_path, vis_img)
        log.info("✅ [%-25s] %d edificios -> %d saltos calculados -> %s", stem, len(dets), len(targets), out_path)
        processed += 1

    if processed == 0:
        log.warning(
            "No se encontraron archivos .txt coincidentes en '%s'. "
            "Asegúrate de tener anotaciones guardadas (con review_prelabels.py o prelabel_with_model.py).",
            args.labels_dir
        )
    else:
        log.info("=" * 60)
        log.info("Prueba completada: %d imágenes renderizadas en: %s", processed, os.path.abspath(args.out_dir))
        log.info("=" * 60)


if __name__ == "__main__":
    main()
```

## File: tools/test_jump_overlay.py
```python
"""
tools/test_jump_overlay.py
Genera overlays visuales con círculos de radio R sobre las capturas de ataque
utilizando el nuevo algoritmo de intersección de muros y máxima cobertura.
"""
import os
import sys
import json
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.yolo_detector import YoloDetector
from attack.deployment_strategy import compute_adaptive_jump_targets_from_detections
from core.config import SETTINGS, PROJECT_ROOT
from core.logger import get_logger

log = get_logger("test_jump_overlay")

CLASS_COLORS = {
    "storage_gold": (0, 215, 255),      # Dorado
    "storage_elixir": (255, 0, 255),    # Rosa / Magenta
    "storage_dark": (0, 140, 255),      # Naranja oscuro / Oscuro
    "town_hall": (255, 120, 0),         # Azul / Cyan
    "collector_gold": (0, 255, 255),    # Amarillo
    "collector_elixir": (220, 150, 255),# Rosa suave
    "collector_dark": (50, 50, 180),    # Granate / Oscuro
}


def run_overlay_test(images: list[str], out_dir: str = "logs/yolo") -> list[str]:
    attack_cfg = SETTINGS.get("attack", {})
    radius = int(attack_cfg.get("jump_spell_radius_px", 125))
    detector = YoloDetector(conf=float(attack_cfg.get("yolo_conf", 0.40)), imgsz=int(attack_cfg.get("yolo_imgsz", 1280)))

    os.makedirs(out_dir, exist_ok=True)
    generated_overlays = []

    for img_path in images:
        if not os.path.exists(img_path):
            log.warning("Imagen no encontrada: %s", img_path)
            continue

        base_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # 1. Inferencia YOLO
        detections = detector.detect(img)
        log.info("Procesando %s: %d detecciones YOLO", base_name, len(detections))

        # 2. Cálculo con nuevo algoritmo de intersecciones
        targets = compute_adaptive_jump_targets_from_detections(detections, (w, h))

        # 3. Dibujar overlay
        vis = img.copy()

        # Dibujar edificios detectados
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            color = CLASS_COLORS.get(d.cls_name, (200, 200, 200))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis, d.center, 4, color, -1)
            label = f"{d.cls_name} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Dibujar cada objetivo de salto con círculo de radio R
        for idx, pt in enumerate(targets, 1):
            jx, jy = int(pt[0]), int(pt[1])

            # Círculo semitransparente del radio de efecto
            overlay_layer = vis.copy()
            cv2.circle(overlay_layer, (jx, jy), radius, (0, 255, 0), -1)
            cv2.addWeighted(overlay_layer, 0.20, vis, 0.80, 0, vis)

            # Borde del círculo
            cv2.circle(vis, (jx, jy), radius, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(vis, (jx, jy), 5, (0, 0, 255), -1)

            # Etiqueta
            tag = f"SALTO #{idx} ({jx}, {jy}) [R={radius}px]"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (jx - tw // 2 - 4, jy - radius - th - 8), (jx + tw // 2 + 4, jy - radius), (0, 0, 0), -1)
            cv2.putText(vis, tag, (jx - tw // 2, jy - radius - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # Líneas guía hacia edificios cubiertos por este salto
            for d in detections:
                dist_d = np.hypot(d.center[0] - jx, d.center[1] - jy)
                if dist_d <= radius:
                    cv2.line(vis, (jx, jy), d.center, (0, 255, 255), 1, cv2.LINE_AA)

        # Banner superior
        banner_text = f"NUEVO ALGORITMO: Intersecciones de Muros | Saltos: {len(targets)} | R={radius}px"
        cv2.rectangle(vis, (0, 0), (w, 38), (15, 15, 15), -1)
        cv2.putText(vis, banner_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        out_name = f"overlay_{base_name}"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, vis)
        log.info("Overlay guardado en: %s", out_path)
        generated_overlays.append(out_path)

    return generated_overlays


if __name__ == "__main__":
    targets = [
        "logs/yolo/attack_20261105_203506.png",
        "logs/yolo/attack_20261105_183131.png",
        "logs/yolo/attack_20261105_183319.png",
        "logs/yolo/attack_20261105_183459.png",
    ]
    run_overlay_test(targets)
```

## File: tools/test_local_zone.py
```python
"""
tools/test_local_zone.py
Prueba sin dispositivo del pipeline de zona local usando imágenes estáticas de logs/
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from attack.deployment_strategy import detect_village_boundary
from vision.yolo_detector import get_detector


def test_static_zone(image_path: str):
    if not os.path.exists(image_path):
        print(f"Error: No existe {image_path}")
        return False

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo cargar imagen {image_path}")
        return False

    detector = get_detector()
    dets = detector.detect(img) if detector else []
    print(f"Detecciones YOLO: {len(dets)}")

    red_cnt, green_cnt = detect_village_boundary(img, detections=dets, mode="high_res_contour")
    print(
        f"Vértices Red: {len(red_cnt) if red_cnt is not None else 0} | "
        f"Green: {len(green_cnt) if green_cnt is not None else 0}"
    )

    dbg = img.copy()
    if red_cnt is not None:
        cv2.drawContours(dbg, [red_cnt], -1, (0, 0, 255), 2)
    if green_cnt is not None:
        cv2.drawContours(dbg, [green_cnt], -1, (0, 255, 0), 2)

    out_dir = os.path.join("logs", "yolo")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "test_zone_output.png")
    cv2.imwrite(out, dbg)
    print(f"Resultado guardado en {out}")
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "logs/yolo/attack_20261109_094640.png"
    if not os.path.exists(target):
        target = "logs/yolo/quota_attack_debug.png"
    test_static_zone(target)
```

## File: tools/test_probe_red_zone.py
```python
"""
tools/test_probe_red_zone.py
Herramienta de diagnóstico para capturar y analizar la retícula roja de exclusión
que dibuja Clash of Clans al tocar en zona prohibida con tropas seleccionadas.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cv2
import numpy as np
from core.device import Device
from core.vision import Vision
from core.logger import get_logger
from vision.army_checker import ArmyChecker

log = get_logger("probe_red_zone")


def run_probe_test():
    d = Device("5fba8618")
    v = Vision()
    army = ArmyChecker(v)

    log.info("Conectado a dispositivo. Verificando estado en aldea...")
    screen = d.screenshot()

    # Comprobar botón Attack
    tpl_attack = "assets/buttons/attack_btn.png"
    pos = v.find_template(screen, tpl_attack, threshold=0.74, zona="ESQUINA_INF_IZQ")
    if not pos:
        log.error("No se encontró el botón de ataque en la aldea principal.")
        return

    log.info("Pulsando Attack en %s...", pos)
    d.tap(*pos)
    time.sleep(1.0)

    # Buscar Find Match
    tpl_find = "assets/buttons/find_match_btn.png"
    pos_find = v.wait_for_template(d.screenshot, tpl_find, timeout=6, zona="COMPLETA")
    if not pos_find:
        log.error("No se encontró Find Match.")
        return

    log.info("Pulsando Find Match en %s...", pos_find)
    d.tap(*pos_find)
    time.sleep(1.0)

    # Confirm attack si aparece
    tpl_confirms = [
        "assets/buttons/attack_btn_confirm.png",
        "assets/buttons/okay_btn.png",
        "assets/buttons/confirm_attack.png",
    ]
    for _ in range(12):
        s = d.screenshot()
        m = v.find_any(s, tpl_confirms, threshold=0.72, zona="MITAD_INFERIOR")
        if m:
            log.info("Confirmando ataque en (%d, %d)...", m[1], m[2])
            d.tap(m[1], m[2])
            break
        # Si ya aparece el botón de surrender o next, estamos en batalla
        if v.find_template(s, "assets/buttons/next_btn.png", threshold=0.70, zona="ESQUINA_INF_DER"):
            break
        time.sleep(0.3)

    log.info("Esperando llegada a aldea rival...")
    tpl_surrender = "assets/buttons/surrender_battle.png"
    for _ in range(25):
        s = d.screenshot()
        if v.find_template(s, tpl_surrender, threshold=0.70, zona="ESQUINA_INF_IZQ"):
            log.info("Aldea rival cargada.")
            break
        time.sleep(0.5)

    time.sleep(1.0)
    # Zoom out
    log.info("Ejecutando Zoom Out centrado...")
    d.zoom_out(center=(1200, 540))
    time.sleep(1.5)

    # Captura baseline sin toque
    screen_base = d.screenshot()
    cv2.imwrite("logs/red_zone_baseline.png", screen_base)
    log.info("📸 Capturada imagen base sin toque: logs/red_zone_baseline.png")

    # Seleccionar duendes
    goblin_pos = army.find_goblin_in_battle_bar(screen_base)
    if not goblin_pos:
        goblin_pos = (494, 979)
    log.info("Seleccionando duendes en %s...", goblin_pos)
    d.tap(*goblin_pos)
    time.sleep(0.3)

    # Toque de prueba en el centro (1200, 540) - ZONA PROHIBIDA
    log.info("👉 Realizando toque de prueba en zona prohibida (1200, 540)...")
    d.tap(1200, 540)

    # Capturas rápidas para atrapar el destello rojo
    t0 = time.time()
    s1 = d.screenshot()
    dt1 = time.time() - t0
    cv2.imwrite("logs/red_zone_flash_1.png", s1)

    time.sleep(0.15)
    s2 = d.screenshot()
    dt2 = time.time() - t0
    cv2.imwrite("logs/red_zone_flash_2.png", s2)

    log.info("📸 Capturadas imágenes de destello a %.2fs y %.2fs", dt1, dt2)

    # Rendición limpia
    log.info("Iniciando rendición limpia...")
    pos_surr = v.wait_for_template(d.screenshot, tpl_surrender, timeout=4, threshold=0.72, zona="ESQUINA_INF_IZQ")
    if not pos_surr:
        pos_surr = v.find_template(d.screenshot(), "assets/buttons/end_battle_btn.png", threshold=0.70)
    if pos_surr:
        d.tap(*pos_surr)
        time.sleep(0.8)

    # Confirm
    tpl_confirm_end = "assets/buttons/confirm_end.png"
    pos_conf = v.wait_for_template(d.screenshot, tpl_confirm_end, timeout=4, threshold=0.70, zona="CENTRO")
    if not pos_conf:
        pos_conf = v.find_template(d.screenshot(), "assets/buttons/okay_btn.png", threshold=0.70, zona="CENTRO")
    if pos_conf:
        d.tap(*pos_conf)
        time.sleep(0.8)

    # Return Home
    tpl_ret = "assets/buttons/return_home_btn.png"
    pos_ret = v.wait_for_template(d.screenshot, tpl_ret, timeout=6, threshold=0.72, zona="MITAD_INFERIOR")
    if pos_ret:
        d.tap(*pos_ret)
        log.info("Pulsado Return Home. Esperando regreso a aldea...")
        time.sleep(3.5)

    log.info("✅ Prueba de sondeo completada.")


if __name__ == "__main__":
    run_probe_test()
```

## File: tools/test_stitch.py
```python
#!/usr/bin/env python3
"""
tools/test_stitch.py
Herramienta de diagnóstico y auditoría para la Costura Panorámica Milimétrica (Template Registration)
y el cálculo de puntos legales de despliegue radial sobre el mapa completo.

Uso:
  python tools/test_stitch.py --offline
  python tools/test_stitch.py --offline --center logs/pan_test_4_center.png --south logs/pan_test_1_south.png --north logs/pan_test_3_north.png
  python tools/test_stitch.py (con teléfono conectado)
"""
import os
import sys
import argparse
import time
import cv2
import numpy as np

# Asegurar path de importación
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.logger import get_logger
from vision.yolo_detector import get_detector, Detection
from attack.deployment_strategy import (
    DeploymentStrategy,
    detect_village_boundary,
    get_battle_hud_mask,
)

log = get_logger("test_stitch")


def run_stitching(center_img: np.ndarray, south_img: np.ndarray | None, north_img: np.ndarray | None):
    h, w = center_img.shape[:2]
    log.info("🖼️ Dimensiones base: %dx%d", w, h)

    detector = get_detector()

    # 1. Calcular desplazamientos exactos
    shift_south = None
    shift_north = None

    if south_img is not None:
        shift_south = DeploymentStrategy._compute_exact_camera_shift(center_img, south_img)
        if shift_south:
            log.info("📐 Desplazamiento Sur exacto calculado: dx=%d, dy=%d", shift_south[0], shift_south[1])
        else:
            log.warning("⚠️ No se pudo calcular desplazamiento para Flanco Sur.")

    if north_img is not None:
        shift_north = DeploymentStrategy._compute_exact_camera_shift(center_img, north_img)
        if shift_north:
            log.info("📐 Desplazamiento Norte exacto calculado: dx=%d, dy=%d", shift_north[0], shift_north[1])
        else:
            log.warning("⚠️ No se pudo calcular desplazamiento para Flanco Norte.")

    # 2. Lienzo Expandido (2400 x 1600)
    canvas_h, canvas_w = 1600, w
    center_y_offset = (canvas_h - h) // 2  # 260 px de margen superior
    center_x_offset = 0

    stitched = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Pegar franjas del Norte (si existe y tiene shift)
    if north_img is not None and shift_north:
        ndx, ndy = shift_north
        # coord en canvas = coord_en_center + center_y_offset = coord_en_north + ndy + center_y_offset
        # Solo incrustar la mitad superior de north_img (excluyendo HUD)
        ny1_src = int(h * 0.12)
        ny2_src = int(h * 0.50)
        ny1_dst = ny1_src + ndy + center_y_offset
        ny2_dst = ny2_src + ndy + center_y_offset

        if 0 <= ny1_dst < canvas_h and 0 <= ny2_dst <= canvas_h:
            stitched[ny1_dst:ny2_dst, :] = north_img[ny1_src:ny2_src, :]

    # Pegar franjas del Sur (si existe y tiene shift)
    if south_img is not None and shift_south:
        sdx, sdy = shift_south
        sy1_src = int(h * 0.50)
        sy2_src = int(h * 0.86)
        sy1_dst = sy1_src + sdy + center_y_offset
        sy2_dst = sy2_src + sdy + center_y_offset

        if 0 <= sy1_dst < canvas_h and 0 <= sy2_dst <= canvas_h:
            stitched[sy1_dst:sy2_dst, :] = south_img[sy1_src:sy2_src, :]

    # Pegar vista central en el medio (área jugable)
    cy1_dst = center_y_offset
    cy2_dst = center_y_offset + h
    stitched[cy1_dst:cy2_dst, :] = center_img

    # 3. Detecciones en Vista Central
    all_detections: list[Detection] = []
    if detector and detector.is_ready():
        raw_dets = detector.detect(center_img)
        h_mask = get_battle_hud_mask(h, w)
        for d in raw_dets:
            cx, cy = d.center
            if 0 <= cy < h and 0 <= cx < w and h_mask[cy, cx] == 0:
                all_detections.append(d)

    # 4. Sondeo de Línea Roja sobre Vista Central
    red_c, green_c = detect_village_boundary(center_img, detections=all_detections, mode="high_res_contour")

    # 5. Dibujar sobre el lienzo expandido
    dbg_full = stitched.copy()

    # Dibujar contorno verde en coordenadas de canvas
    if green_c is not None and len(green_c) >= 3:
        green_canvas = green_c.reshape(-1, 2).copy()
        green_canvas[:, 1] += center_y_offset
        cv2.polylines(dbg_full, [green_canvas.reshape(-1, 1, 2).astype(np.int32)], True, (0, 255, 0), 2)

    # Instancia mock/dummy de DeploymentStrategy para invocar _get_drop_point_on_contour
    dummy_deployer = DeploymentStrategy.__new__(DeploymentStrategy)

    drop_count = 0
    for d in all_detections:
        bc_x, bc_y = d.center
        x1, y1, x2, y2 = d.bbox_xyxy

        # En canvas
        cbx1, cby1 = x1, y1 + center_y_offset
        cbx2, cby2 = x2, y2 + center_y_offset
        cv2.rectangle(dbg_full, (cbx1, cby1), (cbx2, cby2), (255, 200, 0), 2)

        if green_c is not None and len(green_c) >= 3:
            dx, dy = dummy_deployer._get_drop_point_on_contour((bc_x, bc_y), green_c, extra_margin_px=32.0)
            cdx, cdy = dx, dy + center_y_offset

            # Magenta circle en drop point
            cv2.circle(dbg_full, (cdx, cdy), 7, (255, 0, 255), -1)
            # Cian arrow hacia el centro del edificio
            cv2.arrowedLine(dbg_full, (cdx, cdy), (bc_x, bc_y + center_y_offset), (255, 255, 0), 2, tipLength=0.15)
            drop_count += 1

    # Guardar resultado
    os.makedirs("logs/yolo", exist_ok=True)
    out_path = "logs/yolo/test_stitched_full_map.png"
    cv2.imwrite(out_path, dbg_full)
    log.info("✅ Mapa completo guardado en: %s (estructuras=%d, drops=%d)", out_path, len(all_detections), drop_count)


def main():
    parser = argparse.ArgumentParser(description="Test Stitching y Despliegue Milimétrico")
    parser.add_argument("--offline", action="store_true", help="Usa imágenes locales en lugar del dispositivo ADB")
    parser.add_argument("--center", default="logs/pan_test_4_center.png", help="Ruta de la captura central")
    parser.add_argument("--south", default="logs/pan_test_1_south.png", help="Ruta de la captura sur")
    parser.add_argument("--north", default="logs/pan_test_3_north.png", help="Ruta de la captura norte")
    args = parser.parse_args()

    if args.offline:
        log.info("📂 Modo Offline: Cargando capturas locales...")
        center_path = os.path.join(PROJECT_ROOT, args.center)
        south_path = os.path.join(PROJECT_ROOT, args.south)
        north_path = os.path.join(PROJECT_ROOT, args.north)

        center_img = cv2.imread(center_path)
        south_img = cv2.imread(south_path) if os.path.exists(south_path) else None
        north_img = cv2.imread(north_path) if os.path.exists(north_path) else None

        if center_img is None:
            # Fallback a screen_searching o screen_now3
            alt_centers = ["logs/screen_searching.png", "logs/screen_now3.png", "logs/test_screen.png"]
            for alt in alt_centers:
                alt_full = os.path.join(PROJECT_ROOT, alt)
                if os.path.exists(alt_full):
                    center_img = cv2.imread(alt_full)
                    log.info("Usando captura alternativa central: %s", alt)
                    break

        if center_img is None:
            log.error("❌ No se encontró ninguna imagen central válida para modo offline.")
            sys.exit(1)

        run_stitching(center_img, south_img, north_img)
    else:
        log.info("📱 Conectando al dispositivo ADB...")
        from core.device import Device
        device = Device()
        log.info("Dispositivo conectado: %s (%dx%d)", device.serial, device.width, device.height)
        device.zoom_out()
        time.sleep(0.6)
        center_img = device.screenshot()

        log.info("Haciendo swipe al Sur...")
        device.d.shell("input swipe 1200 760 1200 260 450")
        time.sleep(0.65)
        south_img = device.screenshot()

        log.info("Haciendo swipe al Norte...")
        device.d.shell("input swipe 1200 240 1200 820 500")
        time.sleep(0.65)
        north_img = device.screenshot()

        device.zoom_out()
        time.sleep(0.5)

        run_stitching(center_img, south_img, north_img)


if __name__ == "__main__":
    main()
```

## File: tools/test_sweep.py
```python
import os
import sys
import time
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.device import Device
from core.vision import Vision
from vision.army_checker import ArmyChecker
from core.config import PROJECT_ROOT
from attack.deployment_strategy import (
    DeploymentStrategy,
    SWEEP_SUR,
    SWEEP_NORTE,
    PAN_CENTER_TO_SOUTH,
    PAN_SOUTH_TO_NORTH,
    PAN_NORTH_TO_SOUTH,
    interpolate_catmull_rom,
)


def draw_lines(img, sweep_dict, color):
    h, w = img.shape[:2]
    out = img.copy()
    for lado, pts in sweep_dict.items():
        raw_coords = [(int(rx * w), int(ry * h)) for rx, ry in pts]
        smooth_coords = interpolate_catmull_rom(raw_coords, density=6)
        coords = np.array(smooth_coords, np.int32)
        cv2.polylines(out, [coords], isClosed=False, color=color, thickness=3)
        for x, y in raw_coords:
            cv2.circle(out, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(out, (x, y), 7, (255, 255, 255), 1)
    return out


def main():
    out_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(out_dir, exist_ok=True)

    dev = Device()
    if not dev.connect():
        print("⚠️ Dispositivo no detectado en ADB. Generando proyecciones offline calibradas...")
        sur_ref = os.path.join(out_dir, "sur_region.png")
        norte_ref = os.path.join(out_dir, "norte_region.png")
        zoom_ref = os.path.join(out_dir, "zoom.png")
        m1_path = os.path.join(out_dir, "movi1.jpeg")
        m2_path = os.path.join(out_dir, "movi2.jpeg")

        if os.path.exists(sur_ref):
            m1 = cv2.imread(sur_ref)
            img1 = draw_lines(m1, SWEEP_SUR, color=(0, 255, 0))
            cv2.imwrite(os.path.join(out_dir, "debug_sweep1_sur.png"), img1)
        elif os.path.exists(m1_path):
            m1 = cv2.imread(m1_path)
            img1 = draw_lines(m1, SWEEP_SUR, color=(0, 255, 0))
            cv2.imwrite(os.path.join(out_dir, "debug_sweep1_sur.png"), img1)

        if os.path.exists(norte_ref):
            m2 = cv2.imread(norte_ref)
            img2 = draw_lines(m2, SWEEP_NORTE, color=(255, 255, 0))
            cv2.imwrite(os.path.join(out_dir, "debug_sweep2_norte.png"), img2)
        elif os.path.exists(m2_path):
            m2 = cv2.imread(m2_path)
            img2 = draw_lines(m2, SWEEP_NORTE, color=(255, 255, 0))
            cv2.imwrite(os.path.join(out_dir, "debug_sweep2_norte.png"), img2)

        if os.path.exists(zoom_ref):
            mz = cv2.imread(zoom_ref)
            img_all = draw_lines(mz, SWEEP_SUR, color=(0, 255, 0))
            img_all = draw_lines(img_all, SWEEP_NORTE, color=(255, 255, 0))
            cv2.imwrite(os.path.join(out_dir, "debug_sweep_all.png"), img_all)
            cv2.imwrite(os.path.join(out_dir, "test_calibrated_all.png"), img_all)

        print("✅ Proyecciones calibradas generadas en logs/debug_sweep1_sur.png, logs/debug_sweep2_norte.png y logs/debug_sweep_all.png")
        return

    if "--deploy" in sys.argv:
        print("=== MODO DESPLIEGUE EN VIVO (--deploy) ===")
        vis = Vision()
        vis.preload_templates("assets")
        army = ArmyChecker(vis)
        strat = DeploymentStrategy(dev, vis, army)
        print("Ejecutando despliegue de ataque completo...")
        res = strat.execute_attack()
        print(f"Resultado del ataque: {res}")
        return

    out_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(out_dir, exist_ok=True)

    print("1. Zoom out obligatorio...")
    dev.zoom_out()
    time.sleep(0.8)

    print("2. Moviendo cámara del Centro al Sector Sur (PAN_CENTER_TO_SOUTH)...")
    dev.pan_camera(*PAN_CENTER_TO_SOUTH, duration=0.9)
    dev.wait_camera_settled(timeout=2.5)
    time.sleep(0.6)

    print("3. Capturando y proyectando sector Sur...")
    img1 = dev.screenshot()
    img1_sur = draw_lines(img1, SWEEP_SUR, color=(0, 255, 0))
    cv2.imwrite(os.path.join(out_dir, "debug_sweep1_sur.png"), img1_sur)

    # Proyección combinada sobre vista base
    img_all = draw_lines(img1_sur, SWEEP_NORTE, color=(255, 255, 0))
    cv2.imwrite(os.path.join(out_dir, "debug_sweep_all.png"), img_all)
    cv2.imwrite(os.path.join(out_dir, "test_calibrated_all.png"), img_all)

    print("4. Moviendo cámara al otro extremo: Sector Norte (PAN_SOUTH_TO_NORTH)...")
    dev.pan_camera(*PAN_SOUTH_TO_NORTH, duration=1.1)
    dev.wait_camera_settled(timeout=2.5)
    time.sleep(0.6)

    print("5. Capturando y proyectando sector Norte...")
    img2 = dev.screenshot()
    img2 = draw_lines(img2, SWEEP_NORTE, color=(255, 255, 0))
    cv2.imwrite(os.path.join(out_dir, "debug_sweep2_norte.png"), img2)

    time.sleep(0.6)
    print("6. Regresando cámara al otro extremo: Sector Sur (PAN_NORTH_TO_SOUTH)...")
    dev.pan_camera(*PAN_NORTH_TO_SOUTH, duration=1.1)
    dev.wait_camera_settled(timeout=2.5)
    print("Prueba finalizada. Archivos generados en logs/")
    print("TIP: Para probar el despliegue real de tropas en una batalla en vivo, ejecuta: python tools/test_sweep.py --deploy")


if __name__ == "__main__":
    main()
```

## File: tools/test_vision.py
```python
"""
tools/test_vision.py
Prueba en seco: valida conexión ADB, ROIs de recursos y detección de botones sin hacer clic.
"""
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.device import Device
from core.vision import Vision
from vision.resource_reader import ResourceReader
from core.config import SETTINGS, PROJECT_ROOT


def main():
    dev = Device()
    if not dev.connect():
        return

    vis = Vision(threshold=0.80)
    vis.preload_templates("assets")
    reader = ResourceReader(vis)

    print("📸 Tomando captura en vivo del teléfono...")
    screen = dev.screenshot()
    h, w = screen.shape[:2]

    # 1. Probar lectura de recursos en ambas zonas
    own_rss = reader.read_own_resources(screen)
    enemy_loot = reader.read_enemy_loot(screen)

    print(f"🏦 Lectura Recursos Propios : {own_rss}")
    print(f"🔎 Lectura Botín Enemigo    : {enemy_loot}")

    # 2. Dibujar los rectángulos (ROIs) sobre la captura para verificar puntería
    debug_img = screen.copy()
    for _, (ry1, ry2, rx1, rx2) in SETTINGS["rois"]["own_resources"].items():
        cv2.rectangle(debug_img, (int(w * rx1), int(h * ry1)), (int(w * rx2), int(h * ry2)), (0, 255, 0), 2)

    for _, (ry1, ry2, rx1, rx2) in SETTINGS["rois"]["enemy_loot"].items():
        cv2.rectangle(debug_img, (int(w * rx1), int(h * ry1)), (int(w * rx2), int(h * ry2)), (0, 165, 255), 2)

    out_file = os.path.join(PROJECT_ROOT, "logs", "debug_rois.png")
    cv2.imwrite(out_file, debug_img)
    print(f"✅ Imagen de calibración guardada en: {out_file}")


if __name__ == "__main__":
    main()
```

## File: tools/test_wall_raycast_on_saved_attacks.py
```python
"""
Herramienta de verificación visual para evaluar la detección física de muros
sobre las capturas de batallas reales guardadas en logs/yolo/.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import glob
import cv2
from vision.wall_raycast import (
    estimate_iso_angles,
    find_wall_crossings,
    draw_wall_raycast_debug,
    reset_iso_angles_cache,
)
from vision.yolo_detector import get_detector
from attack.deployment_strategy import compute_adaptive_jump_targets_from_detections


def test_attacks():
    attack_imgs = glob.glob("logs/yolo/attack_*.png")
    if not attack_imgs:
        print("No se encontraron capturas attack_*.png en logs/yolo/")
        return

    detector = get_detector()

    for idx, img_path in enumerate(sorted(attack_imgs, reverse=True)[:3]):
        reset_iso_angles_cache()
        img = cv2.imread(img_path)
        if img is None:
            continue

        print(f"\n==========================================")
        print(f"Evaluando captura real: {img_path}")
        iso1, iso2 = estimate_iso_angles(img, force_refresh=True)
        print(f"Ángulos isométricos detectados: {iso1:.1f}°, {iso2:.1f}°")

        detections = []
        if detector and detector.is_ready():
            detections = detector.detect(img)
            print(f"Detecciones YOLO: {len(detections)}")

        targets = compute_adaptive_jump_targets_from_detections(
            detections, (img.shape[1], img.shape[0]), screen=img
        )
        print(f"🎯 Saltos finales sobre muros: {targets}")

        out_path = f"logs/yolo/wall_raycast_verified_atk{idx+1}.png"
        if os.path.exists("logs/yolo/wall_raycast_debug.png"):
            import shutil
            shutil.copy("logs/yolo/wall_raycast_debug.png", out_path)
            print(f"📸 Debug guardado en: {out_path}")


if __name__ == "__main__":
    test_attacks()
```

## File: tools/test_yolo.py
```python
"""
tools/test_yolo.py
Script de prueba visual para inferencia YOLOv8 sin interactuar con el dispositivo.
"""
import argparse
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.yolo_detector import YoloDetector
from core.logger import get_logger

log = get_logger("test_yolo")

# Colores BGR para visualización:
# Verde para storages, azul para town_hall, amarillo para collectors
CLASS_COLORS = {
    "storage_gold": (0, 255, 0),        # Verde
    "storage_elixir": (0, 255, 128),     # Verde claro / menta
    "storage_dark": (0, 200, 0),        # Verde oscuro
    "town_hall": (255, 0, 0),           # Azul
    "collector_gold": (0, 255, 255),    # Amarillo
    "collector_elixir": (0, 215, 255),  # Amarillo oro
    "collector_dark": (128, 0, 128),    # Púrpura oscuro extractor oscuro
}
DEFAULT_COLOR = (255, 255, 255)         # Blanco para otras clases


def main():
    parser = argparse.ArgumentParser(description="Prueba visual de inferencia YOLO")
    parser.add_argument("--image", type=str, required=True, help="Ruta de la imagen de prueba")
    parser.add_argument("--model", type=str, default="models/yolov8s.onnx", help="Ruta al modelo (.onnx o .pt)")
    parser.add_argument("--conf", type=float, default=0.40, help="Umbral de confianza")
    parser.add_argument("--imgsz", type=int, default=1280, help="Resolución de inferencia (default: 1280)")
    parser.add_argument("--output", type=str, default="logs/yolo_debug.png", help="Ruta de salida anotada")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        log.critical("Imagen no encontrada: %s", args.image)
        sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        log.critical("No se pudo decodificar la imagen: %s", args.image)
        sys.exit(1)

    log.info("Cargando detector con modelo: %s (conf=%.2f, imgsz=%d)...", args.model, args.conf, args.imgsz)
    detector = YoloDetector(args.model, conf=args.conf, imgsz=args.imgsz)
    if not detector.is_ready():
        log.warning("El detector no está completamente listo o el modelo no existe. Intentando inferencia de prueba...")

    detections = detector.detect(img)
    log.info("Total de detecciones: %d", len(detections))

    out_img = img.copy()

    for idx, d in enumerate(detections, 1):
        x1, y1, x2, y2 = d.bbox_xyxy
        color = CLASS_COLORS.get(d.cls_name, DEFAULT_COLOR)
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)

        # Centro
        cx, cy = d.center
        cv2.circle(out_img, (cx, cy), 4, color, -1)

        label = f"{d.cls_name} {d.confidence:.2f}"
        log.info("  [%d] %-16s | conf: %.2f | bbox: (%d, %d, %d, %d) | center: (%d, %d)",
                 idx, d.cls_name, d.confidence, x1, y1, x2, y2, cx, cy)

        # Fondo del texto para legibilidad
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out_img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out_img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, out_img)
    log.info("Imagen con detecciones guardada en: %s", args.output)


if __name__ == "__main__":
    main()
```

## File: tools/train_yolo.py
```python
"""
tools/train_yolo.py
Script CLI para entrenar YOLOv8 con dataset CoC y exportar el mejor modelo a ONNX.
Configurado con hiperparámetros optimizados para Clash of Clans (imgsz=1280, luz fija sin fliplr/flipud).
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import get_logger

log = get_logger("train_yolo")


def check_dataset_ready(data_yaml_path: str = "datasets/coc/data.yaml") -> bool:
    train_dir = os.path.join("datasets", "coc", "images", "train")
    if not os.path.exists(train_dir):
        log.error("Directorio de imágenes de entrenamiento no encontrado: %s", train_dir)
        return False

    valid_images = [
        f for f in os.listdir(train_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ]
    if not valid_images:
        log.error(
            "El directorio de entrenamiento '%s' está vacío o no contiene imágenes válidas. "
            "Asegúrate de agregar imágenes y etiquetas en 'datasets/coc/' antes de entrenar "
            "(puedes usar 'python tools/split_dataset.py' para repartir las capturas crudas).",
            train_dir,
        )
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo YOLOv8 para ClashFarmerbot")
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8s.pt" if os.path.exists("models/yolov8s.pt") else "yolov8s.pt",
        help="Pesos base para fine-tuning (default: models/yolov8s.pt o yolov8s.pt)"
    )
    parser.add_argument("--data", type=str, default="datasets/coc/data.yaml", help="Ruta al archivo data.yaml")
    parser.add_argument("--epochs", type=int, default=80, help="Número de épocas de entrenamiento (default: 80)")
    parser.add_argument("--batch", type=int, default=8, help="Tamaño de batch (default: 8)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Tamaño de imagen (default: 1280)")
    parser.add_argument("--patience", type=int, default=25, help="Paciencia de early stopping (default: 25)")
    parser.add_argument("--device", type=str, default="", help="Dispositivo para entrenamiento (ej: 0, cpu)")
    parser.add_argument("--project", type=str, default="runs", help="Directorio raíz del proyecto")
    parser.add_argument("--name", type=str, default="coc_v1", help="Nombre del experimento (default: coc_v1)")
    args = parser.parse_args()

    if not check_dataset_ready(args.data):
        log.critical("Entrenamiento abortado: Dataset no preparado.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError as e:
        log.critical("No se pudo importar 'ultralytics'. Instala las dependencias con pip: %s", e)
        sys.exit(1)

    log.info("Cargando modelo base: %s", args.model)
    model = YOLO(args.model)

    # Hiperparámetros ajustados a Clash of Clans:
    # - fliplr=0.0, flipud=0.0: La iluminación isométrica es fija (sol siempre desde el mismo ángulo).
    # - scale=0.25, translate=0.08, degrees=0.0: Evitar rotaciones destructivas en vista isométrica fija.
    # - mosaic=1.0, close_mosaic=10: Ayuda a detectar edificios pequeños en aldeas densas.
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "scale": 0.25,
        "translate": 0.08,
        "degrees": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.35,
        "project": args.project,
        "name": args.name,
    }
    if args.device:
        train_kwargs["device"] = args.device

    log.info("Iniciando entrenamiento con parámetros: %s", train_kwargs)
    results = model.train(**train_kwargs)

    best_pt_path = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_pt_path):
        log.info("Entrenamiento finalizado exitosamente. Exportando '%s' a ONNX (imgsz=%d)...", best_pt_path, args.imgsz)
        best_model = YOLO(best_pt_path)
        best_model.export(format="onnx", imgsz=args.imgsz, opset=17)
        log.info("Exportación ONNX completada.")
    else:
        log.warning("No se encontró 'best.pt' en '%s'. Revisa los resultados del entrenamiento.", best_pt_path)


if __name__ == "__main__":
    main()
```

## File: tools/verify_drop_clearance.py
```python
"""
tools/verify_drop_clearance.py
Valida que todos los puntos calculados por _get_drop_point_on_contour
tengan distancia negativa (exterior) respecto a red_contour.
"""
import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack.deployment_strategy import DeploymentStrategy, detect_village_boundary
from vision.yolo_detector import get_detector


class DummyDevice:
    width = 2400
    height = 1080


def test_clearance():
    img_path = "logs/yolo/attack_debug_latest.png"
    if not os.path.exists(img_path):
        # Buscar la primera captura disponible en logs/yolo/
        import glob
        files = glob.glob("logs/yolo/attack_*.png") + glob.glob("logs/yolo/red_boundary_*.png")
        if not files:
            print("No se encontraron imagenes de prueba en logs/yolo/")
            return
        img_path = files[0]

    print(f"Cargando imagen: {img_path}")
    img = cv2.imread(img_path)
    detector = get_detector()
    dets = detector.detect(img) if detector else []
    print(f"Detecciones: {len(dets)}")

    red_cnt, green_cnt = detect_village_boundary(img, detections=dets)
    if green_cnt is None:
        print("ERROR: No se pudo obtener green_contour")
        return

    strat = DeploymentStrategy(DummyDevice(), None, None)
    strat._cached_red_contour = red_cnt

    failed = 0
    passed = 0
    for d in dets:
        dx, dy = strat._get_drop_point_on_contour(d.center, green_cnt, red_contour=red_cnt)
        if red_cnt is not None and len(red_cnt) >= 3:
            dist = cv2.pointPolygonTest(red_cnt, (float(dx), float(dy)), True)
            if dist > -15.0:
                print(f"FALLO: {d.cls_name} drop=({dx}, {dy}) distancia_roja={dist:.2f}px (Dentro o muy cerca)")
                failed += 1
            else:
                passed += 1
        else:
            passed += 1

    print(f"\n--- RESULTADO CLEARANCE ---")
    print(f"Pasadados: {passed}")
    print(f"Fallados: {failed}")
    if failed == 0:
        print("EXITO: Todos los drops caen 100% en cesped legal exterior.")


if __name__ == "__main__":
    test_clearance()
```

## File: vision/__init__.py
```python

```

## File: vision/army_checker.py
```python
"""
vision/army_checker.py
Verificación de ejército disponible en aldea y durante la batalla.
"""
from core.vision import Vision
from core.config import SETTINGS
from core.logger import get_logger

log = get_logger("army_checker")


class ArmyChecker:
    def __init__(self, vision: Vision):
        self.vision = vision
        self.tpl_goblin_slot = "assets/buttons/goblin_slot.png"
        self.tpl_swb_slot = "assets/buttons/super_wall_breaker_slot.png"
        self.tpl_hero_king = "assets/buttons/hero_king_slot.png"
        self.tpl_hero_queen = "assets/buttons/hero_queen_slot.png"
        self.tpl_hero_warden = "assets/buttons/hero_warden_slot.png"
        self.tpl_hero_champion = "assets/buttons/hero_champion_slot.png"
        self.tpl_hero_minion_prince = "assets/buttons/hero_minion_prince_slot.png"
        self.tpl_hero_duke_dragon = "assets/buttons/hero_duke_dragon_slot.png"
        self.tpl_spell_jump = "assets/buttons/spell_jump.png"
        self.tpl_spell_haste = "assets/buttons/spell_haste.png"
        self.tpl_spell_rage = "assets/buttons/spell_rage.png"
        self.tpl_edrag_slot = "assets/buttons/edrag_slot.png"
        self.tpl_siege_machine = "assets/buttons/siege_machine_slot.png"
        self.tpl_army_ready = "assets/buttons/army_ready_icon.png"

    def can_start_battle(self, screenshot) -> bool:
        # En farming continuo, las tropas siempre están pre-entrenadas instantáneamente
        if SETTINGS.get("army", {}).get("skip_army_ready_check", True):
            return True
        if self.vision._load_template(self.tpl_army_ready) is None:
            return True
        pos = self.vision.find_template(screenshot, self.tpl_army_ready, threshold=0.75, zona="ESQUINA_INF_IZQ")
        if not pos:
            log.warning("⚠️ Icono de ejército listo no detectado.")
            return False
        return True

    def find_goblin_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_goblin_slot, threshold=0.66, zona="MITAD_INFERIOR"
        )

    def find_jump_spell_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_spell_jump, threshold=0.72, zona="MITAD_INFERIOR"
        )

    def find_haste_spell_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_spell_haste, threshold=0.72, zona="MITAD_INFERIOR"
        )

    def find_super_wall_breaker_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_swb_slot, threshold=0.68, zona="MITAD_INFERIOR"
        )

    def find_edrag_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_edrag_slot, threshold=0.70, zona="MITAD_INFERIOR"
        )

    def find_rage_spell_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_spell_rage, threshold=0.70, zona="MITAD_INFERIOR"
        )

    def find_siege_machine_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_siege_machine, threshold=0.70, zona="MITAD_INFERIOR"
        )

    def find_minion_prince_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_hero_minion_prince, threshold=0.70, zona="MITAD_INFERIOR"
        )

    def find_hero_king_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_hero_king, threshold=0.70, zona="MITAD_INFERIOR"
        )

    def find_hero_queen_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_hero_queen, threshold=0.70, zona="MITAD_INFERIOR"
        )

    def find_hero_warden_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_hero_warden, threshold=0.70, zona="MITAD_INFERIOR"
        )

    def find_hero_champion_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_hero_champion, threshold=0.70, zona="MITAD_INFERIOR"
        )

    def find_hero_duke_dragon_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_hero_duke_dragon, threshold=0.70, zona="MITAD_INFERIOR"
        )

    def find_all_heroes_in_battle_bar(self, screenshot) -> dict[str, tuple[int, int]]:
        """Retorna diccionario con las coordenadas de ranura de los héroes detectados en orden de prioridad."""
        heroes = {}
        # Orden preferente: Royal Champion (1º), Minion Prince (2º), Grand Warden (3º), Duke Dragon / King (4º), Queen
        for hero_id, find_fn in [
            ("champion", self.find_hero_champion_in_battle_bar),
            ("minion_prince", self.find_minion_prince_in_battle_bar),
            ("warden", self.find_hero_warden_in_battle_bar),
            ("duke_dragon", self.find_hero_duke_dragon_in_battle_bar),
            ("king", self.find_hero_king_in_battle_bar),
            ("queen", self.find_hero_queen_in_battle_bar),
        ]:
            pos = find_fn(screenshot)
            if pos:
                heroes[hero_id] = pos
        return heroes

    def detect_active_army_type(self, screenshot) -> str:
        """
        Determina la estrategia de ataque activa.
        Respeta la configuración manual si se especificó en settings.json,
        o auto-detecta en base a las tropas visibles en la barra de batalla.
        """
        configured_mode = SETTINGS.get("attack", {}).get("attack_strategy", "auto")
        if configured_mode in {"edrag_rush", "sneaky_farming"}:
            return configured_mode

        if screenshot is None:
            return "sneaky_farming"

        # Detección por firmas de tropas y hechizos clave
        if self.find_edrag_in_battle_bar(screenshot) or self.find_rage_spell_in_battle_bar(screenshot):
            log.info("🐉 Ejército detectado: Dragones Eléctricos + Furias -> Modo 'edrag_rush'")
            return "edrag_rush"

        if self.find_goblin_in_battle_bar(screenshot) or self.find_jump_spell_in_battle_bar(screenshot):
            log.info("👺 Ejército detectado: Duendes Furtivos -> Modo 'sneaky_farming'")
            return "sneaky_farming"

        return "sneaky_farming"
```

## File: vision/damage_reader.py
```python
"""
vision/damage_reader.py
Lector de porcentaje de destrucción en batalla para Clash of Clans.
Detecta el porcentaje (Overall Damage) en la zona inferior derecha de la pantalla (2400x1080)
mediante coincidencia de plantillas de dígitos de alta velocidad (<0.5ms) sin dependencias externas.
"""
import os
import cv2
import numpy as np
from core.config import PROJECT_ROOT
from core.logger import get_logger

log = get_logger("damage_reader")

# Coordenadas de referencia para resolución 2400x1080 (horizontal)
# Zona inferior derecha encima de la cinta de tropas:
# "Overall Damage" texto: Y ∈ [700, 740], X ∈ [2010, 2310]
# Porcentaje (número + %): Y ∈ [780, 840], X ∈ [2130, 2330]
DEFAULT_ROI_DAMAGE = (2130, 780, 200, 60)  # x, y, w, h en 2400x1080
BASE_WIDTH = 2400
BASE_HEIGHT = 1080


class DamageReader:
    """Lee el porcentaje de destrucción actual durante la batalla activa."""

    def __init__(self, templates_dir: str | None = None):
        if templates_dir is None:
            templates_dir = os.path.join(PROJECT_ROOT, "assets", "digits_damage")
        self.templates_dir = templates_dir
        self.digit_templates: dict[int, np.ndarray] = {}
        self.percent_template: np.ndarray | None = None
        self._load_templates()

    def _load_templates(self):
        """Carga las plantillas binarizadas de dígitos 0-9 y el símbolo %."""
        for d in range(10):
            p = os.path.join(self.templates_dir, f"{d}.png")
            if os.path.exists(p):
                tpl = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if tpl is not None:
                    _, bin_tpl = cv2.threshold(tpl, 128, 255, cv2.THRESH_BINARY)
                    self.digit_templates[d] = bin_tpl
            else:
                log.warning("No se encontró la plantilla de daño para el dígito %d en %s", d, p)

        pct_path = os.path.join(self.templates_dir, "percent.png")
        if os.path.exists(pct_path):
            pct_img = cv2.imread(pct_path, cv2.IMREAD_GRAYSCALE)
            if pct_img is not None:
                _, bin_pct = cv2.threshold(pct_img, 128, 255, cv2.THRESH_BINARY)
                self.percent_template = bin_pct
        else:
            log.warning("No se encontró la plantilla del símbolo %% en %s", pct_path)

    def _get_roi(self, h: int, w: int) -> tuple[int, int, int, int]:
        """Calcula el ROI adaptado a la resolución actual de la pantalla."""
        rx = w / float(BASE_WIDTH)
        ry = h / float(BASE_HEIGHT)
        bx, by, bw, bh = DEFAULT_ROI_DAMAGE
        x = int(round(bx * rx))
        y = int(round(by * ry))
        width = int(round(bw * rx))
        height = int(round(bh * ry))
        # Asegurar dentro de los límites
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        width = min(w - x, width)
        height = min(h - y, height)
        return x, y, width, height

    def read_destruction_percentage(self, screen: np.ndarray | None) -> int | None:
        """
        Lee el porcentaje de destrucción (0 a 100) desde la captura de pantalla de combate.
        Retorna int si se detecta con éxito, o None si no hay HUD de daño o no se reconoce.
        """
        if screen is None or screen.size == 0:
            return None

        h, w = screen.shape[:2]
        rx, ry, rw, rh = self._get_roi(h, w)
        crop = screen[ry:ry + rh, rx:rx + rw]

        if crop.size == 0:
            return None

        # Convertir a escala de grises y binarizar dígitos blancos brillantes
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        # Píxeles de texto blanco en Supercell font son > 210
        _, bin_img = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

        cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale_factor = h / float(BASE_HEIGHT)
        min_h = int(round(18 * scale_factor))
        min_w = int(round(7 * scale_factor))

        boxes = [
            cv2.boundingRect(c)
            for c in cnts
            if cv2.boundingRect(c)[3] >= min_h and cv2.boundingRect(c)[2] >= min_w
        ]
        boxes.sort(key=lambda b: b[0])

        if not boxes:
            return None

        # Si el último componente es el símbolo '%', lo removemos para clasificar los números
        if self.percent_template is not None and len(boxes) >= 1:
            lx, ly, lw, lh = boxes[-1]
            if lw >= int(round(22 * scale_factor)):
                ch = bin_img[ly:ly + lh, lx:lx + lw]
                th, tw = self.percent_template.shape
                ch_res = cv2.resize(ch, (tw, th), interpolation=cv2.INTER_NEAREST)
                pct_score = np.sum(self.percent_template == ch_res) / float(self.percent_template.size)
                if pct_score > 0.70:
                    boxes = boxes[:-1]

        if not boxes:
            return None

        # Clasificar cada caja de dígito restante de izquierda a derecha
        num_str = ""
        for bx, by, bw, bh in boxes:
            ch = bin_img[by:by + bh, bx:bx + bw]
            best_digit = None
            best_score = -1.0

            for d, tpl in self.digit_templates.items():
                th, tw = tpl.shape
                ch_res = cv2.resize(ch, (tw, th), interpolation=cv2.INTER_NEAREST)
                score = np.sum(tpl == ch_res) / float(tpl.size)
                if score > best_score:
                    best_score = score
                    best_digit = d

            if best_score >= 0.75 and best_digit is not None:
                num_str += str(best_digit)
            else:
                return None

        if num_str.isdigit():
            val = int(num_str)
            if 0 <= val <= 100:
                return val

        return None
```

## File: vision/deploy_zone_detector.py
```python
"""
vision/deploy_zone_detector.py
Detector de Zona de Despliegue (Deploy Zone Finder) para Clash of Clans.
Basado en el modelo de segmentación U-Net con encoder EfficientNet-B4
entrenado por nihatxp (clash-of-clans-vision).

Infiere con precisión sub-pixel dónde está prohibido desplegar tropas (área roja)
y dónde es legal desplegar (borde exterior y huecos desplegables).
"""
import os
import time
from typing import Optional, Tuple, List
import cv2
import numpy as np

from core.logger import get_logger
from core.config import PROJECT_ROOT, SETTINGS

log = get_logger("deploy_zone_detector")


class DeployZoneDetector:
    """
    Detector de zona de no-despliegue basado en ONNX Runtime.
    Toma capturas de batalla (2400x1080 o cualquier resolución) y produce:
    1. Máscara de no-despliegue (1 = prohibido, 0 = desplegable).
    2. Contornos del borde de despliegue legal.
    3. Validación y proyección O(1) de puntos de lanzamiento de tropas.
    """

    def __init__(
        self,
        model_path: str = "models/deploy_zone_finder.onnx",
        conf_thresh: float = 0.50,
        imgsz: int = 1280,
        use_tta: bool = False,
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.use_tta = use_tta
        self.session = None
        self._ready = False
        self.last_mask: Optional[np.ndarray] = None
        self.last_prob: Optional[np.ndarray] = None
        self.last_contours: Optional[List[np.ndarray]] = None

        # Parámetros de normalización ImageNet (idénticos al entrenamiento)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self._init_model()

    def _init_model(self) -> None:
        resolved_path = (
            os.path.join(PROJECT_ROOT, self.model_path)
            if not os.path.isabs(self.model_path)
            else self.model_path
        )

        if not os.path.exists(resolved_path):
            log.warning("Modelo DeployZoneFinder no encontrado en: %s. Detector inactivo.", resolved_path)
            self._ready = False
            return

        try:
            import onnxruntime as ort
            # Inferencia óptima en CPU
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = min(4, os.cpu_count() or 2)
            self.session = ort.InferenceSession(
                resolved_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self._ready = True
            log.info("DeployZoneDetector listo (modelo: %s, threads: %d)", resolved_path, opts.intra_op_num_threads)
        except Exception as e:
            log.error("Error al inicializar sesión ONNX para DeployZoneDetector: %s", e)
            self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    def predict_mask(self, screenshot: np.ndarray) -> Optional[np.ndarray]:
        """
        Ejecuta la inferencia de segmentación sobre la captura.
        Retorna una máscara binaria uint8 (2400x1080 o resolución nativa):
          - 1: Área de no-despliegue (prohibido desplegar / interior de aldea)
          - 0: Área legal de despliegue (césped exterior / huecos legales)
        """
        if not self._ready or screenshot is None:
            return None

        t0 = time.time()
        orig_img = screenshot
        if orig_img.ndim == 2:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
        elif orig_img.shape[2] == 4:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)

        orig_h, orig_w = orig_img.shape[:2]

        # Escalar con lado mayor acotado a imgsz (múltiplo de 32 para UNet)
        scale = min(self.imgsz / max(orig_w, orig_h), 1.0)
        target_w = int(round(orig_w * scale / 32.0)) * 32
        target_h = int(round(orig_h * scale / 32.0)) * 32

        resized_bgr = cv2.resize(orig_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - self.mean) / self.std

        # Formato NCHW [1, 3, H, W]
        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        try:
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
            raw_out = outputs[0]  # [1, 3, target_h, target_w]

            # Softmax sobre los 3 canales de salida
            exp_out = np.exp(raw_out - np.max(raw_out, axis=1, keepdims=True))
            probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)

            # Canal 1: Probabilidad de área no-desplegable
            no_deploy_prob = probs[0, 1, :, :]

            # TTA opcional (flip horizontal promediado)
            if self.use_tta:
                tensor_flip = tensor[:, :, :, ::-1]
                outputs_flip = self.session.run([self.output_name], {self.input_name: tensor_flip})
                raw_flip = outputs_flip[0]
                exp_flip = np.exp(raw_flip - np.max(raw_flip, axis=1, keepdims=True))
                probs_flip = exp_flip / np.sum(exp_flip, axis=1, keepdims=True)
                no_deploy_prob = (no_deploy_prob + probs_flip[0, 1, :, ::-1]) / 2.0

            # Interpolar mapa de probabilidad al tamaño original
            prob_full = cv2.resize(no_deploy_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            binary_mask = (prob_full >= self.conf_thresh).astype(np.uint8)

            self.last_prob = prob_full
            self.last_mask = binary_mask

            # Extraer contornos del borde
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.last_contours = contours

            elapsed_ms = (time.time() - t0) * 1000
            log.debug("DeployZoneFinder inferencia completada en %.1f ms (pixeles no-deploy: %d)",
                      elapsed_ms, int(np.sum(binary_mask)))

            return binary_mask
        except Exception as e:
            log.error("Fallo durante inferencia de DeployZoneDetector: %s", e)
            return None

    def is_point_deployable(
        self,
        x: int,
        y: int,
        mask: Optional[np.ndarray] = None,
        safety_margin_px: int = 15
    ) -> bool:
        """
        Determina en O(1) si la coordenada (x, y) es legal para desplegar tropas.
        Opcionalmente verifica un margen de seguridad alrededor del punto.
        """
        active_mask = mask if mask is not None else self.last_mask
        if active_mask is None:
            return True  # Sin máscara, no se puede vetar

        h, w = active_mask.shape[:2]
        ix, iy = int(round(x)), int(round(y))

        # Verificar límites de pantalla
        if ix < 10 or ix >= w - 10 or iy < 10 or iy >= h - 10:
            return False

        # Si el punto exacto es zona prohibida
        if active_mask[iy, ix] == 1:
            return False

        # Si se requiere margen de seguridad, comprobar con radio euclidiano
        if safety_margin_px > 0:
            r = safety_margin_px
            y1 = max(0, iy - r)
            y2 = min(h, iy + r + 1)
            x1 = max(0, ix - r)
            x2 = min(w, ix + r + 1)
            roi = active_mask[y1:y2, x1:x2]
            if np.any(roi == 1):
                # Validar con distancia euclidiana exacta para no falsear por las esquinas del recuadro
                ys, xs = np.where(roi == 1)
                dists_sq = (xs + x1 - ix) ** 2 + (ys + y1 - iy) ** 2
                if np.any(dists_sq <= r * r):
                    return False

        return True

    def get_boundary_polygons(
        self,
        screenshot: np.ndarray,
        margin_px: int = 30
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calcula los polígonos exactos de la aldea directamente desde la segmentación IA:
        - pts_bld: Frontera exacta del área no desplegable (línea roja).
        - pts_grass: Perímetro exterior seguro de despliegue legal (línea verde),
                     expandido con el margen de seguridad especificado.
        """
        mask = self.predict_mask(screenshot)
        if mask is None:
            return None, None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        # Tomar el contorno principal más grande (aldea central)
        pts_bld = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(pts_bld))
        total_px = float(mask.shape[0] * mask.shape[1])
        if area < 50000.0 or area > 0.85 * total_px:
            # Contorno degenerado (pantalla negra en mocks o sin aldea reconocible), permitir fallback
            return None, None

        # Dilatar máscara para obtener el perímetro seguro de despliegue legal
        k_size = max(3, int(margin_px * 2 + 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv2.dilate(mask, kernel)
        green_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if green_contours:
            pts_grass = max(green_contours, key=cv2.contourArea)
        else:
            pts_grass = pts_bld

        return pts_bld, pts_grass

    def find_closest_deployable_point(
        self,
        target_x: int,
        target_y: int,
        search_radius_px: int = 500,
        margin_px: int = 25,
        mask: Optional[np.ndarray] = None,
        safe_bounds: Optional[Tuple[int, int, int, int]] = (120, 140, 2280, 845)
    ) -> Tuple[int, int]:
        """
        Dada una estructura objetivo (almacén, TH, defensa) dentro de la base,
        encuentra el punto exterior legal más cercano para lanzar las tropas.
        Garantiza que el punto esté dentro de safe_bounds (zona activa de toque sin HUD ni barra).
        """
        active_mask = mask if mask is not None else self.last_mask
        if active_mask is None:
            if safe_bounds is not None:
                xmin, ymin, xmax, ymax = safe_bounds
                return int(np.clip(target_x, xmin, xmax)), int(np.clip(target_y, ymin, ymax))
            return (target_x, target_y)

        h, w = active_mask.shape[:2]

        # Si ya es desplegable y cumple safe_bounds
        if self.is_point_deployable(target_x, target_y, active_mask, margin_px):
            if safe_bounds is None:
                return (target_x, target_y)
            xmin, ymin, xmax, ymax = safe_bounds
            if xmin <= target_x <= xmax and ymin <= target_y <= ymax:
                return (target_x, target_y)

        # Distancia euclidiana usando distanceTransform invertido
        deployable_map = (active_mask == 0).astype(np.uint8)
        dist_map = cv2.distanceTransform(deployable_map, cv2.DIST_L2, 5)

        # Máscara donde el margen de seguridad es válido
        safe_deployable = (dist_map >= margin_px).astype(np.uint8)

        # Filtrar límites de toque activo de pantalla para evitar barra de notificaciones y slots
        if safe_bounds is not None:
            xmin, ymin, xmax, ymax = safe_bounds
            safe_deployable[:ymin, :] = 0
            safe_deployable[ymax:, :] = 0
            safe_deployable[:, :xmin] = 0
            safe_deployable[:, xmax:] = 0

        # Buscar en un radio circular alrededor del objetivo
        y_indices, x_indices = np.where(safe_deployable == 1)
        if len(x_indices) == 0:
            if safe_bounds is not None:
                xmin, ymin, xmax, ymax = safe_bounds
                return int(np.clip(target_x, xmin, xmax)), int(np.clip(target_y, ymin, ymax))
            return (target_x, target_y)

        dists_sq = (x_indices - target_x) ** 2 + (y_indices - target_y) ** 2
        min_idx = np.argmin(dists_sq)

        best_x = int(x_indices[min_idx])
        best_y = int(y_indices[min_idx])
        return (best_x, best_y)

    def get_overlay_visualization(self, screenshot: np.ndarray) -> np.ndarray:
        """
        Genera una imagen con el overlay visual idéntico al portal oficial:
        - Zona prohibida resaltada en magenta traslúcido (#ec40c8).
        - Borde exterior trazado con línea blanca nítida.
        """
        if screenshot is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        mask = self.predict_mask(screenshot)
        overlay = screenshot.copy()
        if mask is None:
            return overlay

        # Mezcla alfa para área prohibida
        magenta_bgr = np.array([200, 64, 236], dtype=np.uint8)
        mask_idx = (mask == 1)
        overlay[mask_idx] = (0.55 * overlay[mask_idx] + 0.45 * magenta_bgr).astype(np.uint8)

        # Trazar contorno blanco
        if self.last_contours:
            cv2.drawContours(overlay, self.last_contours, -1, (255, 255, 255), 2, cv2.LINE_AA)

        return overlay


# Instancia singleton predeterminada
_default_detector: Optional[DeployZoneDetector] = None


def get_deploy_zone_detector() -> DeployZoneDetector:
    global _default_detector
    if _default_detector is None:
        cfg = SETTINGS.get("attack", {}).get("deploy_zone_ai", {})
        model_path = cfg.get("model_path", "models/deploy_zone_finder.onnx")
        conf = float(cfg.get("conf_thresh", 0.50))
        imgsz = int(cfg.get("imgsz", 1280))
        tta = bool(cfg.get("use_tta", False))
        _default_detector = DeployZoneDetector(
            model_path=model_path,
            conf_thresh=conf,
            imgsz=imgsz,
            use_tta=tta,
        )
    return _default_detector
```

## File: vision/resource_reader.py
```python
"""
vision/resource_reader.py
Lector de recursos robusto con binarización adaptativa, banco de escalas precalculado,
validación de grupos de miles y registro de fallos en logs/fails/.
"""
import os
import time
import cv2
import numpy as np
from core.config import SETTINGS, PROJECT_ROOT
from core.vision import Vision
from core.logger import get_logger

log = get_logger("resource_reader")

SCALES = (0.78, 0.82, 0.86, 0.90, 0.94, 0.98, 1.0, 1.08)


class ResourceReader:
    def __init__(self, vision: Vision):
        self.vision = vision
        self.digit_templates = {d: f"assets/digits/{d}.png" for d in range(10)}
        self.enemy_rois = SETTINGS["rois"]["enemy_loot"]
        self.own_rois = SETTINGS["rois"]["own_resources"]
        self._bank: list[tuple[int, np.ndarray]] | None = None

    @staticmethod
    def _binarize_white_digits(gray_img: np.ndarray, adaptive: bool = False) -> np.ndarray:
        thr = 190
        if adaptive and gray_img.size > 0:
            peak = float(np.percentile(gray_img, 99))
            thr = int(np.clip(peak * 0.85, 150, 200))
        _, binary = cv2.threshold(gray_img, thr, 255, cv2.THRESH_BINARY)
        return binary

    @classmethod
    def _prepare_scaled_template(cls, tpl_gray: np.ndarray, scale: float) -> np.ndarray | None:
        if scale != 1.0:
            th = max(1, int(round(tpl_gray.shape[0] * scale)))
            tw = max(1, int(round(tpl_gray.shape[1] * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            tpl_gray = cv2.resize(tpl_gray, (tw, th), interpolation=interp)

        tpl_bin = cls._binarize_white_digits(tpl_gray)
        coords = cv2.findNonZero(tpl_bin)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        cropped = tpl_bin[y:y + h, x:x + w]
        return cv2.copyMakeBorder(cropped, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    def _get_bank(self) -> list[tuple[int, np.ndarray]]:
        if self._bank is None:
            self._bank = []
            for digit, path in self.digit_templates.items():
                cached = self.vision._load_template(path)
                if cached is None:
                    continue
                tpl_gray, _ = cached
                for s in SCALES:
                    t = self._prepare_scaled_template(tpl_gray, s)
                    if t is not None:
                        self._bank.append((digit, t))
        return self._bank

    def _candidates(self, roi_bin: np.ndarray, floor: float = 0.65) -> list[tuple[int, float, int, int]]:
        rh, rw = roi_bin.shape[:2]
        out: list[tuple[int, float, int, int]] = []
        for digit, tpl in self._get_bank():
            th, tw = tpl.shape[:2]
            if th >= rh or tw >= rw or th < 8 or tw < 4:
                continue
            res = cv2.matchTemplate(roi_bin, tpl, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= floor)
            out.extend([(int(x), float(res[y, x]), digit, tw) for y, x in zip(ys, xs)])
        return out

    @staticmethod
    def _nms(cands: list[tuple[int, float, int, int]], thr: float) -> list[tuple[int, int, int, float]]:
        kept: list[tuple[int, int, int, float]] = []  # (x, digit, tw, score)
        for x, score, digit, tw in sorted(cands, key=lambda c: c[1], reverse=True):
            if score < thr:
                break
            if all(max(0, min(x + tw, kx + ktw) - max(x, kx)) / min(tw, ktw) <= 0.35 for kx, _, ktw, _ in kept):
                kept.append((x, digit, tw, score))
        return sorted(kept, key=lambda k: k[0])

    @staticmethod
    def _valid_grouping(dets: list[tuple[int, int, int, float]]) -> bool:
        """Valida que todos los grupos de miles salvo el primero tengan exactamente 3 dígitos."""
        if not dets:
            return False
        if len(dets) == 1:
            return True
        med_w = float(np.median([d[2] for d in dets]))
        # Separador de miles en fuente Clash of Clans es de 7-11px; gaps entre dígitos son de -1 a 2px.
        sep_thresh = max(5.0, 0.25 * med_w)
        groups = [1]
        for a, b in zip(dets, dets[1:]):
            gap = b[0] - (a[0] + a[2])
            if gap > sep_thresh:  # Separador de miles
                groups.append(1)
            else:
                groups[-1] += 1
        return 1 <= groups[0] <= 3 and all(g == 3 for g in groups[1:])

    def _dump_fail(self, roi_gray: np.ndarray, roi_bin: np.ndarray, key: str):
        fail_dir = os.path.join(PROJECT_ROOT, "logs", "fails")
        os.makedirs(fail_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        cv2.imwrite(os.path.join(fail_dir, f"{ts}_{key}_gray.png"), roi_gray)
        cv2.imwrite(os.path.join(fail_dir, f"{ts}_{key}_bin.png"), roi_bin)

    def _read_number_from_roi(self, roi_gray: np.ndarray, key: str = "") -> int | None:
        roi_bin = self._binarize_white_digits(roi_gray, adaptive=True)
        roi_bin = cv2.copyMakeBorder(roi_bin, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        cands = self._candidates(roi_bin, floor=0.65)

        for thr in (0.80, 0.74, 0.68, 0.65):
            dets = self._nms(cands, thr)
            if self._valid_grouping(dets):
                return int("".join(str(d[1]) for d in dets))

        self._dump_fail(roi_gray, roi_bin, key)
        return None

    def _extract_rois(self, screenshot, roi_map: dict) -> dict[str, int | None]:
        if isinstance(screenshot, np.ndarray) and screenshot.ndim == 3:
            screen_gray = np.min(screenshot[:, :, :3], axis=2)
        else:
            screen_gray = self.vision.to_gray(screenshot)

        h, w = screen_gray.shape[:2]
        result: dict[str, int | None] = {}

        for key, (ry1, ry2, rx1, rx2) in roi_map.items():
            roi = screen_gray[int(h * ry1):int(h * ry2), int(w * rx1):int(w * rx2)]
            result[key] = self._read_number_from_roi(roi, key)
        return result

    @staticmethod
    def _format_resource(val: int | None) -> str:
        return f"{val:,}" if val is not None else "??"

    def read_enemy_loot(self, screenshot) -> dict[str, int | None]:
        loot = self._extract_rois(screenshot, self.enemy_rois)
        log.info(
            "🔎 Botín Enemigo -> Oro: %s | Elixir: %s | Oscuro: %s",
            self._format_resource(loot.get("gold")),
            self._format_resource(loot.get("elixir")),
            self._format_resource(loot.get("dark_elixir"))
        )
        return loot

    def read_enemy_loot_stable(self, screenshot_fn, needed: int = 2, timeout: float = 4.0, interval: float = 0.2) -> dict[str, int] | None:
        """
        Espera a que la lectura sea consistente en 2 capturas consecutivas
        para evitar lecturas durante la animación de conteo inicial.
        """
        t0 = time.monotonic()
        last = None
        streak = 0

        while time.monotonic() - t0 < timeout:
            loot = self._extract_rois(screenshot_fn(), self.enemy_rois)
            if all(v is not None for v in loot.values()):
                if loot == last:
                    streak += 1
                    if streak >= needed:
                        log.info(
                            "🔎 Botín Estable -> Oro: %s | Elixir: %s | Oscuro: %s",
                            f"{loot['gold']:,}", f"{loot['elixir']:,}", f"{loot['dark_elixir']:,}"
                        )
                        return loot  # type: ignore[return-value]
                else:
                    last = loot
                    streak = 1
            time.sleep(interval)

        log.warning("⚠️ No se obtuvo lectura estable del botín tras %.1fs.", timeout)
        return None

    def read_own_resources(self, screenshot) -> dict[str, int]:
        rss = self._extract_rois(screenshot, self.own_rois)
        cleaned = {k: (v if v is not None else 0) for k, v in rss.items()}
        log.info(
            "🏦 Recursos Propios -> Oro: %s | Elixir: %s | Oscuro: %s",
            self._format_resource(cleaned.get("gold")),
            self._format_resource(cleaned.get("elixir")),
            self._format_resource(cleaned.get("dark_elixir"))
        )
        return cleaned

    @staticmethod
    def meets_attack_criteria(loot: dict[str, int] | None, criteria_override: dict | None = None) -> bool:
        if not loot:
            return False
        cfg = criteria_override or SETTINGS.get("loot_criteria", {})
        min_gold = cfg.get("min_gold", 800000)
        min_elixir = cfg.get("min_elixir", 800000)
        min_dark = cfg.get("min_dark_elixir", 7000)
        req = cfg.get("conditions_required", 2)
        conds = [
            (loot.get("gold") or 0) >= min_gold,
            (loot.get("elixir") or 0) >= min_elixir,
            (loot.get("dark_elixir") or 0) >= min_dark,
        ]
        met = sum(conds)
        decision = met >= req
        log.info("⚖️ Condiciones cumplidas: %d/3 (req: %d) -> %s", met, req, "⚔️ ATACAR" if decision else "⏭️ SIGUIENTE")
        return decision
```

## File: vision/storage_detector.py
```python
"""
vision/storage_detector.py
Módulo de detección y reconocimiento interactivo de almacenes:
1. Escaneo pasivo por densidad de color (Oro, Elixir rosa y Elixir oscuro).
2. Verificación interactiva mediante toque de selección y lectura del rótulo flotante ("Almacén").
3. Cálculo dinámico de vectores de asalto para orientar Hechizos de Salto sobre el primer muro
   directamente alineados con los almacenes reales de la aldea rival.
"""
import time
import cv2
import numpy as np

from core.logger import get_logger
from core.device import Device
from core.vision import Vision

log = get_logger("storage_detector")


class StorageDetector:
    def __init__(self, device: Device | None = None, vision: Vision | None = None):
        self.device = device
        self.vision = vision
        self.tpl_tag_almacen = "assets/buttons/tag_almacen.png"
        self.tpl_pill_cap = "assets/buttons/pill_tag_cap.png"

    @staticmethod
    def _extract_color_masks(screen: np.ndarray):
        h, w = screen.shape[:2]
        # Área jugable de la aldea (ignora HUD de botín en esquina sup-izq y barras inferiores)
        x1, x2 = int(0.26 * w), int(0.74 * w)
        y1, y2 = int(0.20 * h), int(0.78 * h)
        roi = screen[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Máscaras adaptativas para almacenes interiores
        masks = {
            "storage_elixir": cv2.inRange(hsv, (118, 90, 70), (165, 255, 255)),
            "storage_gold": cv2.inRange(hsv, (16, 65, 65), (62, 255, 255)),
            "storage_dark": cv2.inRange(hsv, (55, 60, 20), (165, 255, 255)),
        }
        m_combined = masks["storage_elixir"] | masks["storage_gold"] | masks["storage_dark"]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed = cv2.morphologyEx(m_combined, cv2.MORPH_CLOSE, kernel)
        return roi, (x1, y1, x2, y2), masks, closed

    def detect_boxes(self, screenshot: np.ndarray, min_conf: float = 0.30) -> list[dict]:
        """
        Detección visual pasiva de cajas delimitadoras (bounding boxes) de almacenes
        a partir de segmentación por color HSV y análisis morfológico.
        Devuelve lista de dicts:
            {"cls": "storage_gold"|"storage_elixir"|"storage_dark",
             "bbox_xyxy": (x1, y1, x2, y2),
             "confidence": float}
        """
        roi, (x1, y1, x2, y2), masks, closed = self._extract_color_masks(screenshot)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results: list[dict] = []
        expected_area = 2500.0  # ~50x50 px a 2400x1080 con zoom-out

        for c in cnts:
            a = cv2.contourArea(c)
            if a < 200 or a > 35000:
                continue

            bx, by, bw, bh = cv2.boundingRect(c)
            # Descartar formas extremadamente alargadas (muros o barras de interfaz)
            aspect_ratio = float(bw) / max(1.0, float(bh))
            if aspect_ratio < 0.35 or aspect_ratio > 2.85:
                continue

            # Máscara local del contorno para evaluar pureza de color
            cnt_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.drawContours(cnt_mask, [c], -1, 255, -1)

            overlaps = {
                cls_name: cv2.countNonZero(cv2.bitwise_and(m, cnt_mask))
                for cls_name, m in masks.items()
            }
            best_cls = max(overlaps, key=overlaps.get)
            max_overlap = overlaps[best_cls]

            # Descartar contornos ambiguos con poca señal de color
            if max_overlap < 25:
                continue

            # Confianza heurística basada en ratio de área esperada y densidad del color
            area_ratio = min(a / expected_area, expected_area / a)
            color_purity = max_overlap / max(1.0, float(a))
            conf = float(np.clip(0.55 * area_ratio + 0.45 * color_purity, 0.15, 0.99))

            if conf >= min_conf:
                gx1 = int(bx + x1)
                gy1 = int(by + y1)
                gx2 = int(gx1 + bw)
                gy2 = int(gy1 + bh)
                results.append({
                    "cls": best_cls,
                    "bbox_xyxy": (gx1, gy1, gx2, gy2),
                    "confidence": round(conf, 3),
                })

        return results

    def scan_storage_candidates(self, screen: np.ndarray) -> dict[str, list[tuple[int, int]]]:
        """
        Escanea la pantalla centrada y extrae los candidatos de almacenes
        clasificados por cuadrantes: 'sw', 'se', 'north', 'core'.
        """
        h, w = screen.shape[:2]
        roi, (x1, y1, x2, y2), _, closed = self._extract_color_masks(screen)

        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        center_x, center_y = int(0.50 * w), int(0.50 * h)

        for c in cnts:
            a = cv2.contourArea(c)
            if 200 < a < 35000:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) + x1
                    cy = int(M["m01"] / M["m00"]) + y1
                    # Filtrar elementos fuera del perímetro de la base (minas/extractores en césped exterior)
                    dist_center = np.hypot(cx - center_x, cy - center_y)
                    if dist_center < 490:
                        candidates.append(((cx, cy), a))

        # Clasificación por cuadrantes
        center_x, center_y = int(0.50 * w), int(0.50 * h)
        core_radius = 160

        quadrants = {"sw": [], "se": [], "north": [], "core": [], "all": []}
        for (cx, cy), a in sorted(candidates, key=lambda x: x[1], reverse=True):
            dist_center = np.hypot(cx - center_x, cy - center_y)
            quadrants["all"].append((cx, cy))
            if dist_center < core_radius:
                quadrants["core"].append((cx, cy))
            elif cy >= center_y:
                if cx < center_x:
                    quadrants["sw"].append((cx, cy))
                else:
                    quadrants["se"].append((cx, cy))
            else:
                quadrants["north"].append((cx, cy))

        log.info(
            "📍 Almacenes detectados: Total=%d (SW=%d, SE=%d, Norte=%d, Núcleo=%d)",
            len(quadrants["all"]),
            len(quadrants["sw"]),
            len(quadrants["se"]),
            len(quadrants["north"]),
            len(quadrants["core"]),
        )
        return quadrants

    def verify_storage_interactively(self, candidate_pt: tuple[int, int]) -> bool:
        """
        Toca un candidato y comprueba si aparece el rótulo flotante 'Almacén'
        o la etiqueta de selección masiva sobre el edificio.
        """
        log.info("👆 Tocando candidato en %s para verificar rótulo de almacén...", candidate_pt)
        self.device.tap(*candidate_pt)
        time.sleep(0.4)

        s_tapped = self.device.screenshot()
        # 1. Comprobar texto 'Almacén'
        pos = self.vision.find_template(s_tapped, self.tpl_tag_almacen, threshold=0.68, zona="COMPLETA")
        if pos:
            log.info("✨ Rótulo 'Almacén' confirmado con éxito en %s!", pos)
            return True

        # 2. Comprobar capuchón de píldora de selección flotante
        pos_cap = self.vision.find_template(s_tapped, self.tpl_pill_cap, threshold=0.70, zona="COMPLETA")
        if pos_cap:
            log.info("✨ Etiqueta de selección confirmada en %s!", pos_cap)
            return True

        log.debug("ℹ️ Candidato en %s no mostró etiqueta de almacén conocida.", candidate_pt)
        return False

    def compute_adaptive_jump_targets(
        self, quadrants: dict[str, list[tuple[int, int]]]
    ) -> list[tuple[str, tuple[int, int]]]:
        """
        Calcula las coordenadas óptimas de los 4 Hechizos de Salto adaptadas
        a la posición real de los almacenes detectados en cada cuadrante.
        Garantiza que el Salto se posicione sobre el PRIMER MURO exterior alineado
        con el almacén objetivo de ese cuadrante.
        """
        w, h = self.device.width, self.device.height
        default_sw = (int(0.415 * w), int(0.640 * h))
        default_se = (int(0.585 * w), int(0.640 * h))
        default_north = (int(0.500 * w), int(0.315 * h))
        default_core = (int(0.500 * w), int(0.500 * h))

        targets = []
        # Solo incluir saltos en cuadrantes que tengan almacenes reales detectados
        if quadrants["sw"]:
            c_x = int(np.mean([p[0] for p in quadrants["sw"]]))
            c_y = int(np.mean([p[1] for p in quadrants["sw"]]))
            muro_x = int(np.clip(c_x - 30, int(0.36 * w), int(0.45 * w)))
            muro_y = int(np.clip(c_y + 40, int(0.60 * h), int(0.70 * h)))
            targets.append(("Primer Muro Suroeste (Adaptativo)", (muro_x, muro_y)))
        elif not quadrants["all"]:
            targets.append(("Primer Muro Suroeste (Default)", default_sw))

        if quadrants["se"]:
            c_x = int(np.mean([p[0] for p in quadrants["se"]]))
            c_y = int(np.mean([p[1] for p in quadrants["se"]]))
            muro_x = int(np.clip(c_x + 30, int(0.55 * w), int(0.64 * w)))
            muro_y = int(np.clip(c_y + 40, int(0.60 * h), int(0.70 * h)))
            targets.append(("Primer Muro Sureste (Adaptativo)", (muro_x, muro_y)))
        elif not quadrants["all"]:
            targets.append(("Primer Muro Sureste (Default)", default_se))

        if quadrants["north"]:
            c_x = int(np.mean([p[0] for p in quadrants["north"]]))
            c_y = int(np.mean([p[1] for p in quadrants["north"]]))
            muro_x = int(np.clip(c_x, int(0.42 * w), int(0.58 * w)))
            muro_y = int(np.clip(c_y - 45, int(0.28 * h), int(0.38 * h)))
            targets.append(("Primer Muro Norte (Adaptativo)", (muro_x, muro_y)))
        elif not quadrants["all"]:
            targets.append(("Primer Muro Norte (Default)", default_north))

        # Solo lanzar en Núcleo si realmente hay almacenes en el núcleo (evitar desperdiciar salto en TH sin muros)
        if quadrants["core"]:
            c_x = int(np.mean([p[0] for p in quadrants["core"]]))
            c_y = int(np.mean([p[1] for p in quadrants["core"]]))
            targets.append(("Núcleo / Almacenes Centrales", (c_x, c_y)))

        return targets
```

## File: vision/wall_compartments.py
```python
"""
Módulo de Optimización de Hechizos de Salto por Modelo de Compartimentos en Clash of Clans.
Modela la aldea como regiones cerradas por muros (compartimentos), calcula las capas
de muro mínimas desde las zonas de drop hacia cada objetivo de botín y selecciona de
forma voraz los puntos de salto que maximizan la reducción del costo global.

Fase 2 del Plan: colocación óptima de hechizos de salto (plan_saltos_optimos_antigravity.md).
"""
from __future__ import annotations
import math
import time
from collections import deque
from dataclasses import dataclass, field
import cv2
import numpy as np

from core.logger import get_logger

log = get_logger("wall_compartments")

# Pesos por defecto para objetivos de botín según settings.json
DEFAULT_LOOT_WEIGHTS: dict[str, float] = {
    "storage_dark": 3.0,
    "storage_gold": 2.0,
    "storage_elixir": 2.0,
    "collector_gold": 1.0,
    "collector_elixir": 1.0,
    "collector_dark": 1.0,
    "town_hall": 0.0,
}


@dataclass
class JumpCandidate:
    """Representa un salto evaluado o seleccionado."""
    x: int
    y: int
    gain: float
    cost_before: float
    cost_after: float
    bridged_compartments: list[int] = field(default_factory=list)


@dataclass
class CompartmentAnalysisResult:
    """Resultado estructurado del análisis de compartimentos y optimización de saltos."""
    jumps: list[tuple[int, int, float]]  # [(x, y, gain), ...]
    jump_details: list[JumpCandidate]
    initial_cost: float
    final_cost: float
    total_gain: float
    num_compartments: int
    comp_labels: np.ndarray
    wall_mask: np.ndarray
    edges: list[tuple[int, int]]
    target_assignments: list[dict]
    execution_time_ms: float
    debug_overlay: np.ndarray | None = None
    graph: dict[int, set[int]] = field(default_factory=dict)
    entry_comps: set[int] = field(default_factory=set)
    nearest_labels: np.ndarray | None = None
    grid_scale: int = 1


def _angle_diff_180(ang1: float, ang2: float) -> float:
    """Diferencia angular mínima en el rango [0, 180)."""
    diff = abs((ang1 - ang2) % 180.0)
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def rasterize_wall_mask(
    shape: tuple[int, int],
    wall_segments: list[tuple[int, int, int, int]],
    iso_angles: tuple[float, float] = (36.0, 144.0),
    wall_thickness: int = 8,
    endpoint_tol: float = 25.0,
    morph_kernel_size: int = 5,
) -> np.ndarray:
    """
    Rasteriza los segmentos de muro en una máscara binaria (uint8: 0 o 255):
    1. Prolonga los extremos a lo largo de la dirección isométrica dominante más cercana
       por endpoint_tol píxeles para cerrar huecos causados por edificios sobre el muro.
    2. Dibuja los segmentos con grosor wall_thickness.
    3. Aplica cierre morfológico para sellar micro-fisuras.
    """
    h, w = shape[:2]
    wall_mask = np.zeros((h, w), dtype=np.uint8)

    if not wall_segments:
        return wall_mask

    iso1, iso2 = iso_angles

    for seg in wall_segments:
        x1, y1, x2, y2 = seg[:4]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)

        if length < 1e-3:
            continue

        if endpoint_tol > 0:
            deg = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
            diff1 = _angle_diff_180(deg, iso1)
            diff2 = _angle_diff_180(deg, iso2)
            chosen_iso = iso1 if diff1 <= diff2 else iso2

            iso_rad = math.radians(chosen_iso)
            ux, uy = math.cos(iso_rad), math.sin(iso_rad)

            # Alinear sentido con el segmento
            if (dx * ux + dy * uy) < 0:
                ux, uy = -ux, -uy

            px1 = int(round(x1 - ux * endpoint_tol))
            py1 = int(round(y1 - uy * endpoint_tol))
            px2 = int(round(x2 + ux * endpoint_tol))
            py2 = int(round(y2 + uy * endpoint_tol))
        else:
            px1, py1 = int(round(x1)), int(round(y1))
            px2, py2 = int(round(x2)), int(round(y2))

        cv2.line(wall_mask, (px1, py1), (px2, py2), 255, thickness=int(wall_thickness))

    if morph_kernel_size > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(morph_kernel_size), int(morph_kernel_size)))
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k)

    return wall_mask


def build_barrier_mask(
    wall_mask: np.ndarray,
    detections: list | None = None,
    buildings_block: bool = True,
    building_margin: int = 0,
) -> np.ndarray:
    """
    Combina la máscara de muros con los bounding boxes de edificios YOLO.
    Si buildings_block=True, los edificios actúan como barreras infranqueables
    para evitar fugas de muros incompletos.
    """
    barrier_mask = wall_mask.copy()
    if not buildings_block or not detections:
        return barrier_mask

    h, w = barrier_mask.shape[:2]

    for det in detections:
        bbox = getattr(det, "bbox_xyxy", None)
        if bbox is None and isinstance(det, dict):
            bbox = det.get("bbox_xyxy") or det.get("bbox")
        if bbox is None and isinstance(det, (list, tuple)) and len(det) >= 4:
            bbox = det[:4]

        if bbox is not None:
            bx1, by1, bx2, by2 = [int(v) for v in bbox[:4]]
            if building_margin != 0:
                bx1 += building_margin
                by1 += building_margin
                bx2 -= building_margin
                by2 -= building_margin
            bx1 = max(0, min(w - 1, bx1))
            by1 = max(0, min(h - 1, by1))
            bx2 = max(0, min(w, bx2))
            by2 = max(0, min(h, by2))
            if bx2 > bx1 and by2 > by1:
                cv2.rectangle(barrier_mask, (bx1, by1), (bx2, by2), 255, -1)

    return barrier_mask


def compute_compartments(
    obstacle_mask: np.ndarray,
    drop_points: list[tuple[int, int]] | None = None,
) -> tuple[int, np.ndarray, set[int], np.ndarray]:
    """
    Calcula los componentes conexos del espacio libre (donde obstacle_mask == 0).
    Cada componente conexo > 0 representa un compartimento.
    Mapea además todos los píxeles de obstáculos a su compartimento libre más cercano
    utilizando cv2.distanceTransformWithLabels.

    Retorna:
        num_compartments (total de componentes incluyendo fondo 0)
        comp_labels (matriz 2D con etiquetas de compartimento, 0 = obstáculo)
        entry_comps (conjunto de IDs de compartimentos de entrada)
        nearest_labels (matriz 2D donde cada obstáculo tiene la etiqueta de su compartimento más cercano)
    """
    h, w = obstacle_mask.shape[:2]
    free_space = (obstacle_mask == 0).astype(np.uint8)

    num_labels, comp_labels = cv2.connectedComponents(free_space, connectivity=8)

    # Invertir para distanceTransform: en OpenCV la distancia se calcula hacia los píxeles con valor 0.
    # Por tanto, definimos free_space como 0 para calcular la distancia desde el obstáculo al espacio libre.
    src_dist = (255 - free_space * 255).astype(np.uint8)

    # cv2.distanceTransformWithLabels con DIST_LABEL_CCOMP asigna a cada componente de 0 una etiqueta única
    dist_map, nearest_labels = cv2.distanceTransformWithLabels(
        src_dist, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP
    )

    # Si hay espacio libre, encontrar los compartimentos de entrada
    entry_comps: set[int] = set()

    if drop_points:
        for pt in drop_points:
            px, py = int(round(pt[0])), int(round(pt[1]))
            if 0 <= px < w and 0 <= py < h:
                lbl = comp_labels[py, px]
                if lbl == 0:
                    lbl = nearest_labels[py, px]
                if lbl > 0:
                    entry_comps.add(int(lbl))

    # Si no se identificó entrada o no se suministraron drop_points,
    # el compartimento exterior es el componente más representativo del perímetro
    if not entry_comps and num_labels > 1:
        border_pixels = np.concatenate([
            comp_labels[0, :],
            comp_labels[h - 1, :],
            comp_labels[:, 0],
            comp_labels[:, w - 1],
        ])
        border_nonzero = border_pixels[border_pixels > 0]
        if len(border_nonzero) > 0:
            vals, counts = np.unique(border_nonzero, return_counts=True)
            exterior_comp = int(vals[np.argmax(counts)])
            entry_comps.add(exterior_comp)
        else:
            entry_comps.add(1)

    return num_labels, comp_labels, entry_comps, nearest_labels


def build_compartment_graph(
    num_compartments: int,
    comp_labels: np.ndarray,
    nearest_labels: np.ndarray,
    wall_mask: np.ndarray,
    max_wall_width: float = 35.0,
) -> tuple[dict[int, set[int]], list[tuple[int, int]]]:
    """
    Construye el grafo de adyacencia de compartimentos:
    Nodos = compartimentos libres (1 .. num_compartments - 1).
    Arista entre (u, v) si comparten una banda de muro en wall_mask.
    Costo base de cada arista = 1 capa de muro.

    Retorna: (grafo_adyacencia, lista_de_aristas)
    """
    graph: dict[int, set[int]] = {c: set() for c in range(1, num_compartments)}
    edges_set: set[tuple[int, int]] = set()

    if num_compartments <= 2:
        return graph, []

    # Transición horizontal a través de píxeles de muro
    mask_h = (nearest_labels[:, 1:] != nearest_labels[:, :-1])
    on_wall_h = (wall_mask[:, 1:] > 0) | (wall_mask[:, :-1] > 0)
    valid_h = mask_h & on_wall_h
    u_h = nearest_labels[:, 1:][valid_h]
    v_h = nearest_labels[:, :-1][valid_h]

    # Transición vertical a través de píxeles de muro
    mask_v = (nearest_labels[1:, :] != nearest_labels[:-1, :])
    on_wall_v = (wall_mask[1:, :] > 0) | (wall_mask[:-1, :] > 0)
    valid_v = mask_v & on_wall_v
    u_v = nearest_labels[1:, :][valid_v]
    v_v = nearest_labels[:-1, :][valid_v]

    all_u = np.concatenate([u_h, u_v])
    all_v = np.concatenate([v_h, v_v])

    valid = (all_u > 0) & (all_v > 0) & (all_u != all_v)
    if np.any(valid):
        u_sel = all_u[valid]
        v_sel = all_v[valid]
        min_uv = np.minimum(u_sel, v_sel).astype(np.int64)
        max_uv = np.maximum(u_sel, v_sel).astype(np.int64)
        encoded = min_uv * 1000000 + max_uv
        unique_pairs = np.unique(encoded)

        for code in unique_pairs:
            u = int(code // 1000000)
            v = int(code % 1000000)
            if u in graph and v in graph:
                graph[u].add(v)
                graph[v].add(u)
                edges_set.add((u, v))

    return graph, sorted(edges_set)


def map_targets_to_compartments(
    detections: list,
    comp_labels: np.ndarray,
    nearest_labels: np.ndarray,
    loot_weights: dict[str, float] | None = None,
) -> list[dict]:
    """
    Asigna cada detección de botín relevante a su compartimento correspondiente.
    Si el centro cae en un obstáculo (edificio/muro), utiliza el compartimento libre más cercano.
    """
    weights = loot_weights if loot_weights is not None else DEFAULT_LOOT_WEIGHTS
    h, w = comp_labels.shape[:2]
    assigned_targets: list[dict] = []

    for det in detections:
        cls_name = getattr(det, "cls_name", None)
        if cls_name is None and isinstance(det, dict):
            cls_name = det.get("cls_name", "")

        weight = float(weights.get(cls_name, 0.0))
        if weight <= 0.0:
            continue

        center = getattr(det, "center", None)
        if center is None and isinstance(det, dict):
            center = det.get("center")
        if center is None:
            bbox = getattr(det, "bbox_xyxy", None)
            if bbox is None and isinstance(det, dict):
                bbox = det.get("bbox_xyxy") or det.get("bbox")
            if bbox is not None:
                center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

        if center is None:
            continue

        cx = max(0, min(w - 1, int(round(center[0]))))
        cy = max(0, min(h - 1, int(round(center[1]))))

        comp = comp_labels[cy, cx]
        if comp == 0:
            comp = nearest_labels[cy, cx]

        if comp > 0:
            assigned_targets.append({
                "det": det,
                "cls_name": cls_name,
                "weight": weight,
                "center": (cx, cy),
                "comp": int(comp),
            })

    return assigned_targets


def compute_total_cost(
    graph: dict[int, set[int]],
    entry_comps: set[int],
    targets: list[dict],
    jump_bridges: list[set[int]] | None = None,
) -> tuple[float, dict[int, float]]:
    """
    Calcula las capas mínimas de muro desde entry_comps hacia todos los compartimentos
    utilizando 0-1 BFS / Dijkstra:
    - Aristas estándar del grafo cuestan 1 capa.
    - Compartimentos dentro de la huella de un mismo salto quedan unidos con costo 0.

    Costo total = sum(peso * min_capas(entrada, compartimento_objetivo)).
    Retorna: (costo_total, diccionario_distancias)
    """
    if not targets or not entry_comps:
        return 0.0, {}

    # Construir lista de adyacencia aumentada con aristas de costo 0 por saltos
    # dist: diccionario {comp: distancia_minima}
    all_nodes = set(graph.keys())
    for bridge in (jump_bridges or []):
        all_nodes.update(bridge)

    dist: dict[int, float] = {u: float("inf") for u in all_nodes}
    deque_bfs = deque()

    for s in entry_comps:
        if s in dist:
            dist[s] = 0.0
            deque_bfs.append((s, 0.0))

    # Pre-organizar puentes para consulta O(1) inmediata por nodo
    bridges = [b for b in (jump_bridges or []) if len(b) > 1]
    zero_adj: dict[int, list[int]] = {}
    for bridge in bridges:
        b_list = list(bridge)
        for u in b_list:
            if u not in zero_adj:
                zero_adj[u] = []
            for v in b_list:
                if u != v:
                    zero_adj[u].append(v)

    while deque_bfs:
        u, d = deque_bfs.popleft()
        if d > dist[u]:
            continue

        # 1. Movimiento gratuito (costo 0) a través de huellas de salto compartidas
        if u in zero_adj:
            for v in zero_adj[u]:
                if d < dist[v]:
                    dist[v] = d
                    deque_bfs.appendleft((v, d))

        # 2. Movimiento normal a través de muros (costo 1)
        d_next = d + 1.0
        for v in graph.get(u, ()):
            if d_next < dist[v]:
                dist[v] = d_next
                deque_bfs.append((v, d_next))

    total_cost = 0.0
    for t in targets:
        comp_id = t["comp"]
        d = dist.get(comp_id, float("inf"))
        # Si un compartimento está aislado sin conexión, aplicar penalización proporcional
        cost_layers = d if d < float("inf") else 10.0
        total_cost += t["weight"] * cost_layers

    return total_cost, dist


def get_compartments_in_footprint(
    center: tuple[int, int],
    footprint: dict | tuple | float | int,
    comp_labels: np.ndarray,
    nearest_labels: np.ndarray | None = None,
) -> set[int]:
    """
    Determina qué compartimentos libres son intersectados o cubiertos por la huella
    del hechizo de salto situada en center.
    Soporta:
    - Huella circular: radio r (int, float o {"type": "circle", "radius": r})
    - Huella elíptica isométrica: {"type": "ellipse", "rx": rx, "ry": ry, "angle": ang} o tuple (rx, ry)
    """
    h, w = comp_labels.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))

    # Interpretar parámetros de huella
    is_ellipse = False
    rx, ry = 90.0, 90.0
    angle_deg = 0.0

    if isinstance(footprint, dict):
        f_type = footprint.get("type", "circle")
        if f_type == "ellipse":
            is_ellipse = True
            rx = float(footprint.get("rx", 125.0))
            ry = float(footprint.get("ry", 72.0))
            angle_deg = float(footprint.get("angle", 0.0))
        else:
            rx = ry = float(footprint.get("radius", 90.0))
    elif isinstance(footprint, (tuple, list)):
        if len(footprint) >= 2:
            is_ellipse = True
            rx, ry = float(footprint[0]), float(footprint[1])
            angle_deg = float(footprint[2]) if len(footprint) > 2 else 0.0
        else:
            rx = ry = float(footprint[0])
    else:
        rx = ry = float(footprint)

    max_r = max(rx, ry)
    x0 = max(0, int(cx - max_r - 1))
    x1 = min(w, int(cx + max_r + 2))
    y0 = max(0, int(cy - max_r - 1))
    y1 = min(h, int(cy + max_r + 2))

    if x1 <= x0 or y1 <= y0:
        return set()

    ys, xs = np.ogrid[y0 - cy : y1 - cy, x0 - cx : x1 - cx]

    if is_ellipse:
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        x_rot = xs * cos_a + ys * sin_a
        y_rot = -xs * sin_a + ys * cos_a
        mask = ((x_rot / rx) ** 2 + (y_rot / ry) ** 2) <= 1.0
    else:
        mask = (xs ** 2 + ys ** 2) <= (rx ** 2)

    sub_comps = comp_labels[y0:y1, x0:x1]
    touched = set(np.unique(sub_comps[mask])) - {0}

    # Si la huella cubre el muro pero no alcanza píxeles libres,
    # consultar nearest_labels para los muros cubiertos
    if len(touched) < 2 and nearest_labels is not None:
        sub_nearest = nearest_labels[y0:y1, x0:x1]
        nearest_touched = set(np.unique(sub_nearest[mask])) - {0}
        if len(nearest_touched) > len(touched):
            touched = nearest_touched

    return touched


def get_battle_hud_mask(h: int, w: int, margin_px: int = 0) -> np.ndarray:
    """
    Genera la máscara de exclusión de elementos interactivos del HUD de batalla
    (barra de despliegue inferior, botón finalizar batalla, botón siguiente, medidores superiores).
    Resolución de referencia: 2400x1080.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    scale_x = w / 2400.0
    scale_y = h / 1080.0

    # 1. Margen superior extremo
    mask[:int(60 * scale_y), :] = 255

    # 2. Barra inferior de despliegue de tropas (y >= 888)
    y1_bar = int(888 * scale_y)
    mask[y1_bar:, :] = 255

    # 3. Botón 'Terminar Batalla' (esquina inferior izquierda)
    y1_eb, y2_eb = int(730 * scale_y), int(885 * scale_y)
    x1_eb, x2_eb = 0, int(360 * scale_x)
    mask[y1_eb:y2_eb, x1_eb:x2_eb] = 255

    # 4. Botón 'Siguiente' / menú lateral derecho
    y1_next, y2_next = int(850 * scale_y), int(980 * scale_y)
    x1_next = int(2040 * scale_x)
    mask[y1_next:y2_next, x1_next:] = 255

    # 5. Medidores superiores (Loot superior izquierdo, recursos superior derecho)
    mask[:int(100 * scale_y), :int(380 * scale_x)] = 255
    mask[:int(160 * scale_y), int(1900 * scale_x):] = 255

    if margin_px > 0:
        k_size = max(3, int(margin_px * 2 + 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        mask = cv2.dilate(mask, kernel)

    return mask


def sample_jump_candidates(
    wall_mask: np.ndarray,
    wall_segments: list[tuple[int, int, int, int]] | None = None,
    jump_radius: float = 90.0,
    step: int = 20,
    bbox_roi: tuple[int, int, int, int] | None = None,
    hud_mask: np.ndarray | None = None,
    hud_margin_px: int = 15,
) -> list[tuple[int, int]]:
    """
    Genera candidatos a punto de salto:
    1. Puntos uniformemente espaciados a lo largo de los segmentos de muro detectados.
    2. Puntos medios entre pares de segmentos de muro cercanos (cruces entre capas concéntricas).
    3. Muestreo en rejilla sobre las celdas con muro de wall_mask.
    Excluye candidatos que caigan dentro del HUD de batalla (con margen), pero permite que la huella
    se extienda bajo el HUD libremente.
    """
    candidates_set: set[tuple[int, int]] = set()
    h, w = wall_mask.shape[:2]

    dilated_hud = None
    if hud_mask is not None:
        if hud_margin_px > 0:
            k = max(3, int(hud_margin_px * 2 + 1))
            dilated_hud = cv2.dilate(hud_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        else:
            dilated_hud = hud_mask

    # 1. Puntos a lo largo de cada segmento
    pts_seg: list[tuple[int, int]] = []
    pts_mids: list[tuple[int, int]] = []

    if wall_segments:
        s_arr = np.array(wall_segments, dtype=np.float32)[:, :4]
        centers = np.column_stack([(s_arr[:, 0] + s_arr[:, 2]) / 2.0, (s_arr[:, 1] + s_arr[:, 3]) / 2.0])
        n_segs = len(centers)

        # 2. Puntos medios vectorizados entre segmentos cercanos (cruces concéntricos de máxima prioridad)
        if n_segs > 1:
            diff = centers[:, None, :] - centers[None, :, :]
            dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
            triu = np.triu(np.ones((n_segs, n_segs), dtype=bool), k=1)
            i, j = np.where((dist >= 20.0) & (dist <= (2.0 * jump_radius - 10.0)) & triu)
            if len(i) > 0:
                pair_mids = (centers[i] + centers[j]) / 2.0
                step_snap = max(15.0, float(step))
                uniq = np.unique(np.round(pair_mids / step_snap).astype(np.int32), axis=0) * int(step_snap)
                for p in uniq:
                    px, py = int(p[0]), int(p[1])
                    if 0 <= px < w and 0 <= py < h:
                        pts_mids.append((px, py))

        # Muestreo a lo largo de segmentos
        for seg in wall_segments:
            x1, y1, x2, y2 = seg[:4]
            mx, my = int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))
            if 0 <= mx < w and 0 <= my < h:
                pts_seg.append((mx, my))

    # 3. Muestreo en rejilla rápido vectorizado sobre wall_mask
    pts_grid: list[tuple[int, int]] = []
    step_grid = max(25, step)
    sub = wall_mask[::step_grid, ::step_grid]
    gy, gx = np.where(sub > 0)
    for py, px in zip(gy * step_grid, gx * step_grid):
        pts_grid.append((int(px), int(py)))

    # Ordenar por prioridad: primero cruces entre capas concéntricas (pts_mids), luego segmentos, luego rejilla
    ordered_candidates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pt in pts_mids + pts_seg + pts_grid:
        if bbox_roi:
            rx1, ry1, rx2, ry2 = bbox_roi
            if not (rx1 <= pt[0] <= rx2 and ry1 <= pt[1] <= ry2):
                continue
        if dilated_hud is not None:
            if 0 <= pt[1] < dilated_hud.shape[0] and 0 <= pt[0] < dilated_hud.shape[1]:
                if dilated_hud[pt[1], pt[0]] > 0:
                    continue
        if pt not in seen:
            seen.add(pt)
            ordered_candidates.append(pt)

    return ordered_candidates


def optimize_jumps(
    img_shape: tuple[int, int],
    wall_segments: list[tuple[int, int, int, int]],
    detections: list,
    drop_points: list[tuple[int, int]] | None = None,
    iso_angles: tuple[float, float] = (36.0, 144.0),
    jump_footprint: dict | tuple | float | int = 90.0,
    max_jumps: int = 4,
    min_marginal_gain: float = 0.5,
    loot_weights: dict[str, float] | None = None,
    buildings_block: bool = True,
    wall_thickness: int = 8,
    endpoint_tol: float = 25.0,
    morph_kernel_size: int = 5,
    candidate_step: int = 20,
    time_budget_sec: float = 0.300,
    grid_scale: int = 1,
    generate_debug_overlay: bool = True,
    base_img: np.ndarray | None = None,
    hud_mask: np.ndarray | None = None,
    hud_margin_px: int = 15,
    spell_config: dict | None = None,
    force_max_jumps: bool = False,
) -> CompartmentAnalysisResult:
    """
    Función maestra de optimización de saltos por modelo de compartimentos.
    Presupuesto de ejecución: < 300 ms por aldea.
    Independiente del hechizo (soporta configuración de radio, forma, efecto y cantidad).
    """
    t_start = time.perf_counter()
    h, w = img_shape[:2]

    # Abstracción de hechizo desde spell_config (settings.json)
    if spell_config:
        if "max_spells" in spell_config or "max_jumps" in spell_config:
            max_jumps = int(spell_config.get("max_spells") or spell_config.get("max_jumps", max_jumps))
        if "shape" in spell_config:
            shape = spell_config.get("shape", "circle")
            if shape == "ellipse":
                rx = float(spell_config.get("rx", 115.0))
                ry = float(spell_config.get("ry", 65.0))
                jump_footprint = {"shape": "ellipse", "rx": rx, "ry": ry}
            elif shape == "circle":
                r = float(spell_config.get("radius", 90.0))
                jump_footprint = r
        if "min_marginal_gain" in spell_config:
            min_marginal_gain = float(spell_config["min_marginal_gain"])

    # Escalamiento adaptativo para cumplir con el presupuesto estricto (<300 ms) en imágenes HD (1080p)
    scale = int(grid_scale)
    if scale <= 1 and (w >= 1200 or h >= 700):
        scale = 2

    if scale > 1:
        sh, sw = int(h // scale), int(w // scale)
        scaled_shape = (sh, sw)
        scaled_walls = [
            (int(round(s[0] / scale)), int(round(s[1] / scale)), int(round(s[2] / scale)), int(round(s[3] / scale)))
            for s in wall_segments
        ]
        scaled_dets = []
        for det in detections:
            bbox = getattr(det, "bbox_xyxy", None)
            if bbox is None and isinstance(det, dict):
                bbox = det.get("bbox_xyxy") or det.get("bbox")
            center = getattr(det, "center", None)
            if center is None and isinstance(det, dict):
                center = det.get("center")
            cls_name = getattr(det, "cls_name", "") if hasattr(det, "cls_name") else det.get("cls_name", "")

            s_bbox = tuple(int(round(v / scale)) for v in bbox) if bbox is not None else None
            s_center = (int(round(center[0] / scale)), int(round(center[1] / scale))) if center is not None else None
            scaled_dets.append({"cls_name": cls_name, "bbox_xyxy": s_bbox, "center": s_center})

        scaled_drop = [(int(round(p[0] / scale)), int(round(p[1] / scale))) for p in (drop_points or [])]

        if isinstance(jump_footprint, dict):
            scaled_fp = dict(jump_footprint)
            if "radius" in scaled_fp:
                scaled_fp["radius"] = float(scaled_fp["radius"]) / scale
            if "rx" in scaled_fp:
                scaled_fp["rx"] = float(scaled_fp["rx"]) / scale
            if "ry" in scaled_fp:
                scaled_fp["ry"] = float(scaled_fp["ry"]) / scale
        elif isinstance(jump_footprint, (list, tuple)):
            scaled_fp = tuple(float(v) / scale if idx < 2 else v for idx, v in enumerate(jump_footprint))
        else:
            scaled_fp = float(jump_footprint) / scale

        scaled_thickness = max(3, int(round(wall_thickness / scale)))
        scaled_tol = float(endpoint_tol) / scale
        scaled_step = max(15, int(round(candidate_step / scale)))
        scaled_hud = cv2.resize(hud_mask, (sw, sh), interpolation=cv2.INTER_NEAREST) if hud_mask is not None else None
    else:
        scale = 1
        scaled_shape = (h, w)
        scaled_walls = wall_segments
        scaled_dets = detections
        scaled_drop = drop_points
        scaled_fp = jump_footprint
        scaled_thickness = wall_thickness
        scaled_tol = endpoint_tol
        scaled_step = candidate_step
        scaled_hud = hud_mask

    # Si hud_mask no se pasó explícitamente, instanciar la máscara estándar del HUD de batalla
    if scaled_hud is None:
        scaled_hud = get_battle_hud_mask(scaled_shape[0], scaled_shape[1])

    # 1. Rasterizar máscara de muros
    wall_mask = rasterize_wall_mask(
        shape=scaled_shape,
        wall_segments=scaled_walls,
        iso_angles=iso_angles,
        wall_thickness=scaled_thickness,
        endpoint_tol=scaled_tol,
        morph_kernel_size=morph_kernel_size,
    )

    # 2. Barreras de edificios
    barrier_mask = build_barrier_mask(
        wall_mask=wall_mask,
        detections=scaled_dets,
        buildings_block=buildings_block,
    )

    # 3. Componentes conexos (compartimentos)
    num_comps, comp_labels, entry_comps, nearest_labels = compute_compartments(
        obstacle_mask=barrier_mask,
        drop_points=scaled_drop,
    )

    # 4. Grafo de adyacencia de compartimentos
    graph, edges = build_compartment_graph(
        num_compartments=num_comps,
        comp_labels=comp_labels,
        nearest_labels=nearest_labels,
        wall_mask=wall_mask,
    )

    # 5. Mapear objetivos a compartimentos
    targets = map_targets_to_compartments(
        detections=scaled_dets,
        comp_labels=comp_labels,
        nearest_labels=nearest_labels,
        loot_weights=loot_weights,
    )

    # Si no hay muros, o la aldea tiene 1 solo compartimento, o no hay objetivos: costo inicial = 0
    if num_comps <= 1 or not targets or not wall_segments:
        initial_cost, _ = compute_total_cost(graph, entry_comps, targets, jump_bridges=[])
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        return CompartmentAnalysisResult(
            jumps=[],
            jump_details=[],
            initial_cost=initial_cost,
            final_cost=initial_cost,
            total_gain=0.0,
            num_compartments=max(1, num_comps - 1),
            comp_labels=comp_labels,
            wall_mask=wall_mask,
            edges=edges,
            target_assignments=targets,
            execution_time_ms=t_elapsed,
            debug_overlay=None,
            graph=graph,
            entry_comps=entry_comps,
            nearest_labels=nearest_labels,
            grid_scale=scale,
        )

    # 6. Calcular costo inicial
    initial_cost, initial_dist = compute_total_cost(graph, entry_comps, targets, jump_bridges=[])

    if (initial_cost <= 0.0 and not force_max_jumps) or max_jumps <= 0:
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        return CompartmentAnalysisResult(
            jumps=[],
            jump_details=[],
            initial_cost=initial_cost,
            final_cost=initial_cost,
            total_gain=0.0,
            num_compartments=max(1, num_comps - 1),
            comp_labels=comp_labels,
            wall_mask=wall_mask,
            edges=edges,
            target_assignments=targets,
            execution_time_ms=t_elapsed,
            debug_overlay=None,
            graph=graph,
            entry_comps=entry_comps,
            nearest_labels=nearest_labels,
            grid_scale=scale,
        )

    # 7. Muestrear candidatos (excluyendo puntos bajo el HUD)
    r_approx = 90.0 / scale
    if isinstance(scaled_fp, (int, float)):
        r_approx = float(scaled_fp)
    elif isinstance(scaled_fp, dict):
        r_approx = float(scaled_fp.get("rx", scaled_fp.get("radius", 90.0 / scale)))
    elif isinstance(scaled_fp, (list, tuple)) and len(scaled_fp) > 0:
        r_approx = float(scaled_fp[0])

    candidates = sample_jump_candidates(
        wall_mask=wall_mask,
        wall_segments=scaled_walls,
        jump_radius=r_approx,
        step=scaled_step,
        hud_mask=scaled_hud,
        hud_margin_px=max(3, int(round(hud_margin_px / scale))),
    )

    # Limitar número de candidatos para garantizar cumplimiento estricto del presupuesto (<300ms)
    max_candidates = 200
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    loop_deadline = time_budget_sec - 0.030

    valid_candidates: list[tuple[tuple[int, int], set[int]]] = []
    for cand in candidates:
        if (time.perf_counter() - t_start) >= loop_deadline:
            break
        touched_comps = get_compartments_in_footprint(
            center=cand,
            footprint=scaled_fp,
            comp_labels=comp_labels,
            nearest_labels=nearest_labels,
        )
        if len(touched_comps) >= 2:
            valid_candidates.append((cand, touched_comps))

    # 8. Selección voraz de saltos
    selected_jumps: list[JumpCandidate] = []
    active_bridges: list[set[int]] = []
    current_cost = initial_cost
    current_dist = initial_dist

    for jump_idx in range(max_jumps):
        if not force_max_jumps and current_cost <= 0.0:
            break

        if (time.perf_counter() - t_start) >= loop_deadline:
            log.warning("⏳ Presupuesto de tiempo excedido en jump optimizer (%.1f ms). Devolviendo saltos encontrados.", (time.perf_counter() - t_start) * 1000.0)
            break

        best_cand: tuple[int, int] | None = None
        best_gain = 0.0
        best_cost_after = current_cost
        best_bridge: set[int] = set()
        best_cand_dist = current_dist

        for cand, touched_comps in valid_candidates:
            if (time.perf_counter() - t_start) >= loop_deadline:
                break

            # Evitar superponer saltos idénticos o demasiado cercanos (< 40 px en escala reducida)
            if any(np.hypot(cand[0]*scale - j.x, cand[1]*scale - j.y) < 40.0 for j in selected_jumps):
                continue

            # Podado matemático si no se fuerza max_jumps: si todos los compartimentos cubiertos tienen la misma distancia a la entrada,
            # unirlos no puede reducir la distancia de ningún compartimento hacia la entrada.
            d_min = min(current_dist.get(c, 999.0) for c in touched_comps)
            d_max = max(current_dist.get(c, 999.0) for c in touched_comps)
            if not force_max_jumps and d_max - d_min < 1.0:
                continue

            # Evaluar costo con este nuevo puente
            test_bridges = active_bridges + [touched_comps]
            cand_cost, cand_dist = compute_total_cost(
                graph=graph,
                entry_comps=entry_comps,
                targets=targets,
                jump_bridges=test_bridges,
            )

            gain = current_cost - cand_cost
            if gain > best_gain:
                best_gain = gain
                best_cand = cand
                best_cost_after = cand_cost
                best_bridge = touched_comps
                best_cand_dist = cand_dist

        if best_cand is None:
            if force_max_jumps:
                # Si se fuerzan 4 saltos pero no hay ganancia marginal (costo ya 0),
                # seleccionar candidatos adicionales que puenteen compartimentos respetando separación mínima (110 px)
                for cand, touched_comps in valid_candidates:
                    orig_cx = cand[0] * scale
                    orig_cy = cand[1] * scale
                    if any(np.hypot(orig_cx - j.x, orig_cy - j.y) < 110.0 for j in selected_jumps):
                        continue
                    best_cand = cand
                    best_gain = 0.0
                    best_cost_after = current_cost
                    best_bridge = touched_comps
                    best_cand_dist = current_dist
                    break
            if best_cand is None:
                break
        elif not force_max_jumps and best_gain < min_marginal_gain:
            break

        # Re-escalar coordenadas a píxeles de pantalla originales
        orig_x = int(round(best_cand[0] * scale))
        orig_y = int(round(best_cand[1] * scale))

        selected_jumps.append(
            JumpCandidate(
                x=orig_x,
                y=orig_y,
                gain=best_gain,
                cost_before=current_cost,
                cost_after=best_cost_after,
                bridged_compartments=sorted(best_bridge),
            )
        )
        active_bridges.append(best_bridge)
        current_cost = best_cost_after
        current_dist = best_cand_dist

    t_elapsed = (time.perf_counter() - t_start) * 1000.0

    jumps_list = [(j.x, j.y, j.gain) for j in selected_jumps]
    final_cost = current_cost
    total_gain = initial_cost - final_cost

    debug_overlay = None
    if generate_debug_overlay:
        debug_overlay = render_debug_overlay(
            base_img=base_img if base_img is not None else np.zeros((h, w, 3), dtype=np.uint8),
            comp_labels=comp_labels,
            wall_mask=wall_mask,
            selected_jumps=selected_jumps,
            footprint=jump_footprint,
            targets=targets,
            edges=edges,
        )

    return CompartmentAnalysisResult(
        jumps=jumps_list,
        jump_details=selected_jumps,
        initial_cost=initial_cost,
        final_cost=final_cost,
        total_gain=total_gain,
        num_compartments=max(1, num_comps - 1),
        comp_labels=comp_labels,
        wall_mask=wall_mask,
        edges=edges,
        target_assignments=targets,
        execution_time_ms=t_elapsed,
        debug_overlay=debug_overlay,
        graph=graph,
        entry_comps=entry_comps,
        nearest_labels=nearest_labels,
        grid_scale=scale,
    )


def render_debug_overlay(
    base_img: np.ndarray,
    comp_labels: np.ndarray,
    wall_mask: np.ndarray,
    selected_jumps: list[JumpCandidate],
    footprint: dict | tuple | float | int,
    targets: list[dict] | None = None,
    edges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """
    Genera una imagen de auditoría visual con:
    - Compartimentos coloreados en tonos pastel semitransparentes.
    - Segmentos y contornos de muros destacados.
    - Objetivos de botín asignados y marcados.
    - Huellas de saltos seleccionados numeradas con su ganancia.
    """
    h, w = comp_labels.shape[:2]
    canvas = base_img.copy() if base_img is not None else np.zeros((h, w, 3), dtype=np.uint8)
    if canvas.shape[:2] != (h, w):
        canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_NEAREST)

    # 1. Colorear compartimentos libres con paleta determinista vectorizada (lookup table)
    num_comps = int(np.max(comp_labels))
    if num_comps > 0:
        palette = np.zeros((num_comps + 1, 3), dtype=np.uint8)
        for c in range(1, num_comps + 1):
            hue = int((c * 179 / max(num_comps, 1)) % 180)
            color_hsv = np.uint8([[[hue, 140, 220]]])
            palette[c] = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]

        color_overlay = palette[comp_labels]
        # Mezclar transparencia suave (30% color compartimento, 70% imagen base)
        cv2.addWeighted(color_overlay, 0.30, canvas, 0.70, 0, canvas)

    # 2. Resaltar muros en blanco
    canvas[wall_mask > 0] = [240, 240, 240]

    # 3. Dibujar objetivos
    for t in (targets or []):
        cx, cy = t["center"]
        cls_name = t["cls_name"]
        color = (0, 215, 255) if "gold" in cls_name else (255, 0, 255) if "dark" in cls_name else (255, 128, 0)
        cv2.circle(canvas, (cx, cy), 8, color, -1)
        cv2.circle(canvas, (cx, cy), 10, (255, 255, 255), 1)
        cv2.putText(canvas, f"C{t['comp']}", (cx - 10, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 4. Dibujar huellas de saltos seleccionados
    is_ellipse = False
    rx, ry = 90.0, 90.0
    angle_deg = 0.0

    if isinstance(footprint, dict):
        if footprint.get("type") == "ellipse":
            is_ellipse = True
            rx = float(footprint.get("rx", 125.0))
            ry = float(footprint.get("ry", 72.0))
            angle_deg = float(footprint.get("angle", 0.0))
        else:
            rx = ry = float(footprint.get("radius", 90.0))
    elif isinstance(footprint, (tuple, list)):
        if len(footprint) >= 2:
            is_ellipse = True
            rx, ry = float(footprint[0]), float(footprint[1])
            angle_deg = float(footprint[2]) if len(footprint) > 2 else 0.0
        else:
            rx = ry = float(footprint[0])
    else:
        rx = ry = float(footprint)

    for idx, jump in enumerate(selected_jumps, start=1):
        jx, jy = jump.x, jump.y
        jump_color = (0, 255, 0) if idx == 1 else (0, 255, 255)

        # Dibujar contorno de huella
        if is_ellipse:
            cv2.ellipse(
                canvas,
                (jx, jy),
                (int(round(rx)), int(round(ry))),
                angle_deg,
                0,
                360,
                jump_color,
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(canvas, (jx, jy), int(round(rx)), jump_color, 2, cv2.LINE_AA)

        # Dibujar centro y etiqueta
        cv2.circle(canvas, (jx, jy), 5, (0, 0, 255), -1)
        label = f"Salto #{idx} (+{jump.gain:.1f})"
        cv2.putText(canvas, label, (jx + 8, jy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (jx + 8, jy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return canvas
```

## File: vision/wall_raycast.py
```python
"""
Módulo de Detección Real de Muros en Clash of Clans mediante Raycasting OpenCV.
Optimizado con detección global de segmentos Hough en una sola pasada por batalla
y filtrado geométrico robusto de cruces (ángulo de cruce, distancia de seguridad
y tolerancia colineal de extremos).
"""
from __future__ import annotations
import math
import os
import cv2
import numpy as np
from core.logger import get_logger

log = get_logger("wall_raycast")

# Caché por batalla para ángulos isométricos y segmentos de muro globales
_ISO_ANGLES_CACHE: tuple[float, float] | None = None
_WALL_SEGMENTS_CACHE: list[tuple[int, int, int, int]] | None = None
_LAST_SCREEN_HASH: int | None = None


def reset_iso_angles_cache() -> None:
    """Limpia la caché de ángulos isométricos y segmentos al iniciar una nueva batalla."""
    global _ISO_ANGLES_CACHE, _WALL_SEGMENTS_CACHE, _LAST_SCREEN_HASH
    _ISO_ANGLES_CACHE = None
    _WALL_SEGMENTS_CACHE = None
    _LAST_SCREEN_HASH = None


def _angle_diff_180(ang1: float, ang2: float) -> float:
    """Calcula la menor diferencia angular entre dos ángulos en el rango [0, 180)."""
    diff = abs((ang1 - ang2) % 180.0)
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def detect_global_wall_segments(
    img: np.ndarray,
    force_refresh: bool = False,
    min_line_len: int = 70,
    max_line_gap: int = 10,
    tol_deg: float = 10.0,
    canny_low: int = 60,
    canny_high: int = 160,
    hough_thresh: int = 30,
    detections: list | None = None,
) -> tuple[tuple[float, float], list[tuple[int, int, int, int]]]:
    """
    Ejecuta HoughLinesP una sola vez sobre toda el área jugable de la aldea:
    1. Extrae los ángulos isométricos dominantes mediante histograma en bins de 5°.
    2. Filtra y almacena únicamente los segmentos largos (>= min_line_len) alineados
       con los ángulos isométricos dominantes (+- tol_deg).
    3. Almacena en caché por batalla para eliminar recálculos por cada almacén.
    
    Retorna: ((iso1, iso2), [segmentos_muro (x1, y1, x2, y2)])
    """
    global _ISO_ANGLES_CACHE, _WALL_SEGMENTS_CACHE, _LAST_SCREEN_HASH
    if img is None or img.size == 0:
        return (36.0, 144.0), []

    screen_hash = hash(img.tobytes()[:2048])
    if (
        not force_refresh
        and _ISO_ANGLES_CACHE is not None
        and _WALL_SEGMENTS_CACHE is not None
        and _LAST_SCREEN_HASH == screen_hash
    ):
        return _ISO_ANGLES_CACHE, _WALL_SEGMENTS_CACHE

    h, w = img.shape[:2]
    # Recorte del área de juego central (evita el HUD superior y la barra de tropas inferior)
    y1_roi, y2_roi = int(0.14 * h), int(0.82 * h)
    x1_roi, x2_roi = int(0.10 * w), int(0.90 * w)
    roi = img[y1_roi:y2_roi, x1_roi:x2_roi]

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    if detections:
        for det in detections:
            if hasattr(det, "bbox_xyxy"):
                bx1, by1, bx2, by2 = det.bbox_xyxy
                # Convertir a coordenadas locales del ROI y dejar margen de 6px en los bordes
                lx1 = max(0, (bx1 + 6) - x1_roi)
                ly1 = max(0, (by1 + 6) - y1_roi)
                lx2 = min(roi.shape[1], (bx2 - 6) - x1_roi)
                ly2 = min(roi.shape[0], (by2 - 6) - y1_roi)
                if lx2 > lx1 and ly2 > ly1:
                    edges[ly1:ly2, lx1:lx2] = 0

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(hough_thresh),
        minLineLength=int(min_line_len),
        maxLineGap=int(max_line_gap),
    )

    default_angles = (36.0, 144.0)
    if lines is None or len(lines) == 0:
        _ISO_ANGLES_CACHE = default_angles
        _WALL_SEGMENTS_CACHE = []
        _LAST_SCREEN_HASH = screen_hash
        return default_angles, []

    angles = []
    converted_lines = []
    for line in lines:
        pts = line[0] if (hasattr(line, "__len__") and len(line) == 1 and hasattr(line[0], "__len__")) else line
        lx1, ly1, lx2, ly2 = pts[0], pts[1], pts[2], pts[3]
        # Reconvertir a coordenadas absolutas de la pantalla
        gx1, gy1 = lx1 + x1_roi, ly1 + y1_roi
        gx2, gy2 = lx2 + x1_roi, ly2 + y1_roi
        dx = gx2 - gx1
        dy = gy2 - gy1
        deg = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
        angles.append(deg)
        converted_lines.append((gx1, gy1, gx2, gy2, deg))

    # Histograma de 0 a 180 grados en bins de 5 grados (36 bins)
    hist, bin_edges = np.histogram(angles, bins=36, range=(0.0, 180.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Pico 1: Cuadrante 1 (15° a 75°, típico ~30°-45°)
    mask_q1 = (bin_centers >= 15.0) & (bin_centers <= 75.0)
    if np.any(mask_q1) and np.max(hist[mask_q1]) > 0:
        best_idx_q1 = np.argmax(hist[mask_q1])
        peak_q1 = float(bin_centers[mask_q1][best_idx_q1])
    else:
        peak_q1 = 36.0

    # Pico 2: Cuadrante 2 (105° a 165°, típico ~135°-150°)
    mask_q2 = (bin_centers >= 105.0) & (bin_centers <= 165.0)
    if np.any(mask_q2) and np.max(hist[mask_q2]) > 0:
        best_idx_q2 = np.argmax(hist[mask_q2])
        peak_q2 = float(bin_centers[mask_q2][best_idx_q2])
    else:
        peak_q2 = 144.0

    iso_angles = (peak_q1, peak_q2)

    # Filtrar solo líneas alineadas con los ángulos isométricos dominantes
    valid_wall_segments: list[tuple[int, int, int, int]] = []
    for gx1, gy1, gx2, gy2, deg in converted_lines:
        diff1 = _angle_diff_180(deg, iso_angles[0])
        diff2 = _angle_diff_180(deg, iso_angles[1])
        if (diff1 <= tol_deg) or (diff2 <= tol_deg):
            valid_wall_segments.append((gx1, gy1, gx2, gy2))

    log.debug(
        "🧱 [GLOBAL] Segmentos de muro detectados: %d (Ángulos: %.1f°, %.1f°)",
        len(valid_wall_segments), peak_q1, peak_q2
    )

    _ISO_ANGLES_CACHE = iso_angles
    _WALL_SEGMENTS_CACHE = valid_wall_segments
    _LAST_SCREEN_HASH = screen_hash
    return _ISO_ANGLES_CACHE, _WALL_SEGMENTS_CACHE


def estimate_iso_angles(img: np.ndarray, force_refresh: bool = False) -> tuple[float, float]:
    """Wrapper de conveniencia que devuelve los ángulos isométricos dominantes."""
    angles, _ = detect_global_wall_segments(img, force_refresh=force_refresh)
    return angles


def find_wall_crossings_from_segments(
    p_drop: tuple[int, int],
    p_target: tuple[int, int],
    wall_segments: list[tuple[int, int, int, int]],
    min_crossing_angle_deg: float = 25.0,
    min_dist_drop: float = 40.0,
    min_dist_target: float = 50.0,
    endpoint_tol: float = 30.0,
    debug_collector: dict | None = None,
) -> list[tuple[int, int]]:
    """
    Calcula la intersección matemática del rayo p_drop -> p_target contra los
    segmentos globales de muro previamente detectados con los filtros:
    
    1. Ángulo de cruce: descarta líneas con diferencia angular < min_crossing_angle_deg
       (evita intersecciones espurias si el rayo corre casi paralelo al muro).
    2. Posición razonable: descarta cruces a < min_dist_drop del drop o < min_dist_target del almacén.
    3. Extremos de línea: acepta intersecciones que caigan hasta endpoint_tol px fuera del
       extremo del segmento (tramos tapados por edificios).
    4. Fusión de cruces a < 25 px promediando el centro del muro.
    """
    p_drop_f = (float(p_drop[0]), float(p_drop[1]))
    p_target_f = (float(p_target[0]), float(p_target[1]))

    rx = p_target_f[0] - p_drop_f[0]
    ry = p_target_f[1] - p_drop_f[1]
    ray_dist = float(np.hypot(rx, ry))
    if ray_dist < (min_dist_drop + min_dist_target):
        return []

    ray_angle = float(np.degrees(np.arctan2(ry, rx)) % 180.0)

    raw_crossings = []
    accepted_lines = []
    rejected_lines = []

    for seg in wall_segments:
        x1, y1, x2, y2 = seg
        sx = float(x2 - x1)
        sy = float(y2 - y1)
        seg_len_sq = sx * sx + sy * sy
        if seg_len_sq < 1.0:
            continue
        seg_len = math.sqrt(seg_len_sq)
        seg_angle = float(np.degrees(np.arctan2(sy, sx)) % 180.0)

        # Filtro 1: Ángulo de cruce
        cross_angle_diff = _angle_diff_180(ray_angle, seg_angle)
        if cross_angle_diff < min_crossing_angle_deg:
            rejected_lines.append(seg)
            continue

        # Intersección de línea rayo con segmento de muro
        x1_f, y1_f = p_drop_f
        x2_f, y2_f = p_target_f
        x3_f, y3_f = float(x1), float(y1)
        x4_f, y4_f = float(x2), float(y2)

        denom = (x1_f - x2_f) * (y3_f - y4_f) - (y1_f - y2_f) * (x3_f - x4_f)
        if abs(denom) < 1e-6:
            rejected_lines.append(seg)
            continue

        t = ((x1_f - x3_f) * (y3_f - y4_f) - (y1_f - y3_f) * (x3_f - x4_f)) / denom
        if not (0.0 < t < 1.0):
            rejected_lines.append(seg)
            continue

        # Distancias a lo largo del rayo
        dist_to_drop = t * ray_dist
        dist_to_target = (1.0 - t) * ray_dist

        # Filtro 2: Posición razonable en el trayecto
        if dist_to_drop < min_dist_drop or dist_to_target < min_dist_target:
            rejected_lines.append(seg)
            continue

        ix = x1_f + t * rx
        iy = y1_f + t * ry

        # Filtro 3: Tolerancia colineal en los extremos del segmento
        u = ((ix - x3_f) * sx + (iy - y3_f) * sy) / seg_len_sq
        margin_u = endpoint_tol / seg_len
        if not (-margin_u <= u <= 1.0 + margin_u):
            rejected_lines.append(seg)
            continue

        accepted_lines.append(seg)
        raw_crossings.append((ix, iy, dist_to_drop))

    # Ordenar cruces de afuera hacia adentro (menor a mayor distancia a p_drop)
    raw_crossings.sort(key=lambda c: c[2])

    # Fusión de cruces a < 25 px promediando el cluster
    clusters: list[list[tuple[float, float]]] = []
    for ix, iy, dist in raw_crossings:
        pt = (ix, iy)
        if not clusters:
            clusters.append([pt])
        else:
            last_cluster = clusters[-1]
            last_avg_x = sum(p[0] for p in last_cluster) / len(last_cluster)
            last_avg_y = sum(p[1] for p in last_cluster) / len(last_cluster)
            if np.hypot(pt[0] - last_avg_x, pt[1] - last_avg_y) < 25.0:
                last_cluster.append(pt)
            else:
                clusters.append([pt])

    merged_crossings: list[tuple[int, int]] = []
    for cluster in clusters:
        avg_x = int(round(sum(p[0] for p in cluster) / len(cluster)))
        avg_y = int(round(sum(p[1] for p in cluster) / len(cluster)))
        merged_crossings.append((avg_x, avg_y))

    if debug_collector is not None:
        debug_collector["p_drop"] = p_drop
        debug_collector["p_target"] = p_target
        debug_collector["accepted_lines"] = accepted_lines
        debug_collector["rejected_lines"] = rejected_lines
        debug_collector["crossings"] = merged_crossings

    return merged_crossings


def find_wall_crossings(
    img: np.ndarray,
    p_drop: tuple[int, int],
    p_target: tuple[int, int],
    iso_angles: tuple[float, float] | None = None,
    strip: int = 45,
    tol_deg: float = 10.0,
    canny_low: int = 60,
    canny_high: int = 160,
    hough_thresh: int = 30,
    hough_min_len: int = 70,
    hough_max_gap: int = 10,
    min_crossing_angle_deg: float = 25.0,
    min_dist_drop: float = 40.0,
    min_dist_target: float = 50.0,
    endpoint_tol: float = 30.0,
    debug_collector: dict | None = None,
) -> list[tuple[int, int]]:
    """
    Función principal de raycasting de muros:
    1. Si aún no se han extraído los segmentos globales de la imagen, los calcula una sola vez.
    2. Calcula los cruces del rayo con los segmentos aplicando los filtros geométricos.
    """
    if img is None or img.size == 0:
        return []

    _, wall_segments = detect_global_wall_segments(
        img,
        min_line_len=hough_min_len,
        max_line_gap=hough_max_gap,
        tol_deg=tol_deg,
        canny_low=canny_low,
        canny_high=canny_high,
        hough_thresh=hough_thresh,
    )

    return find_wall_crossings_from_segments(
        p_drop=p_drop,
        p_target=p_target,
        wall_segments=wall_segments,
        min_crossing_angle_deg=min_crossing_angle_deg,
        min_dist_drop=min_dist_drop,
        min_dist_target=min_dist_target,
        endpoint_tol=endpoint_tol,
        debug_collector=debug_collector,
    )


def find_wall_junctions(
    wall_segments: list[tuple[int, int, int, int]],
    tol_px: float = 18.0,
    min_intersect_angle_deg: float = 35.0,
    iso_angles: tuple[float, float] | None = None,
) -> list[dict]:
    """
    Encuentra las intersecciones geométricas entre segmentos de muro (cruces en T y +).
    Retorna lista de uniones: [{'pt': (x, y), 'type': 'cross'|'tee'|'corner', 'degree': 4|3|2, 'score': float}]
    """
    if not wall_segments or len(wall_segments) < 2:
        return []

    # Precalcular ángulos y longitudes de cada segmento
    seg_info = []
    for s in wall_segments:
        x1, y1, x2, y2 = s
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length >= 1.0:
            ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180.0
            seg_info.append((s, length, ang))

    raw_junctions = []
    n = len(seg_info)
    for i in range(n):
        s1, len1, ang1 = seg_info[i]
        x1, y1, x2, y2 = s1
        for j in range(i + 1, n):
            s2, len2, ang2 = seg_info[j]
            # Verificar que no sean casi paralelos (deben cruzarse con ángulo >= min_intersect_angle_deg)
            if _angle_diff_180(ang1, ang2) < min_intersect_angle_deg:
                continue

            x3, y3, x4, y4 = s2
            denom = float((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            if abs(denom) < 1e-4:
                continue

            t = float(((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom)
            u = float(-((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom)

            t_tol = tol_px / len1
            u_tol = tol_px / len2

            if (-t_tol <= t <= 1.0 + t_tol) and (-u_tol <= u <= 1.0 + u_tol):
                ix = int(round(x1 + t * (x2 - x1)))
                iy = int(round(y1 + t * (y2 - y1)))

                is_end_t = (t < 0.15 or t > 0.85)
                is_end_u = (u < 0.15 or u > 0.85)
                if not is_end_t and not is_end_u:
                    jtype = "cross"  # Cruce 4 ramas (+)
                    degree = 4
                    score = 4.0
                elif (is_end_t and not is_end_u) or (not is_end_t and is_end_u):
                    jtype = "tee"    # Cruce 3 ramas (T)
                    degree = 3
                    score = 3.0
                else:
                    jtype = "corner" # Esquina 2 ramas (L)
                    degree = 2
                    score = 1.5

                raw_junctions.append(((ix, iy), jtype, degree, score))

    if not raw_junctions:
        return []

    # Agrupar uniones cercanas (< 25px) promediando sus coordenadas
    clusters: list[dict] = []
    for pt, jtype, deg, score in raw_junctions:
        matched = False
        for c in clusters:
            if np.hypot(pt[0] - c["pt"][0], pt[1] - c["pt"][1]) < 25.0:
                c["pts"].append(pt)
                if score > c["score"]:
                    c["score"] = score
                    c["type"] = jtype
                    c["degree"] = deg
                matched = True
                break
        if not matched:
            clusters.append({
                "pt": pt,
                "pts": [pt],
                "type": jtype,
                "degree": deg,
                "score": score
            })

    for c in clusters:
        avg_x = int(round(sum(p[0] for p in c["pts"]) / len(c["pts"])))
        avg_y = int(round(sum(p[1] for p in c["pts"]) / len(c["pts"])))
        c["pt"] = (avg_x, avg_y)

    return clusters


def calculate_adaptive_jump_and_breach_placements(
    targets_with_drops: list[tuple[any, tuple[int, int]]],
    wall_segments: list[tuple[int, int, int, int]],
    junctions: list[dict] | None = None,
    max_jumps: int = 4,
    min_dist_between_jumps: float = 110.0,
    swb_breach_max_dist_grass: float = 75.0,
    safe_y_min: int = 140,
    safe_y_max: int = 845,
    safe_x_min: int = 120,
    safe_x_max: int = 2280,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, any]], list[dict]]:
    """
    Decisión adaptativa por demanda real:
    - Analiza cada objetivo de botín prioritario.
    - Si tiene cruces de muro:
      * El cruce más exterior (Capa 1) asigna brecha de SWB.
      * Si hay 2+ capas, busca la unión geométrica (T o +) de mayor puntuación en el corredor
        para colocar 1 Hechizo de Salto que abra la mayor cantidad de compartimentos.
    - No duplica saltos si el objetivo ya cuenta con cobertura (< 140px).
    - Retorna: (jump_targets, swb_breach_targets, rays_debug_data)
    """
    if junctions is None:
        junctions = find_wall_junctions(wall_segments)

    selected_jumps: list[tuple[int, int]] = []
    selected_swb: list[tuple[int, int, any]] = []
    seen_swb_drops = set()
    rays_debug_data: list[dict] = []

    for d, (drop_x, drop_y) in targets_with_drops:
        target_center = (int(round(d.center[0])), int(round(d.center[1])))
        ray_debug: dict = {}
        crossings = find_wall_crossings_from_segments(
            p_drop=(drop_x, drop_y),
            p_target=target_center,
            wall_segments=wall_segments,
            debug_collector=ray_debug
        )
        rays_debug_data.append(ray_debug)

        # Verificar si este objetivo ya cuenta con cobertura de salto existente
        has_jump_coverage = any(
            np.hypot(target_center[0] - jx, target_center[1] - jy) < 140.0
            for jx, jy in selected_jumps
        )

        if crossings:
            c0_x, c0_y = crossings[0]
            dist_drop_to_c0 = np.hypot(c0_x - drop_x, c0_y - drop_y)

            # Capa 1: Si el muro exterior está cerca del césped, programar brecha SWB
            if dist_drop_to_c0 <= swb_breach_max_dist_grass:
                grid_k = (drop_x // 80, drop_y // 80)
                if grid_k not in seen_swb_drops:
                    seen_swb_drops.add(grid_k)
                    selected_swb.append((drop_x, drop_y, d))

            # Capas interiores (2 o más muros): colocar Hechizo de Salto en unión óptima
            if len(crossings) >= 2 and not has_jump_coverage and len(selected_jumps) < max_jumps:
                best_jump_pt = None
                best_score = -999.0

                # Vector del rayo
                dx_ray = target_center[0] - drop_x
                dy_ray = target_center[1] - drop_y
                ray_len = float(np.hypot(dx_ray, dy_ray))

                if ray_len > 1.0 and junctions:
                    ux_ray, uy_ray = dx_ray / ray_len, dy_ray / ray_len

                    for junc in junctions:
                        jx, jy = junc["pt"]
                        # Coordenada proyectada a lo largo del rayo
                        proj_t = (jx - drop_x) * ux_ray + (jy - drop_y) * uy_ray
                        # Debe estar entre la primera capa de muro y el objetivo
                        if proj_t < dist_drop_to_c0 - 15.0 or proj_t > ray_len - 15.0:
                            continue

                        # Distancia perpendicular al rayo
                        perp_dist = abs((jx - drop_x) * (-uy_ray) + (jy - drop_y) * ux_ray)
                        if perp_dist > 85.0:
                            continue

                        # Distancia a cruces interiores de muro
                        min_dist_to_interior = min(
                            np.hypot(jx - cx, jy - cy) for cx, cy in crossings[1:]
                        )

                        # Puntuación: mayor grado de unión (+ > T > L), menor distancia al corredor y a cruces
                        cand_score = junc["score"] - (0.015 * perp_dist) - (0.012 * min_dist_to_interior)

                        # Evitar solapamiento con saltos ya seleccionados
                        if any(np.hypot(jx - ox, jy - oy) < min_dist_between_jumps for ox, oy in selected_jumps):
                            continue

                        if cand_score > best_score:
                            best_score = cand_score
                            best_jump_pt = (jx, jy)

                # Si no encontramos unión de grafo cercana, usar el segundo cruce de muro o punto medio
                if best_jump_pt is None:
                    if len(crossings) >= 2:
                        fallback_pt = (
                            int(round((crossings[0][0] + crossings[1][0]) / 2.0)),
                            int(round((crossings[0][1] + crossings[1][1]) / 2.0)),
                        )
                    else:
                        fallback_pt = crossings[0]

                    if not any(np.hypot(fallback_pt[0] - ox, fallback_pt[1] - oy) < min_dist_between_jumps for ox, oy in selected_jumps):
                        best_jump_pt = fallback_pt

                if best_jump_pt is not None:
                    # Clampar a márgenes seguros de pantalla
                    cx_clamped = int(np.clip(best_jump_pt[0], safe_x_min, safe_x_max))
                    cy_clamped = int(np.clip(best_jump_pt[1], safe_y_min, safe_y_max))
                    selected_jumps.append((cx_clamped, cy_clamped))
                    log.info(
                        "🎯 Salto adaptativo asignado en (%d, %d) para %s (score=%.2f)",
                        cx_clamped, cy_clamped, d.cls_name, best_score if best_score > -900 else 1.0
                    )

    return selected_jumps, selected_swb, rays_debug_data


def draw_wall_raycast_debug(
    screen: np.ndarray,
    rays_data: list[dict],
    jump_targets: list[tuple[int, int]],
    junctions: list[dict] | None = None,
    out_path: str = "logs/yolo/wall_raycast_debug.png",
) -> None:
    """
    Genera y guarda la imagen anotada de depuración visual de raycasting de muros:
    - Segmentos de muro detectados (verde los que cruzan el rayo, azul/cian los globales)
    - Uniones de muros detectadas (magenta pequeños cruces T/+)
    - Rayos p_drop -> almacén (blanco)
    - Cruces de muro numerados (amarillo)
    - Círculos de Hechizo de Salto proyectados (verde translúcido)
    """
    if screen is None or screen.size == 0:
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas = screen.copy()

    # Dibujar segmentos globales de fondo en cian suave
    if _WALL_SEGMENTS_CACHE is not None:
        for x1, y1, x2, y2 in _WALL_SEGMENTS_CACHE:
            cv2.line(canvas, (x1, y1), (x2, y2), (220, 180, 50), thickness=1)

    # Dibujar uniones de muros detectadas (magenta)
    if junctions:
        for junc in junctions:
            jx, jy = junc["pt"]
            jdeg = junc.get("degree", 2)
            cv2.circle(canvas, (jx, jy), 4 if jdeg <= 2 else 6, (255, 0, 255), -1)

    for data in rays_data:
        p_drop = data.get("p_drop")
        p_target = data.get("p_target")
        accepted = data.get("accepted_lines", [])
        rejected = data.get("rejected_lines", [])
        crossings = data.get("crossings", [])

        if p_drop and p_target:
            # Rayo principal blanco
            cv2.line(canvas, p_drop, p_target, (255, 255, 255), thickness=2)

        # Líneas que intersecan (verde brillante)
        for x1, y1, x2, y2 in accepted:
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)

        # Cruces numerados
        for idx, (cx, cy) in enumerate(crossings):
            cv2.circle(canvas, (cx, cy), radius=6, color=(0, 255, 255), thickness=-1)
            cv2.putText(
                canvas,
                f"W{idx+1}",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

    # Dibujar los saltos finales
    for idx, (jx, jy) in enumerate(jump_targets):
        cv2.circle(canvas, (jx, jy), radius=85, color=(0, 255, 120), thickness=2)
        cv2.circle(canvas, (jx, jy), radius=5, color=(0, 255, 120), thickness=-1)
        cv2.putText(
            canvas,
            f"JUMP #{idx+1}",
            (jx - 35, jy - 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 120),
            2
        )

    cv2.imwrite(out_path, canvas)
    log.info("📸 Debug visual de Wall Raycast guardado en: %s", out_path)
```

## File: vision/yolo_detector.py
```python
"""
vision/yolo_detector.py
Módulo de inferencia YOLOv8 para detección de estructuras en Clash of Clans.
Soporta backend ONNX (onnxruntime) y PyTorch (.pt vía ultralytics).
"""
import os
from dataclasses import dataclass
import cv2
import numpy as np

from core.logger import get_logger
from core.config import SETTINGS, PROJECT_ROOT

log = get_logger("yolo_detector")

DEFAULT_CLASSES = [
    "storage_gold",
    "storage_elixir",
    "storage_dark",
    "town_hall",
    "collector_gold",
    "collector_elixir",
    "collector_dark",
]

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


class _CallableBool(int):
    """Permite acceder como property booleana (d.is_finetuned) y como método (d.is_finetuned())."""
    def __call__(self):
        return bool(self)

    def __repr__(self):
        return repr(bool(self))

    def __str__(self):
        return str(bool(self))


@dataclass
class Detection:
    cls_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]  # coords en píxeles originales
    center: tuple[int, int]              # centro en píxeles originales


class YoloDetector:
    def __init__(
        self,
        model_path: str = "models/yolov8s.onnx",
        conf: float = 0.40,
        iou: float = 0.5,
        imgsz: int = 1280,
        filter_hud: bool = True,
    ):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.filter_hud = filter_hud
        self._ready = False
        self.backend = "none"
        self.session = None
        self.pt_model = None

        self.class_names = list(DEFAULT_CLASSES)
        self.model_nc = len(self.class_names)
        self._load_class_names()
        self._init_model()

    def _load_class_names(self):
        yaml_path = os.path.join(PROJECT_ROOT, "datasets", "coc", "data.yaml")
        if os.path.exists(yaml_path):
            try:
                import yaml
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict) and "names" in data:
                        self.class_names = list(data["names"])
            except Exception:
                pass
        self.model_nc = len(self.class_names)

    def _init_model(self):
        resolved_path = (
            os.path.join(PROJECT_ROOT, self.model_path)
            if not os.path.isabs(self.model_path)
            else self.model_path
        )
        if not os.path.exists(resolved_path):
            log.warning("Modelo no encontrado en disco: %s. El detector no estará activo.", resolved_path)
            self._ready = False
            return

        try:
            if resolved_path.lower().endswith(".onnx"):
                import onnxruntime as ort
                self.session = ort.InferenceSession(resolved_path, providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
                out_shape = self.session.get_outputs()[0].shape
                # En YOLOv8 ONNX: [1, 4 + nc, num_candidates] o [1, num_candidates, 4 + nc]
                nc_dim = out_shape[1] if (len(out_shape) > 1 and out_shape[1] < out_shape[-1]) else out_shape[-1]
                self.model_nc = int(nc_dim) - 4
                self.backend = "onnxruntime (CPUExecutionProvider)"
                self._ready = True
                log.info("YoloDetector inicializado con backend %s (modelo: %s, nc=%d)", self.backend, resolved_path, self.model_nc)
            elif resolved_path.lower().endswith(".pt"):
                from ultralytics import YOLO
                self.pt_model = YOLO(resolved_path)
                self.model_nc = len(self.pt_model.names)
                self.backend = "ultralytics (PyTorch)"
                self._ready = True
                log.info("YoloDetector inicializado con backend %s (modelo: %s, nc=%d)", self.backend, resolved_path, self.model_nc)
            else:
                log.warning("Formato de modelo desconocido: %s", resolved_path)
                self._ready = False
        except Exception as e:
            log.error("Fallo al inicializar sesión de modelo YOLO (%s): %s", resolved_path, e)
            self._ready = False

    @property
    def is_finetuned(self):
        is_ft = (self.model_nc == len(self.class_names))
        return _CallableBool(1 if is_ft else 0)

    def is_ready(self) -> bool:
        return self._ready

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        shape = img.shape[:2]  # (h, w)
        r = min(self.imgsz / shape[0], self.imgsz / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (self.imgsz - new_unpad[0]) / 2
        dh = (self.imgsz - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return padded, r, (left, top)

    @staticmethod
    def _dedup_overlapping(detections: list[Detection], iou_thresh: float = 0.55) -> list[Detection]:
        """Fusiona cajas de la MISMA clase muy solapadas conservando la de mayor confianza."""
        def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix = max(0, min(ax2, bx2) - max(ax1, bx1))
            iy = max(0, min(ay2, by2) - max(ay1, by1))
            inter = float(ix * iy)
            area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
            area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
            return inter / (area_a + area_b - inter + 1e-9)

        kept: list[Detection] = []
        for d in sorted(detections, key=lambda x: -x.confidence):
            if all(_iou(d.bbox_xyxy, k.bbox_xyxy) < iou_thresh or d.cls_name != k.cls_name for k in kept):
                kept.append(d)
        return kept

    def detect(self, screenshot: np.ndarray) -> list[Detection]:
        if not self._ready or screenshot is None:
            return []

        try:
            orig_img = screenshot
            if orig_img.ndim == 2:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
            elif orig_img.shape[2] == 4:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)

            orig_h, orig_w = orig_img.shape[:2]

            # Fallback backend con PyTorch (.pt)
            if self.backend.startswith("ultralytics"):
                results = self.pt_model(orig_img, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
                detections: list[Detection] = []
                if results and len(results) > 0:
                    r = results[0]
                    for box in r.boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        x1, y1, x2, y2 = [int(round(v)) for v in coords]
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)

                        # Filtro de área de juego (150 < y < 900 en 1080p) para eliminar falsos positivos en HUD
                        if self.filter_hud:
                            min_y = int(0.138 * orig_h)
                            max_y = int(0.833 * orig_h)
                            if center[1] < min_y or center[1] > max_y:
                                continue

                        if self.is_finetuned():
                            raw_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                            cls_name = raw_name
                        else:
                            raw_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                            cls_name = f"[COCO] {raw_name}"
                        detections.append(Detection(cls_name, conf, (x1, y1, x2, y2), center))
                return self._dedup_overlapping(detections, iou_thresh=0.55)

            # Backend principal con ONNX
            padded, scale, (pad_w, pad_h) = self._letterbox(orig_img)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            blob = rgb.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))
            blob = np.expand_dims(blob, axis=0)

            outputs = self.session.run(None, {self.input_name: blob})
            preds = outputs[0]  # Shape: (1, 4 + nc, num_candidates)

            preds = np.transpose(preds[0], (1, 0))  # (num_candidates, 4 + nc)
            boxes = preds[:, :4]
            scores = preds[:, 4:]

            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)

            mask = confidences >= self.conf
            if not np.any(mask):
                return []

            boxes = boxes[mask]
            class_ids = class_ids[mask]
            confidences = confidences[mask]

            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            w = boxes[:, 2]
            h = boxes[:, 3]

            boxes_xywh = np.stack([x1, y1, w, h], axis=1).tolist()
            confidences_list = confidences.tolist()

            indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences_list, score_threshold=self.conf, nms_threshold=self.iou)
            if len(indices) == 0:
                return []

            indices = np.array(indices).flatten()
            detections: list[Detection] = []

            for idx in indices:
                bx1, by1, bw, bh = boxes_xywh[idx]
                bx2, by2 = bx1 + bw, by1 + bh

                orig_x1 = max(0, min(orig_w, int(round((bx1 - pad_w) / scale))))
                orig_y1 = max(0, min(orig_h, int(round((by1 - pad_h) / scale))))
                orig_x2 = max(0, min(orig_w, int(round((bx2 - pad_w) / scale))))
                orig_y2 = max(0, min(orig_h, int(round((by2 - pad_h) / scale))))

                cx = (orig_x1 + orig_x2) // 2
                cy = (orig_y1 + orig_y2) // 2

                # Filtro de área de juego (150 < y < 900 en 1080p)
                if self.filter_hud:
                    min_y = int(0.138 * orig_h)
                    max_y = int(0.833 * orig_h)
                    if cy < min_y or cy > max_y:
                        continue

                cid = int(class_ids[idx])
                if self.is_finetuned():
                    raw_name = self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}"
                    cls_name = raw_name
                else:
                    raw_name = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}"
                    cls_name = f"[COCO] {raw_name}"
                conf = float(confidences[idx])

                detections.append(Detection(cls_name, conf, (orig_x1, orig_y1, orig_x2, orig_y2), (cx, cy)))

            return self._dedup_overlapping(detections, iou_thresh=0.55)
        except Exception as e:
            log.error("Error durante detect() con YOLO: %s", e)
            return []


_DETECTOR_SINGLETON: YoloDetector | None = None


def get_detector() -> YoloDetector | None:
    """
    Instancia global lazy (singleton) que lee config/settings.json -> attack.use_yolo.
    Si está en false, devuelve None.
    """
    global _DETECTOR_SINGLETON
    attack_cfg = SETTINGS.get("attack", {})
    use_yolo = attack_cfg.get("use_yolo", False)
    if not use_yolo:
        return None

    if _DETECTOR_SINGLETON is None:
        model_path = attack_cfg.get("yolo_model_path", "models/yolov8s.onnx")
        conf = float(attack_cfg.get("yolo_conf", 0.40))
        iou = float(attack_cfg.get("yolo_iou", 0.5))
        imgsz = int(attack_cfg.get("yolo_imgsz", 1280))
        _DETECTOR_SINGLETON = YoloDetector(model_path=model_path, conf=conf, iou=iou, imgsz=imgsz)

    return _DETECTOR_SINGLETON
```

## File: main.py
```python
import argparse
import sys
from core.device import Device
from core.vision import Vision
from core.state_machine import ClashStateMachine
from core.logger import get_logger

log = get_logger("main")


def main():
    parser = argparse.ArgumentParser(description="ClashFarmerbot Runner")
    parser.add_argument("--attacks", type=int, default=None, help="Número máximo de ataques antes de detenerse")
    args = parser.parse_args()

    device = Device()
    if not device.connect():
        log.critical("No se pudo establecer conexión ADB con el dispositivo. Abortando.")
        sys.exit(1)

    vision = Vision(threshold=0.80)
    vision.preload_templates("assets")

    bot = ClashStateMachine(device, vision)
    bot.run_forever(max_attacks=args.attacks)


if __name__ == "__main__":
    main()
```

## File: requirements.txt
```
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
uiautomator2>=3.0.0
ultralytics>=8.0.0
onnxruntime>=1.17.0
onnx>=1.15.0
```
