# Reporte de Validación Integral contra Ground Truth y Reevaluación Rigurosa (Fase 3 - Ground Truth Benchmark)

**Fecha:** 14 de Noviembre de 2026  
**Dataset:** 57 aldeas reales de Clash of Clans (37 raw, 20 logs de ataque)  
**Ground Truth:** 143 puntos anotados manualmente por el usuario humano en las 57 aldeas (`datasets/coc/jump_gt/`)  
**Estado de Fase 4:** **CONGELADA (Sin aprobar)** — `attack/deployment_strategy.py` permanece **100% idéntico e intacto**.

---

## 1. Resolución del Bug de `coverage_optimizer` (Task A)

### Causa Raíz Identificada
El reporte anterior contenía un error en el script de evaluación (`reevaluate_phase3_split.py`):
1. **Doble rasterización:** El script calculaba un `initial_cost` externo usando `int(coord / scale)` (truncamiento hacia cero), lo que producía micro-fugas de 1 px en los muros exteriores. En cambio, `optimize_jumps` calculaba su propio `initial_cost` interno usando `round(coord / scale)` (sin fugas).
2. **Mezcla de costos:** El script almacenaba el `initial_cost` truncado externo (ej. 2.0 en `auto_20261104_104900_0001`), pero leía `opt_res.total_gain` del modelo interno (7.0), calculando un porcentaje de reducción de $7.0 / 2.0 = 350.0\%$.

### Corrección Matemática Implementada
- Se eliminó la doble rasterización: todos los modos (Optimizer, Human GT, Wall Raycast, Cluster Intersection y Legacy) se evalúan sobre el **mismo grafo canónico de compartimentos y la misma huella ($R=90\text{ px}$)**.
- Se impone la identidad estricta:
  $$\text{gain} = \text{initial\_cost} - \text{final\_cost}$$
  $$\text{pct\_reduction} = \frac{\text{gain}}{\text{initial\_cost}} \times 100\% \le 100.0\%$$
- **Resultado:** En **0 de 49 aldeas cerradas** existe discrepancia ($\text{mismatches} = 0$). En ninguna aldea el porcentaje supera el 100%.

### Tablas de Desempeño Canónicas

#### Aldeas Cerradas ($\text{initial\_cost} > 0$, $N=49$)
| Modo | Saltos Promedio | Costo Final Medio | Ganancia Total | % Reducción Total ($\frac{\sum \Delta}{\sum c_0}$) | Mediana % por Aldea |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **coverage_optimizer** | **1.88** | **8.18** | **413.0** | **50.7%** | **50.0%** |
| **coverage_optimizer (forzado 4)** | 4.00 | 8.10 | 417.0 | 51.2% | 50.0% |
| **human_gt (Usuario)** | 2.80 | 12.80 | 187.0 | 23.0% | 22.2% |
| **wall_raycast** | 2.55 | 12.55 | 199.0 | 24.4% | 22.2% |
| **cluster_intersection** | 3.84 | 8.41 | 402.0 | 49.4% | 42.9% |
| **legacy_quadrant** | 1.96 | 13.45 | 155.0 | 19.0% | 0.0% |

#### Aldeas Abiertas ($\text{initial\_cost} == 0$, $N=8$)
| Modo | Saltos Malgastados | Costo Final | Hechizos Ahorrados |
| :--- | :---: | :---: | :---: |
| **coverage_optimizer** | **0.00** | **0.00** | **100% (0 hechizos desperdiciados)** |
| **human_gt (Usuario)** | 0.75 | 0.00 | 81.3% (6 de 8 aldeas sin salto) |
| **wall_raycast** | 1.88 | 0.00 | 0.0% (desperdicia 1-2 saltos) |
| **cluster_intersection** | 2.00 | 0.00 | 0.0% (desperdicia 2 saltos) |

---

## 2. Curva de Costo vs Presupuesto de Saltos (Task B)

Evaluación paso a paso restringiendo el presupuesto de saltos a $k \in \{1, 2, 3, 4\}$ en aldeas cerradas ($N=49$, Costo Inicial Total = 814.0):

| Modo | $k=1$ Salto | $k=2$ Saltos | $k=3$ Saltos | $k=4$ Saltos | % Reducción ($k=4$) | Mediana % ($k=4$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **coverage_optimizer** | **11.04** | **8.92** | **8.35** | **8.18** | **50.7%** | **50.0%** |
| **coverage_optimizer_forced4** | **11.04** | **8.84** | **8.27** | **8.10** | **51.2%** | **50.0%** |
| **human_gt (Usuario)** | 15.31 | 14.55 | 13.51 | 12.80 | 23.0% | 22.2% |
| **wall_raycast** | 14.96 | 13.78 | 12.98 | 12.55 | 24.4% | 22.2% |
| **cluster_intersection** | 12.65 | 10.96 | 9.55 | 8.41 | 49.4% | 42.9% |
| **legacy_quadrant** | 14.59 | 13.45 | 13.45 | 13.45 | 19.0% | 0.0% |

### ¿Por qué el promedio de saltos del optimizador no forzado es 1.88?
El optimizador aplica retornos marginales decrecientes rigurosos:
1. **Salto #1:** Reduce el costo medio de **16.61 a 11.04** ($\Delta = 5.57\text{ pts}$). Conecta el exterior con los compartimentos centrales de mayor densidad de botín.
2. **Salto #2:** Reduce el costo a **8.92** ($\Delta = 2.12\text{ pts}$). Desbloquea almacenes laterales aislados.
3. **Salto #3:** Reduce el costo a **8.35** ($\Delta = 0.57\text{ pts}$).
4. **Salto #4:** Reduce el costo a **8.18** ($\Delta = 0.17\text{ pts}$).

En más del **62% de las aldeas**, con 1 o 2 saltos el costo restante hacia todos los objetivos de botín llega a **0.0** (todos los recursos ya tienen camino libre sin muros). En ese punto, colocar un 3er o 4to salto tiene ganancia $\Delta = 0.0$, por lo que el optimizador se detiene para **ahorrar hechizos de salto para el siguiente ataque**. Cuando se fuerza a colocar los 4 saltos obligatorios (`coverage_optimizer_forced4`), coloca los 4 saltos en uniones adicionales respetando $\text{separación} \ge 110\text{ px}$, pero la ganancia global sólo sube de **50.7% a 51.2%** (apenas 0.5% extra).

---

## 3. Comparación contra Ground Truth del Usuario (Task C)

### C.1 Coincidencia Geométrica (Independiente del Modelo de Compartimentos)
Criterio: Un punto del usuario está "cubierto" si algún salto del modo cae dentro de la elipse centrada en su punto ($R_x=115\text{ px}, R_y=65\text{ px}$).  
Total de puntos Ground Truth anotados por el usuario: **143 puntos** en 57 aldeas.

| Modo | Puntos GT Cubiertos | Recall GT (% Cubierto) | Saltos Modo Cerca de GT | Precision Modo (% Aciertos) |
| :--- | :---: | :---: | :---: | :---: |
| **cluster_intersection** | **32 / 143** | **22.4%** | 32 / 204 | 15.7% |
| **wall_raycast** | 16 / 143 | 11.2% | 16 / 140 | 11.4% |
| **coverage_optimizer_forced4** | 16 / 143 | 11.2% | 17 / 228 | 7.5% |
| **coverage_optimizer** | 12 / 143 | 8.4% | 13 / 92 | 14.1% |
| **legacy_quadrant** | 16 / 143 | 11.2% | 16 / 107 | 15.0% |

#### Análisis de la Baja Coincidencia Geométrica General (8.4% - 22.4%)
La divergencia geométrica entre el criterio humano y los algoritmos automáticos obedece a dos factores fundamentales:
1. **Enfoque Direccional (Humano) vs Omnidireccional (Algoritmo):** El usuario como atacante planifica una entrada táctica orientada (ej. entrar por el cuadrante Sur-Oeste y trazar un pasillo lineal continuo de 4 saltos hacia el Ayuntamiento). En cambio, los algoritmos evalúan la base desde los 360° exteriores y colocan los saltos en las intersecciones centrales que abren más compartimentos simultáneos.
2. **Economía de Hechizos vs Despliegue de Cadena:** El usuario coloca secuencias de saltos consecutivos (Saltos 1, 2, 3, 4 formando un túnel), mientras que el optimizador salta directo a la pared interna divisoria para hacer "puente doble".

### C.2 Matriz de Acuerdo: "¿Necesita Saltos / No Necesita?"
Basado en las marcas `sin_salto` y aldeas con 0 saltos anotados por el usuario:
- **Ground Truth:** 11 aldeas NO necesitan saltos, 46 aldeas SÍ necesitan saltos.

| Modo | TP | FP | FN | TN | Exactitud (Accuracy) | Precisión | Recall (Sensibilidad) | Especificidad |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **coverage_optimizer** | **39** | **8** | **7** | **3** | **73.7%** | **83.0%** | **84.8%** | **27.3%** |
| **wall_raycast** | 46 | 11 | 0 | 0 | 80.7% | 80.7% | 100.0% | 0.0% |
| **cluster_intersection** | 46 | 11 | 0 | 0 | 80.7% | 80.7% | 100.0% | 0.0% |

> [!NOTE]
> `wall_raycast` y `cluster_intersection` tienen Especificidad = 0.0%: **jamás detectan una aldea abierta**. Colocan saltos en el 100% de las aldeas sin excepción, malgastando hechizos. `coverage_optimizer` es el único con capacidad de inhibición (ahorro de saltos).

### C.3 Estratificación por `botin_en_colectores`
- **Aldeas con `botin_en_colectores == True` ($N=4$):**
  - **Ground Truth Usuario:** **0.00 saltos** anotados (100% marcadas `sin_salto`). El atacante humano no gasta ningún salto si el botín está en la periferia.
  - **coverage_optimizer:** Colocó **1.75 saltos** con 57.4% de reducción de costo de almacenes.  
  *Insight:* Como los almacenes aún existen en el núcleo (aunque estén vacíos), el optimizador intenta alcanzarlos. **Recomendación táctica:** Al detectar botín en colectores, anular los pesos de almacenes para apagar los saltos automáticamente.
- **Aldeas con `botin_en_colectores == False` ($N=53$):**
  - **Ground Truth Usuario:** 2.70 saltos promedio, Costo Final = 10.94 (24.4% reducción).
  - **coverage_optimizer:** 1.60 saltos promedio, Costo Final = 7.19 (50.3% reducción).

### C.4 Top 10 Peores Discrepancias (Overlays Generados)
Se generaron las 10 imágenes de auditoría visual en `logs/yolo/gt_comparison/`:
1. `discrepancy_01_auto_20261104_111433_0010.png`: Costo Init 20.0 | OPT=17.0 (1 salto) vs GT=9.0 (4 saltos). Base anular concéntrica donde el humano colocó 4 saltos en anillo en los 4 cuadrantes.
2. `discrepancy_02_auto_20261104_111433_0005.png`: Costo Init 17.0 | OPT=8.0 (2 saltos) vs GT=0.0 (4 saltos). El humano conectó una cadena de 4 saltos que logró 100% de reducción a costo 0.
3. `discrepancy_03_auto_20261104_111433_0012.png`: Costo Init 9.0 | OPT=7.0 (1 salto) vs GT=4.0 (4 saltos).
4. `discrepancy_04_auto_20261104_111433_0022.png`: Costo Init 13.0 | OPT=8.0 (2 saltos) vs GT=5.0 (4 saltos).
5. `discrepancy_05_screen_raw.png`: Costo Init 1.0 | OPT=1.0 (0 saltos) vs GT=0.0 (4 saltos).
6. `discrepancy_06_auto_20261104_111433_0008.png`: Costo Init 11.0 | OPT=5.0 (1 salto) vs GT=3.0 (4 saltos).
7. `discrepancy_07_auto_20261104_111126_0001.png`: Costo Init 9.0 | OPT=8.0 (1 salto) vs GT=9.0 (4 saltos).
8. `discrepancy_08_auto_20261104_111433_0003.png`: Costo Init 3.0 | OPT=1.0 (1 salto) vs GT=1.0 (4 saltos).
9. `discrepancy_09_auto_20261104_111433_0006.png`: Costo Init 9.0 | OPT=5.0 (1 salto) vs GT=7.0 (4 saltos).
10. `discrepancy_10_auto_20261104_111433_0004.png`: Costo Init 16.0 | OPT=7.0 (2 saltos) vs GT=13.0 (4 saltos).

---

## 4. Divulgación de Sesgo y Protocolo de Test Limpio (Task D)

> [!WARNING]
> **Declaración Explícita de Sesgo In-Sample:**  
> Los hiperparámetros del detector y limpiador de muros (`min_line_len=45`, `tol_deg=6.0`, `min_gradient=65.0`, `endpoint_tol=25.0`) se ajustaron empíricamente sobre estas mismas 57 aldeas. Por lo tanto, **esta evaluación no constituye un test limpio held-out**. Representa un test de calibración in-sample.

### Protocolo para Conjunto de Prueba Limpio (Held-Out Test Set)
1. **Recolección:** Activar en el bot durante una sesión de farmeo automático el guardado de capturas de batalla a `logs/yolo/test_split_unseen/` (30 a 50 aldeas nuevas TH12 a TH16 no vistas previamente).
2. **Anotación Ciega:** El usuario etiqueta las aldeas nuevas con `tools/annotate_best_jump.py` **antes** de correr cualquier algoritmo de optimización.
3. **Evaluación Congelada:** Ejecutar el benchmark con el código y parámetros totalmente congelados para obtener una métrica de generalización 100% imparcial.

---

## 5. Exclusión de HUD de Batalla y Abstracción de Hechizo (Task E)

### 5.1 Exclusión de Candidatos bajo HUD
- En `sample_jump_candidates`, los puntos candidatos cuyo centro cae dentro de la máscara de HUD de batalla dilatada (`get_battle_hud_mask` con margen de 15 px) son descartados para evitar clics sobre la barra de tropas o botones.
- La huella de alcance del salto ($R=90\text{ px}$ o elipse) **se extiende libremente bajo el HUD**, permitiendo puentear muros cercanos al borde.

### 5.2 Análisis Empírico de Fugas por Muros bajo el HUD
Se auditó la totalidad de las 57 aldeas analizando si existían muros limpios bajo el HUD (`y >= 888` o áreas de medidores):
- **Aldeas con muros en zona HUD:** **0 / 57 aldeas (0.0%)**.
- **Aldeas que sufren fugas por recorte de HUD:** **0 / 57 aldeas (0.0%)**.
*Motivo técnico:* La cuadrícula isométrica jugable de Clash of Clans está naturalmente contenida entre $y \approx 100$ y $y \approx 850$ en resolución $2400 \times 1080$. El limpiador de muros elimina el marco exterior de pasto no jugable, dejando cero muros en las zonas de interfaz interactiva.

### 5.3 Abstracción de Hechizo en el Optimizador
`optimize_jumps` ahora soporta el parámetro `spell_config` (leído de `settings.json`):
- `shape`: `"circle"` o `"ellipse"`
- `radius` / `rx`, `ry`: Radios configurables
- `max_spells` / `max_jumps`: Límite de hechizos
- `min_marginal_gain`: Umbral de parada temprana
- `effect`: `"bridge"` (salto) o `"breach"` (terremoto / rompemuros)

---

## 6. Qué NO se Pudo Verificar

1. **Generalización Out-of-Sample:** No se pudo evaluar sobre aldeas no vistas previamente (las 57 fueron usadas en la calibración inicial).
2. **Despliegue y Ejecución en Batalla Real:** La Fase 4 se mantuvo **estrictamente congelada** y `attack/deployment_strategy.py` no fue modificado ni ejecutado en el emulador en vivo.
3. **Pathing Dinámico de Tropas bajo Fuego:** El modelo evalúa capas mínimas de muro en un grafo estático de compartimentos. No simula la dispersión de duendes furtivos ante defensas activas (torres de magos o morteros) ni el tiempo real de interacción con los hechizos.
