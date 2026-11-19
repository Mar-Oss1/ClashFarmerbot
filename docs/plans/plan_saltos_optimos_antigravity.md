# Plan: colocación óptima de hechizos de salto (ClashFarmerbot)

> Para Antigravity. Léelo completo antes de tocar código. Trabaja por FASES y respeta los **PUNTOS DE PARADA**: en cada uno te detienes, me entregas el reporte y esperas mi revisión antes de seguir.

## 0. Contexto y objetivo

Hoy el bot decide los saltos con `jump_placement = "wall_raycast"`: para cada objetivo traza un rayo desde el punto de drop hasta el almacén y busca cruces con los segmentos de muro (`vision/wall_raycast.py`: `detect_global_wall_segments`, `find_wall_crossings_from_segments`, `calculate_adaptive_jump_and_breach_placements`; y en `attack/deployment_strategy.py`: `compute_adaptive_jump_targets_from_detections`, `_compute_wall_raycast_jump_targets`, `_compute_cluster_intersection_jump_targets`, `_compute_legacy_quadrant_jump_targets`).

**Problema:** es un cálculo por objetivo, no global. Cada rayo se resuelve por separado, así que el bot gasta muchos saltos (más Super Rompemuros) en aperturas que un humano resuelve con uno solo, colocándolo donde el círculo cruza varias capas de muro a la vez.

**Objetivo:** un optimizador que modele la aldea como regiones cerradas por muros (compartimentos), calcule cuántas capas de muro separan la entrada de cada objetivo y elija los puntos de salto que más reducen ese costo con el menor número de hechizos.

**Alcance:** SOLO saltos. Aceleración (`haste_per_breach`), Super Rompemuros y tiempos de lanzamiento quedan igual, salvo exportar las huellas de los saltos para que `has_jump_coverage(pt)` siga funcionando.

## 1. Reglas generales

1. Crea una rama o un respaldo antes de empezar. No toques `_backups/`.
2. Modo nuevo detrás de flag: `attack.jump_placement = "coverage_optimizer"`. El default actual (`"wall_raycast"`) **no cambia** hasta que yo lo apruebe. Los otros modos siguen intactos.
3. Todo parámetro nuevo en `config/settings.json` bajo `attack.jump_optimizer` con default conservador.
4. Compatibilidad: firmas existentes sin cambios; `python -m unittest` debe seguir en verde, o dime qué falla y por qué.
5. Logs con `core.logger.get_logger`, en español con el estilo actual. Nada de `print`.
6. Sin dependencias pesadas nuevas (usa numpy/OpenCV, que ya están).
7. Todo fallo del optimizador (excepción, timeout, sin compartimentos) hace fallback a `wall_raycast` con `log.warning` y motivo. Nunca silencioso.
8. No cambies `jump_spell_radius_px` ni otros umbrales actuales para "arreglar" resultados sin datos que lo respalden.
9. Rutas de Windows (`C:/ClashFarmerbot`). Usa `os.path.join`/`PROJECT_ROOT`. No subas capturas ni modelos a git.

---

## FASE 1: Ground truth y calibración

### 1.1 Herramienta de anotación `tools/annotate_best_jump.py`

Para cada imagen de `datasets/coc/images/raw` (y las capturas de ataques guardadas que existan), muestra la captura con los segmentos de muro superpuestos y deja que **yo** haga clic en los mejores puntos de salto, en orden (hasta 3), o marque "no hace falta salto". Guarda `datasets/coc/jump_gt/<imagen>.json` con `{puntos: [[x, y], ...], sin_salto: bool, notas: str}`.

Debe permitir saltar, deshacer y reanudar donde se quedó. Objetivo: anotar al menos 30 aldeas.

### 1.2 Calibración de la huella del salto

Verifica de dónde sale el valor actual: `settings.json` tiene `jump_spell_radius_px = 90`, pero `_save_yolo_attack_debug` usa 125 como default si no encuentra la clave. Confirma cuál se usa realmente en cada lugar.

Hipótesis a verificar con datos (no asumas): en vista isométrica la huella real del hechizo es una **elipse** (más ancha que alta), no un círculo de radio fijo en pantalla. Con `estimate_iso_angles` / `detect_global_wall_segments` ya tienes los ángulos isométricos.

- Revisa capturas reales donde el anillo del hechizo sea visible (frames justo después de soltar un salto en ataques guardados o capturados en vivo) y mide ancho y alto del anillo en píxeles.
- Propón `jump_footprint = {rx_px, ry_px}` (o radio en suelo + proyección) y compáralo con 90/125.
- Entrega la medición con imágenes de evidencia. Si no hay frames utilizables, dime cuáles capturar y cómo (dejaré que tú me pidas una prueba en vivo).

### 1.3 Más capturas (opcional)

Las 37 aldeas actuales bastan para empezar a validar. Verifica que `tools/auto_capture_villages.py` guarde automáticamente cada aldea nueva durante ataques normales, sin duplicados, y reporta cuántas tenemos y qué diversidad hay (nivel de ayuntamiento, número de capas de muro, edificios fuera del muro). Meta de largo plazo: unas 100, pero **la diversidad importa más que la cantidad**.

### 🛑 PUNTO DE PARADA 1

Entrégame: cuántas aldeas anotadas tengo pendientes de hacer, la medición de la huella (90 vs 125 vs elipse) y el estado de las capturas. Yo hago la anotación y respondo.

---

## FASE 2: Modelo de compartimentos (solo análisis, sin tocar el ataque)

Crea `vision/wall_compartments.py` con funciones puras y testeables:

1. **Entrada:** captura, segmentos de muro (`detect_global_wall_segments`), detecciones YOLO, ángulos isométricos, huella del salto.
2. **Máscara de muros:** dibuja los segmentos con grosor configurable. Los muros tienen huecos donde hay edificios encima, así que prolonga los extremos a lo largo de la dirección isométrica (usa `endpoint_tol` como referencia) y aplica cierre morfológico. Parámetros en `attack.jump_optimizer`.
3. **Barreras:** los bbox de edificios YOLO cuentan como barreras de paso (configurable `buildings_block: true`), porque los muros "con fugas" son el error más probable.
4. **Plano de suelo:** transforma píxeles a coordenadas de suelo con los ángulos isométricos (donde el círculo del hechizo es un círculo de verdad) y trabaja en una rejilla reducida (por ejemplo 1 celda = 6-8 px) por velocidad. Vuelve a coordenadas de pantalla al final.
5. **Compartimentos:** componentes conexos del espacio libre. El exterior es el componente conectado al borde/zona de drop legal (usa la máscara del `DeployZoneDetector` si está lista).
6. **Grafo:** nodos = compartimentos; arista entre dos compartimentos si comparten una banda de muro; costo de arista = 1 capa. Entrada = compartimentos que contienen los puntos de drop previstos.
7. **Objetivos:** estructuras YOLO de botín con los pesos de `jump_loot_weights`; cada una se asigna a su compartimento (o al libre más cercano si el centro cae en una barrera).
8. **Costo total:** Σ peso × (capas mínimas entrada → compartimento del objetivo).
9. **Efecto de un salto en `p`:** todos los compartimentos que intersecan la huella en `p` quedan conectados entre sí (costo 0 dentro de la huella). Recalcula el costo.
10. **Optimizador:** candidatos muestreados sobre las celdas de muro dentro del bbox de la aldea. Selección voraz de hasta `max_jumps` saltos, cada uno calculado sobre el mapa ya abierto por los anteriores; detente cuando la ganancia marginal baje de un umbral. Variante opcional con beam search (ancho 3-5) para evitar trampas del voraz. Presupuesto: menos de 300 ms por aldea; si se excede, devuelve lo mejor encontrado y registra el aviso.
11. **Salida:** lista `[(x, y, ganancia)]` en píxeles de pantalla, el costo antes/después de cada salto y un overlay de depuración: compartimentos en colores, aristas, huellas y la ganancia de cada salto.

Tests unitarios (`tests/test_wall_compartments.py`, `unittest`, sin ONNX) con mallas sintéticas:

- Dos anillos concéntricos de muro con un objetivo en el centro: un salto bien puesto abre ambos anillos (ganancia 2).
- Un anillo con hueco tapado por edificio: el cierre/prolongación evita la fuga o el test documenta el límite.
- Aldea sin muros: devuelve 0 saltos.
- Dos objetivos en compartimentos distintos: la selección voraz no repite la apertura anterior.

---

## FASE 3: Línea base y evaluación offline

Con el modelo de la Fase 2, sobre todas las aldeas (y las anotadas aparte):

1. **Línea base:** corre los tres modos actuales (`wall_raycast`, `cluster_intersection`, `legacy`) y mide con el mismo modelo de compartimentos: saltos usados, costo antes/después, objetivos que quedan cerrados.
2. **Optimizador nuevo:** las mismas métricas.
3. **Contra mi ground truth** (en las aldeas anotadas): fracción de mis puntos cuya huella se solapa con la de un salto del optimizador (indica cuánto "ve" como yo), y el costo residual con mis saltos frente a los suyos.
4. **Sensibilidad:** barrido de `buildings_block`, grosor de muro, cierre morfológico y huella (90, 125, elipse medida). Reporta cómo cambian las métricas.
5. **Análisis de fallos:** para las 10 peores aldeas, imagen con compartimentos y la causa probable (fuga de muro, mal contorno, objetivo mal asignado, muro mal detectado).

Entrega un `summary.md` con tablas por modo y las imágenes. No inventes umbrales de éxito: propón unos según lo que veas y los acordamos.

### 🛑 PUNTO DE PARADA 2

Entrégame el resumen, las 10 peores imágenes y las 10 mejores. **No integres al ataque hasta que yo apruebe.** Si el modelo de compartimentos "fuga" mucho (muros mal cerrados), lo corregimos aquí antes de seguir; la calidad del mapa de muros es el techo de todo el cálculo.

---

## FASE 4: Integración detrás del flag

Solo tras mi aprobación:

1. En `compute_adaptive_jump_targets_from_detections` añade el modo `"coverage_optimizer"`. Si falla o excede el presupuesto de tiempo, fallback a `wall_raycast` con `log.warning` y motivo.
2. Respeta el formato de retorno actual de los saltos (el resto del flujo y `_save_yolo_attack_debug` los consumen tal como están).
3. Exporta las huellas elegidas para que `has_jump_coverage(pt)` (usada por la lógica de Super Rompemuros en `_execute_single_zone_quota_deploy`) considere cubiertas las aperturas del optimizador.
4. No cambies el momento ni el orden de los lanzamientos (los saltos se lanzan junto con los drops por breach; déjalo igual).
5. Guarda por ataque en `logs/yolo/attack_<timestamp>/` el overlay de compartimentos, `jump_decision.json` (ya existe: amplíalo con costo antes/después y ganancia por salto) y la comparación con lo que habría elegido el modo anterior.
6. Tests: `python -m unittest` completo.

---

## FASE 5: Dry-run en vivo (sin soltar hechizos)

Agrega `--dry-run-jumps` (a `main.py` o a `tools/run_live_test_3_attacks.py`): en una batalla real, captura, corre ambos modos y guarda los paquetes de la sección 4.5 **sin soltar ningún hechizo de salto ni tropas**. Termina rindiéndote por el flujo normal del bot (`allow_surrender`). **No reinicies la app** (el juego impone un temporizador de ~1.5 min si se sale en lugar de terminar la batalla).

Cada búsqueda de partida cuesta oro: haz 3 dry-runs y no más.

### 🛑 PUNTO DE PARADA 3

Entrégame los 3 paquetes con tu lectura. Yo reviso los overlays y decido si pasamos a ataques reales.

---

## FASE 6: Ataques reales supervisados (A/B)

Con mi aprobación, alterna modo anterior y nuevo en ataques consecutivos (impar = actual, par = `coverage_optimizer`), mínimo 10 ataques, con logging completo. Registra por ataque:

- Saltos usados y Super Rompemuros usados.
- Botín obtenido y estructuras alcanzadas (con lo que ya registra el bot).
- Captura antes y después de lanzar los saltos para ver dónde aparece el anillo realmente y si coincide con la huella estimada.
- Cualquier fallback o timeout del optimizador.

Si ocurre algo anormal (salto lanzado fuera de la aldea, fallback repetido, tropas que no cruzan donde se esperaba), **detén la secuencia** y repórtalo.

### Reporte final `logs/findings/jump_optimizer_report.md`

Tabla A/B por ataque, resumen de saltos y botín por modo, imágenes más reveladoras, hipótesis confirmadas o descartadas, riesgos abiertos y lo que NO pudiste verificar. Ten presente que con 10 ataques hay mucha varianza: no saques conclusiones estadísticas fuertes, di qué se ve y qué haría falta.

### 🛑 PUNTO DE PARADA 4 (final)

Entrégame el reporte y espera. Revisamos juntos antes de cambiar el default de `jump_placement`.

---

## Criterios de éxito (a ajustar tras el Punto de Parada 2)

- En offline: con el mismo modelo de compartimentos, el optimizador logra un costo igual o menor que `wall_raycast` usando igual o menos saltos.
- Sin fallbacks silenciosos y tiempo por aldea menor a 300 ms.
- En vivo: igual o más botín con menos saltos/Super Rompemuros por ataque que el modo actual (visto en A/B).
- Tests existentes en verde y los nuevos pasando.

## Qué NO hacer

- No cambiar `jump_spell_radius_px` ni umbrales para forzar resultados sin evidencia.
- No tocar aceleración, Super Rompemuros ni el momento de lanzamiento.
- No reentrenar modelos ni cambiar la detección de muros en este plan. Si el análisis de fallos muestra que el mapa de muros es el cuello de botella, lo anotas como recomendación y lo discutimos después.
- No avanzar de fase sin pasar por el Punto de Parada correspondiente.
