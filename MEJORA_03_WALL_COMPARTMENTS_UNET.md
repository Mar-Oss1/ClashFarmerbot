# MEJORA 03 — Fixes y Aceleración del Optimizador U-Net + Beam Search

## Objetivo
Corregir los bugs pendientes en el motor de compartimentos (`vision/wall_compartments.py`) y en el detector de muros U-Net (`vision/wall_seg_detector.py`), además de reducir drásticamente el tiempo de cómputo del Beam Search sin sacrificar calidad de solución.

## Alcance (Archivos a modificar)
- `vision/wall_seg_detector.py`
- `vision/wall_compartments.py`
- `tests/test_wall_compartments.py`
- `tests/test_wall_unet_pipeline.py`

## Restricciones (NO tocar)
- **NO modificar `attack/deployment_strategy.py`**.
- Mantener el presupuesto de ejecución `< 300 ms` por aldea en `1080p`.
- Mantener 100% compatibilidad de firmas con `tools/evaluate_jumps_offline.py` y `tools/analyze_eval_extremes.py`.

---

## Tareas Técnicas Detalladas

### 1. Leer configuración real desde `settings.json` en `vision/wall_seg_detector.py`
**Problema actual:**
En `config/settings.json`, tienes configurado:
```json
"jump_optimizer": {
  "wall_seg_model_path": "models/wall_seg.onnx",
  "wall_seg_conf": 0.3
}
```
Sin embargo, `get_wall_seg_detector(model_path="models/wall_seg.onnx")` ignora `settings.json` e instancia `WallSegDetector` siempre con `conf_thresh=0.50` por defecto. Además, `save_debug_overlay()` usa `cv2.imwrite()` síncrono.

**Solución requerida:**
1. En `get_wall_seg_detector(model_path: str | None = None, conf_thresh: float | None = None)`:
   - Lee `cfg = SETTINGS.get("attack", {}).get("jump_optimizer", {})`.
   - Resuelve `model_path = model_path or cfg.get("wall_seg_model_path", "models/wall_seg.onnx")`.
   - Resuelve `conf_thresh = conf_thresh if conf_thresh is not None else float(cfg.get("wall_seg_conf", 0.50))`.
2. Cambia `save_debug_overlay()` para usar `save_debug_image_async` de `core.vision` (manteniendo `sync=True` automático cuando `"unittest" in sys.modules`, como ya hace `save_debug_image_async`).

---

### 2. Corregir bug de escala en `render_debug_overlay()` de `vision/wall_compartments.py`
**Problema actual:**
Cuando la imagen de entrada es HD (`w >= 1200`), `optimize_jumps()` usa `scale = 2`, por lo que `comp_labels` y `scaled_wall_mask` tienen resolución `1200x540`. Sin embargo, `selected_jumps` y `jump_footprint` se pasan a `render_debug_overlay()` en coordenadas originales (`2400x1080`), mientras `render_debug_overlay()` redimensiona `canvas` al tamaño de `comp_labels` (`1200x540`). Resultado: los círculos/elipses de salto se dibujan en `(2x, 2y)` con el doble de radio (desplazados o fuera del lienzo).

**Solución requerida:**
1. Agrega el parámetro `grid_scale: int = 1` a `render_debug_overlay(...)`.
2. Al dibujar cada `jump` en `selected_jumps`:
   - `jx = int(round(jump.x / grid_scale))`
   - `jy = int(round(jump.y / grid_scale))`
   - Divide `rx` y `ry` entre `grid_scale` antes de llamar a `cv2.ellipse` / `cv2.circle`.
3. Pasa `grid_scale=scale` desde `optimize_jumps()` al invocar `render_debug_overlay(...)`.

---

### 3. Poda por equivalencia topológica antes del Beam Search en `vision/wall_compartments.py`
**Problema actual:**
`valid_candidates` puede contener hasta `200` puntos distintos que tocan exactamente el **mismo subconjunto de compartimentos** `frozenset(touched_comps)`. En un Beam Search de profundidad 4 con `beam_width=4`, evaluar todos esos candidatos redundantes dispara miles de `compute_total_cost()` (0-1 BFS en Python) y frecuentemente agota el `loop_deadline` (`270 ms`) antes de completar los niveles 3 y 4.

**Solución requerida:**
1. Crea un helper `_count_wall_pixels_in_core(pt, wall_mask, core_radius) -> int` y haz que `is_valid_wall_centered_jump()` lo reutilice (evitando código duplicado).
2. Al construir `valid_candidates` en `optimize_jumps()`:
   - Agrupa los candidatos válidos por `comp_key = frozenset(touched_comps)`.
   - Para cada `comp_key` único, conserva únicamente los **top 1–2 representantes** mejor posicionados:
     - Prioridad 1: mayor cantidad de píxeles de muro en su núcleo (`core_radius`), asegurando que el salto caiga bien centrado sobre los muros.
     - *(Conserva hasta 2 representantes por `comp_key` solo si están separados más de `min_sep_px`, para no perder combinaciones válidas por restricción de separación mínima).*
3. Añade una caché por ejecución `cost_cache: dict[tuple, tuple[float, dict[int, float]]]` dentro de `optimize_jumps()` indexada por `tuple(sorted(frozenset(b) for b in test_bridges))` para no recomputar el mismo 0-1 BFS desde ramas distintas del haz.
4. Registra en `log.debug` cuántos candidatos se podaron (`raw -> deduplicados`) y cuántos estados se evaluaron.

---

### 4. Validar que `min_wall_pixels` escale bien con máscaras delgadas
**Problema actual:**
En `tests` o en aldeas donde la U-Net devuelve muros finos (`~3-4 px`), el filtro `min_wall_pixels=30` funciona bien en `2400x1080`, pero si llega una imagen más pequeña (`720p` o sintética), un umbral fijo en píxeles absolutos puede descartar candidatos válidos.

**Solución requerida:**
- Si `w < 2000` y no se pasó `min_wall_pixels` explícitamente en la llamada, escala `effective_min_wall_pixels` proporcionalmente al factor de área `(w / 2400.0)` (con mínimo de `10`), manteniendo exacto el valor `30` para `2400x1080` y respetando el valor explícito en tests unitarios.

---

## Criterios de Aceptación y Verificación

1. Agrega en `tests/test_wall_compartments.py`:
   - Un test que valide `render_debug_overlay()` con `img_shape=(1080, 2400)` (`grid_scale=2`): verificar que los píxeles del marcador rojo `(0, 0, 255)` del salto caen en `(jx // 2, jy // 2)` y no fuera del canvas.
   - Un test de rendimiento en `1080p` con 40 segmentos de muro y `beam_width=4, max_jumps=4` asegurando que completa **todos los niveles sin agotar `time_budget_sec=0.300`**.
2. Agrega en `tests/test_wall_unet_pipeline.py`:
   - Un test verificando que `get_wall_seg_detector()` lee `wall_seg_conf` desde `SETTINGS["attack"]["jump_optimizer"]`.
3. Ejecuta:
```powershell
python -m unittest tests.test_wall_compartments tests.test_wall_unet_pipeline tests.test_wall_raycast tests.test_wall_junctions -v
```
Todos los tests deben pasar al 100%.