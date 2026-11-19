# MEJORA 02 — Rendimiento de Visión, ONNX, OCR y Caché de Muros

## Objetivo
Corregir la colisión silenciosa de caché en `wall_raycast.py`, eliminar la duplicación de la máscara de HUD, optimizar `DeployZoneDetector` (de cientos de ms por aldea a `< 1 ms` por consulta), acelerar `ResourceReader` cacheando la escala ganadora y alinear `onnxruntime.SessionOptions` en todos los modelos.

## Alcance (Archivos a crear/modificar)
- **Nuevo:** `vision/hud.py`
- `vision/wall_raycast.py`
- `vision/deploy_zone_detector.py`
- `vision/yolo_detector.py`
- `vision/resource_reader.py`
- `vision/wall_compartments.py` (solo importar `get_battle_hud_mask` desde `vision/hud.py`)
- `vision/wall_seg_detector.py` (solo importar `get_battle_hud_mask` desde `vision/hud.py` y usar guardado asíncrono)
- Nuevo test: `tests/test_vision_performance.py`

## Restricciones (NO tocar)
- **NO modificar `attack/deployment_strategy.py`**.
- Mantener 100% compatibilidad de firmas públicas y re-exportar `get_battle_hud_mask` en `vision/wall_compartments.py` y `vision/wall_seg_detector.py` para no romper imports existentes.

---

## Tareas Técnicas Detalladas

### 1. Centralizar `get_battle_hud_mask()` en `vision/hud.py`
**Problema actual:**
La función `get_battle_hud_mask(h, w, margin_px=0)` está copiada casi idéntica en `vision/wall_compartments.py`, `vision/wall_seg_detector.py` y `attack/deployment_strategy.py`.

**Solución requerida:**
1. Crea `vision/hud.py` con una única implementación limpia de `get_battle_hud_mask(h: int, w: int, margin_px: int = 0) -> np.ndarray` con caché LRU (`@functools.lru_cache(maxsize=16)`) devolviendo una copia read-only o `mask.copy()` para evitar recalcular la misma matriz `2400x1080` en cada llamada.
2. En `vision/wall_compartments.py` y `vision/wall_seg_detector.py`, elimina la copia local e importa `from vision.hud import get_battle_hud_mask`.

---

### 2. Corregir colisión de hash en `vision/wall_raycast.py`
**Problema actual:**
`detect_global_wall_segments()` calcula:
```python
screen_hash = hash(img.tobytes()[:2048])
```
En `2400x1080x3`, los primeros 2048 bytes corresponden solo a los primeros ~682 píxeles de la fila 0 (borde superior oscuro/bosque idéntico entre muchas aldeas). Esto provoca falsos hits de caché y reutiliza segmentos de muro de una aldea anterior.

**Solución requerida:**
1. Implementa un helper `_compute_frame_fingerprint(img: np.ndarray) -> int`:
   - Recorta la región central de la aldea (`y: 15%..85%`, `x: 15%..85%`).
   - Reduce con `cv2.resize(..., (32, 16), interpolation=cv2.INTER_AREA)`.
   - Usa `zlib.adler32(thumb.tobytes())` (determinista, rapidísimo, `<0.1 ms` y sensible a cualquier cambio de aldea).
2. En `draw_wall_raycast_debug()`, sustituye `cv2.imwrite(out_path, canvas)` por `save_debug_image_async(out_path, canvas)` desde `core.vision`.

---

### 3. Acelerar `find_closest_deployable_point()` en `vision/deploy_zone_detector.py`
**Problema actual:**
Cada llamada a `find_closest_deployable_point()` (que se ejecuta **una vez por cada edificio detectado**, ~15–20 veces por aldea) recalcula `cv2.distanceTransform` sobre la imagen completa `2400x1080` y luego hace `np.where(safe_deployable == 1)` sobre ~1.5 millones de píxeles.

**Solución requerida:**
1. En `DeployZoneDetector.__init__()` inicializa:
   - `self._last_dist_map: Optional[np.ndarray] = None`
   - `self._boundary_pts_cache: dict[tuple, np.ndarray] = {}`
2. Al actualizar `self.last_mask` en `predict_mask()`:
   - Invalida `self._boundary_pts_cache.clear()`.
   - Precalcula `self._last_dist_map = cv2.distanceTransform((binary_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)`.
3. En `find_closest_deployable_point()`:
   - Si `mask is None` (o `mask is self.last_mask`), reutiliza `self._last_dist_map` (si es `None`, calcúlalo una vez y guárdalo).
   - Para la clave `(margin_px, safe_bounds)`, busca en `self._boundary_pts_cache`. Si no existe:
     - Construye `safe_deployable = (dist_map >= margin_px).astype(np.uint8)` aplicando `safe_bounds`.
     - Extrae **solo los píxeles de borde/contorno** con `cv2.findContours(safe_deployable, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)` y apila todos los puntos `(N, 2)` en un `np.ndarray` (`int32`).
     - *(Razón geométrica: para cualquier punto no desplegable, el punto desplegable más cercano siempre pertenece a la frontera de `safe_deployable`.)*
     - Guarda ese array (~2,000–4,000 puntos en vez de 1.5M) en `self._boundary_pts_cache`.
   - Calcula `dists_sq` únicamente contra ese array compacto de puntos de frontera.
4. Asegura compatibilidad con tests que instancian `DeployZoneDetector.__new__(DeployZoneDetector)` sin llamar `__init__` (usa `getattr(self, "_boundary_pts_cache", None)`).

---

### 4. Optimizar sesión ONNX en `vision/yolo_detector.py`
**Problema actual:**
`YoloDetector._init_model()` crea `ort.InferenceSession(resolved_path, providers=["CPUExecutionProvider"])` sin configurar `SessionOptions`, mientras que `DeployZoneDetector` y `WallSegDetector` sí limitan hilos. En CPU, la sobre-suscripción de hilos degrada la latencia cuando conviven 3 sesiones ONNX.

**Solución requerida:**
1. Configura `ort.SessionOptions()` en `YoloDetector._init_model()`:
   ```python
   opts = ort.SessionOptions()
   opts.intra_op_num_threads = min(4, os.cpu_count() or 2)
   opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
   ```
2. Aplica también `opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL` en `DeployZoneDetector` y `WallSegDetector`.

---

### 5. Cachear escala ganadora y limitar dumps en `vision/resource_reader.py`
**Problema actual:**
1. `_candidates()` corre `cv2.matchTemplate` contra `8 escalas x 10 dígitos = 80 templates` por cada ROI (240 llamadas en cada evaluación de botín). En una resolución fija (`2400x1080`), la fuente siempre coincide en la misma escala.
2. `_dump_fail()` y `read_resource_tooltip()` llaman a `cv2.imwrite()` de forma síncrona, y `logs/fails/` crece sin límite.

**Solución requerida:**
1. Reorganiza el banco de templates en `self._bank_by_scale: dict[float, list[tuple[int, np.ndarray]]]` manteniendo `self._get_bank()` compatible si algún test lo llama.
2. Añade `self._preferred_scale: dict[str, float] = {}` por categoría (`"enemy"`, `"own"`, `"tooltip"`).
3. En `_read_number_from_roi(roi_gray, key)`:
   - Si existe una escala preferida para la categoría de `key`, prueba **primero solo esa escala** (`10` templates en vez de `80`) con umbral `>= 0.74`.
   - Si `_valid_grouping(dets)` pasa, retorna al instante.
   - Si falla, recorre todas las escalas como fallback y actualiza `self._preferred_scale` con la escala que aportó los mejores matches.
4. En `_dump_fail()` y `read_resource_tooltip()`: usa `save_debug_image_async` de `core.vision` y añade una rotación simple en `_dump_fail()` para no superar 200 archivos en `logs/fails/`.

---

## Criterios de Aceptación y Verificación

1. Crea `tests/test_vision_performance.py` validando:
   - `detect_global_wall_segments()` con dos imágenes distintas que comparten exactamente las primeras 10 filas negras produce hashes distintos y no devuelve caché de la primera.
   - `DeployZoneDetector.find_closest_deployable_point()` devuelve un punto legal con distancia idéntica (tolerancia `<= 2 px`) y ejecuta 20 consultas sobre una máscara `2400x1080` en **< 100 ms totales** (excluyendo el primer `predict_mask`).
   - `ResourceReader` acierta usando la escala preferida tras la primera lectura válida.
2. Ejecuta:
```powershell
python -m unittest tests.test_vision_performance tests.test_deploy_zone_detector tests.test_drop_validation tests.test_wall_raycast tests.test_damage_reader tests.test_storage_trigger -v
```
Todos los tests deben pasar sin errores.