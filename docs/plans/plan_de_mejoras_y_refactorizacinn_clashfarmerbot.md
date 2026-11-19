# Plan de Refactorización y Mejoras Tácticas — ClashFarmerbot

> **Instrucciones para el Agente (Antigravity):**
> Aplica estas mejoras de forma **modular y secuencial** (Fase por Fase). 
> **REGLA DE ORO:** No rompas ninguno de los contratos verificados por la suite de pruebas en `tests/`. Si un método es llamado por un `MagicMock` en los tests (por ejemplo, `self.device` en `MockDevice`), usa comprobaciones seguras como `hasattr(self.device, ...)` y mantén los valores por defecto en las firmas de las funciones. Al terminar cada fase, ejecuta `python -m unittest discover -s tests` para garantizar $100\%$ de tests en verde.

---

## Fase 1: Correcciones Críticas en `attack/deployment_strategy.py` y `config/settings.json`

### 1.1. Eliminar código muerto en `_survey_and_analyze_village`
* **Archivo:** `attack/deployment_strategy.py`
* **Problema:** Al final de `_survey_and_analyze_village()` existe un doble `return` consecutivo donde la segunda línea (`return screen, detections, green_hull`) es inalcanzable y referencia variables que no existen en ese scope.
* **Acción:** Eliminar la línea duplicada `return screen, detections, green_hull` dejando únicamente:
  ```python
  return screen_center, center_detections, green_hull
  ```

### 1.2. Corte temprano de paneos cuando se agotan los duendes (`_deploy_by_structure_quotas`)
* **Archivo:** `attack/deployment_strategy.py` y `config/settings.json`
* **Problema:** Si en la Fase 1 (`CENTRO`) se agotan los $96$ duendes (`self._troop_inventory["goblins"] <= 0`), el bot sigue ejecutando los paneos hacia Norte, Sur, Este y Oeste sin tropas para lanzar, perdiendo ~10 segundos.
* **Acción:**
  1. En `config/settings.json`, cambiar `"enable_full_cross_pan": false` para que en producción solo panee cuando existan objetivos `unreached` pendientes.
  2. En `_deploy_by_structure_quotas` (`attack/deployment_strategy.py`), antes de ejecutar cada bloque de flanco (`NORTE`, `SUR`, `ESTE`, `OESTE`), verificar si aún quedan duendes disponibles:
     ```python
     if self._troop_inventory.get("goblins", 0) <= 0:
         log.info("🛑 Inventario de duendes agotado (0 restantes). Omitiendo paneos restantes.")
     ```
     *(Nota para tests: Mantener el soporte de `enable_full_cross_pan=True` cuando `self._troop_inventory["goblins"] > 0` para que `tests/test_heroes_haste_and_cross_pan.py` y `tests/test_quota_drop_and_inventory.py` sigan pasando).*

### 1.3. Tiempo de viaje (Lead Time) para Súper Rompemuros (SWB) antes de soltar Duendes
* **Archivo:** `attack/deployment_strategy.py` (`_execute_single_zone_quota_deploy`)
* **Problema:** En el paso `5.5` se lanzan los SWB, pero si no se despliegan héroes ni hechizos de aceleración en esa vista, el código salta inmediatamente al paso `6` (duendes) en $< 0.15\text{ s}$. Como el SWB tarda ~1.0s en caminar desde el césped hasta el muro exterior y detonar, los primeros duendes chocan contra el muro cerrado o se desvían.
* **Acción:**
  1. Registrar el instante en que termina de lanzarse el último SWB en el paso `5.5`:
     ```python
     t_last_swb = time.monotonic()
     ```
  2. Justo antes de iniciar el bucle de despliegue de duendes en el paso `6`, si se desplegaron SWB en esta zona (`t_last_swb is not None`) y no estamos en un entorno de test mockeado (verificar que `not isinstance(self.device.d, MagicMock)` o usar una constante configurable `swb_travel_lead_sec = 0.85`), esperar el tiempo restante:
     ```python
     if t_last_swb is not None:
         elapsed_swb = time.monotonic() - t_last_swb
         lead_wait = float(SETTINGS.get("attack", {}).get("swb_travel_lead_sec", 0.85))
         if elapsed_swb < lead_wait and self.device.__class__.__name__ == "Device":
             time.sleep(lead_wait - elapsed_swb)
     ```

### 1.4. Coherencia Táctica con el Ayuntamiento (`town_hall`) y Cadencia de Toques
* **Archivo:** `attack/deployment_strategy.py`
* **Problema:** En `settings.json`, `"town_hall": 0` en `jump_loot_weights` hace que no se gasten saltos exclusivamente para el Ayuntamiento, pero en `_GOBLIN_QUOTA` tiene cuota `6` y prioridad `#2`. Si el TH está en el centro encerrado sin salto, se desperdician 6 duendes contra los muros; y si sí está abierto, 6 duendes suelen ser insuficientes para destruir un TH de nivel alto antes de perder la invisibilidad.
* **Acción:**
  1. Subir `_GOBLIN_QUOTA["town_hall"]` de `6` a `10` (sin alterar los tests de `test_quota_drop_and_inventory.py`, los cuales prueban con `storage_gold`, `storage_elixir` y `storage_dark` que mantienen cuota `6`).
  2. En `target_openness_rank(item)` dentro de `_execute_single_zone_quota_deploy`, evaluar un `town_hall` como `is_open = True` únicamente si tiene un salto cercano ($< 150\text{ px}$) o si está cerca del borde exterior ($< 160\text{ px}$ de su punto de drop). Si el TH está en el núcleo profundo sin salto asignado, dejarlo en `Rank 1` (o filtrarlo si no hay cobertura de salto) para no desperdiciar duendes contra muros cerrados.
  3. En el bucle de toques de cuota de duendes:
     ```python
     for idx_q in range(quota):
         jx, jy = HumanBehavior.jitter_point(drop_x, drop_y, max_offset=4)
         jx = int(np.clip(jx, self._SAFE_X_MIN, self._SAFE_X_MAX))
         jy = int(np.clip(jy, safe_y_min, safe_y_max))
         self.device.d.click(jx, jy)
         self._troop_inventory["goblins"] -= 1
         total_deployed += 1
         if idx_q < quota - 1 and self.device.__class__.__name__ == "Device":
             HumanBehavior.burst_tap_delay()
     ```

### 1.5. Estabilización inteligente de cámara tras paneo (`wait_camera_settled`)
* **Archivo:** `attack/deployment_strategy.py`
* **Problema:** Tras cada `input swipe` en `_deploy_by_structure_quotas` y `_survey_and_analyze_village`, se usa un `time.sleep(1.30)` fijo en vez de aprovechar `self.device.wait_camera_settled()`.
* **Acción:** Crear un método auxiliar `_wait_after_pan(self, fallback_sec: float = 0.85)` en `DeploymentStrategy`:
  ```python
  def _wait_after_pan(self, fallback_sec: float = 0.85):
      if self.device.__class__.__name__ == "Device" and hasattr(self.device, "wait_camera_settled"):
          self.device.wait_camera_settled(timeout=1.5)
      else:
          time.sleep(fallback_sec)
  ```
  Y usarlo después de los swipes de paneo en lugar de `time.sleep(1.30)`.

---

## Fase 2: Mejora del Modelo de Visión y Raycast de Muros (`vision/wall_raycast.py`)

### 2.1. Enmascarar Edificios Detectados por YOLO antes de Canny + `HoughLinesP`
* **Archivo:** `vision/wall_raycast.py` y `attack/deployment_strategy.py`
* **Problema:** `detect_global_wall_segments` aplica el filtro de bordes Canny y `HoughLinesP` sobre toda el área de juego. Como los tejados de los edificios, campamentos y barras de vida tienen la misma inclinación isométrica ($36^\circ$ y $144^\circ$) que los muros, generan falsos positivos de segmentos de muro.
* **Acción:**
  1. Añadir un parámetro opcional `detections: list | None = None` a `detect_global_wall_segments(...)` en `vision/wall_raycast.py` (con valor por defecto `None` para no romper ningún test existente en `tests/test_wall_raycast.py`).
  2. Después de calcular `edges = cv2.Canny(blurred, canny_low, canny_high)`, si `detections` fue proporcionado, enmascarar (poner en `0`) el área interior de cada edificio detectado (aplicando un margen interno de $6\text{ px}$ para no borrar los muros pegados al edificio):
     ```python
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
     ```
  3. En `_compute_wall_raycast_jump_targets` (`attack/deployment_strategy.py`), pasar `detections=detections` al llamar a `detect_global_wall_segments(...)`.

---

## Fase 3: Mejoras Tácticas en `attack/edrag_strategy.py`

### 3.1. Puntos de Furia al Núcleo Dinámicos con YOLO (`get_core_rage_points`)
* **Archivo:** `attack/edrag_strategy.py`
* **Problema:** `get_core_rage_points()` devuelve siempre coordenadas fijas `[(1200, 420), (1100, 360), (1300, 360)]` sin importar dónde estén el Ayuntamiento o las defensas/almacenes principales.
* **Acción:**
  1. Modificar la firma a `def get_core_rage_points(self, detections: list | None = None) -> list[tuple[int, int]]:` (manteniendo `detections=None` por defecto para que `test_opening_and_core_rage_points` en `tests/test_edrag_strategy.py` siga funcionando igual).
  2. Si `detections` contiene estructuras válidas, buscar el `town_hall` o el centroide de los almacenes en la mitad norte/central ($300 < y < 650$) y centrar la primera furia sobre ese objetivo (clampeando $x \in [950, 1450]$ e $y \in [320, 620]$) y las dos furias laterales a $\pm 110\text{ px}$ en $X$ y $-50\text{ px}$ en $Y$.
  3. Guardar las detecciones obtenidas en `compute_north_triangle_drops(screen)` en `self._last_detections` y pasarlas a `self.get_core_rage_points(self._last_detections)` en la Fase 5 de `execute_attack()`.

---

## Fase 4: Rendimiento Global y Limpieza de Arquitectura

### 4.1. Guardado Asíncrono de Imágenes de Auditoría (`cv2.imwrite` no bloqueante)
* **Archivos:** `core/vision.py` (o nuevo helper en `core/logger.py`), `attack/deployment_strategy.py` y `core/state_machine.py`
* **Problema:** Guardar imágenes PNG de $2400 \times 1080$ en el hilo principal durante el ataque y el bucle de espera bloquea la ejecución entre $90\text{ ms}$ y $180\text{ ms}$ por imagen.
* **Acción:**
  1. Crear una función utilitaria `save_debug_image_async(path: str, img: np.ndarray)` respaldada por un `ThreadPoolExecutor(max_workers=1)` que reciba una copia (`img.copy()`) y ejecute `cv2.imwrite(path, img)` en segundo plano, o un modo sincrónico si se ejecuta dentro de `unittest`.
  2. Reemplazar los `cv2.imwrite` de progreso de batalla en `core/state_machine.py` y las auditorías por zona en `attack/deployment_strategy.py` por esta función.

### 4.2. Limpieza de `core/config.py`
* **Archivo:** `core/config.py`
* **Problema:** La segunda mitad del archivo re-importa `json` y `os` con alias `_json` y `_os` e itera sobre `globals()` para buscar `SETTINGS`, el cual ya fue definido en la línea 19.
* **Acción:** Simplificar el archivo usando directamente el diccionario `SETTINGS` cargado al inicio, manteniendo exactamente los mismos nombres de constantes exportadas (`LOOT_WAIT_AFTER_NEXT_MIN`, `USE_YOLO`, `YOLO_MODEL_PATH`, etc.) para no afectar ningún import externo.

---

## Fase 5: Validación y Pruebas

Una vez aplicados los cambios de las Fases 1 a 4, ejecuta la batería completa de pruebas:

```bash
python -m unittest discover -s tests -v
```

Verifica que:
1. Todos los tests existentes en `tests/` pasen sin errores (`OK`).
2. Agrega un test unitario en `tests/test_quota_drop_and_inventory.py` que compruebe que `_deploy_by_structure_quotas` omite los swipes de los flancos si `_troop_inventory["goblins"]` queda en `0` tras la fase `CENTRO`.
3. Agrega un test unitario en `tests/test_wall_raycast.py` que verifique que `detect_global_wall_segments(img, detections=dets)` ignora bordes internos dentro de las cajas delimitadoras de los edificios.