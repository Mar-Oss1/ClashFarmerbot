# Plan: validación real de puntos de drop de duendes (ClashFarmerbot)

> Para Antigravity. Léelo completo antes de tocar código. Trabaja por FASES y respeta los **PUNTOS DE PARADA**: en cada uno te detienes, me entregas el reporte y esperas mi revisión antes de seguir.

## 0. Contexto del problema

El bot corre con `attack.deployment_mode = "structure_quota_deployment"` y `attack.boundary_detection_mode = "deploy_zone_finder"`. Para cada estructura detectada por YOLO (storages, colectores, town_hall) calcula un punto de drop en césped legal usando el `DeployZoneDetector` (U-Net ONNX).

**Síntoma:** a veces los duendes se sueltan encima de una mina/almacén que está FUERA del muro.

**Hipótesis a confirmar o descartar con datos (no asumas ninguna):**

- **A. Falla del modelo:** el U-Net no marca como prohibido un edificio chico y aislado. Entonces `find_closest_deployable_point` ve el centro del edificio como "desplegable" y devuelve ese mismo punto (early return al inicio de la función).
- **B. Contornos ciegos:** `get_boundary_polygons` se queda solo con el contorno más grande (`RETR_EXTERNAL` + `max(contourArea)`). Los edificios sueltos fuera del muro forman bloques aparte en la máscara y no entran en `red_contour`/`green_contour`. El chequeo `inside_red` en `_execute_single_zone_quota_deploy` y la imagen `zone_*_debug.png` solo ven ese contorno.
- **C. Fallback silencioso:** si el modelo no está listo, `last_mask` es `None` o hay excepción, cae al rombo canónico fijo (que no conoce edificios sueltos). Además `is_point_deployable` devuelve `True` cuando no hay máscara.
- **D. Máscara vieja:** si se pasan `precomputed_red/green`, `predict_mask` no se reejecuta para esa vista y `last_mask` puede ser de otra posición de cámara.

## 1. Reglas generales

1. Crea una rama o un respaldo antes de empezar. No toques `_backups/`.
2. No modifiques el modo legacy de sweep (`SWEEP_SUR`, `SWEEP_NORTE` y su `execute_attack`).
3. Toda opción nueva va en `config/settings.json` bajo `attack.deploy_zone_ai` con un default que **no cambie el comportamiento actual** hasta que yo lo apruebe, salvo lo que diga explícitamente este plan.
4. Mantén compatibilidad de firmas: los parámetros nuevos son opcionales. No rompas `tests/` existentes (`python -m unittest` debe seguir en verde, o dime qué falla y por qué).
5. Usa `core.logger.get_logger` con el estilo actual (español, emojis). Nada de `print`.
6. Sin dependencias nuevas pesadas. No subas modelos ni capturas a git.
7. Los logs de decisión deben ser **explícitos**: cada vez que se use un fallback, se vete un punto o se detecte un posible fallo del modelo, un `log.warning` con el motivo.
8. Rutas de Windows (`C:/ClashFarmerbot`). Usa `os.path.join`/`PROJECT_ROOT`.

---

## FASE 1: Diagnóstico offline (sin tocar el juego)

### 1.1 Verificaciones rápidas

- Confirma que `models/deploy_zone_finder.onnx` existe y que `get_deploy_zone_detector().is_ready()` es `True`. Reporta tamaño del archivo, providers de ONNX y tiempo medio de inferencia (ms) sobre 5 imágenes.
- Confirma que `models/yolov8s.onnx` (o el que esté en `yolo_model_path`) carga y reporta `class_names` y si `is_finetuned()` es `True`.

### 1.2 Script `tools/diagnose_drop_points.py`

Entrada: carpeta de imágenes (por defecto `datasets/coc/images/raw`, más las capturas de `logs/yolo/` si existen). Argumentos: `--images`, `--out logs/diagnostics/drops`, `--limit`.

Para cada imagen:

1. Corre YOLO (`get_detector().detect`) y filtra el HUD igual que `_execute_single_zone_quota_deploy` (`get_battle_hud_mask`).
2. Corre `DeployZoneDetector.predict_mask` (guarda la máscara completa, no solo el contorno mayor).
3. Para cada detección calcula y registra:
   - `mask_at_center` (0/1) y `frac_bbox_forbidden` (fracción de píxeles del bbox marcados como prohibidos por el U-Net).
   - `deployable_at_center` según `is_point_deployable(cx, cy, safety_margin_px=0)`.
   - Si el centro pertenece a un componente conexo de la máscara que NO es el contorno más grande (blob aislado), marca `isolated_blob=True`.
   - El drop que calcula HOY `_get_drop_point_on_contour(...)` (con la lógica actual, sin modificarla) y su `drop_in_mask`, `drop_inside_bbox_of_any_detection` (con dilatación de 0, 15, 25, 35 y 50 px) y `drop_inside_red_contour_main` (lo que ve hoy el chequeo `inside_red`).
   - Si el drop salió por la rama IA, por el rombo canónico o por clamp.
4. Clasifica cada detección en una causa: `OK`, `A_MODEL_MISS` (centro desplegable), `B_CONTOUR_BLIND` (blob aislado ignorado por el contorno mayor), `C_FALLBACK` (no usó la rama IA), `D_STALE_MASK` (si es detectable; si no, déjalo `UNKNOWN`).
5. Genera una imagen por captura con: máscara completa en magenta traslúcido, TODOS los contornos, bboxes YOLO con su clase, drops con color por estado (verde = OK, rojo = cae sobre bbox o máscara, naranja = fallback) y flecha drop → objetivo.

Salidas: `results.csv` (una fila por detección), PNGs por imagen y `summary.md` con conteos por causa, por clase y por imagen, y la lista de las 10 peores capturas.

### 1.3 Barrido de dilatación (para decidir el margen del veto YOLO)

Para dilataciones `[15, 25, 35, 50]` px reporta, sobre todas las imágenes:

- Cuántos drops actuales quedarían vetados.
- Cuántos objetivos quedarían sin punto legal (inalcanzables) con la lógica nueva (cálculo aproximado: proyecta sobre la máscara combinada, ver Fase 3).

Sirve para elegir un margen que no deje inalcanzables a edificios legítimos.

### 🛑 PUNTO DE PARADA 1

Entrégame `summary.md` + las 10 peores imágenes + tu lectura de qué hipótesis (A/B/C/D) explican los casos. **No implementes nada de la Fase 3 hasta que yo responda.**

---

## FASE 2: Test de regresión que reproduce el bug (antes del fix)

Crea `tests/test_drop_validation.py` (unittest, estilo de los tests actuales) con máscaras sintéticas, sin ONNX:

1. **Centro desplegable = fallo del modelo:** máscara con la aldea grande prohibida y un edificio chico aislado fuera (YOLO dice que existe, la máscara lo deja en 0). Hoy `find_closest_deployable_point` devuelve el centro del edificio → el test debe **fallar ahora** y pasar tras el fix.
2. **Blob aislado:** máscara con dos componentes. Verifica que el chequeo de zona prohibida considera ambos y no solo el mayor.
3. **Sin máscara:** `is_point_deployable` con `mask=None` y sin `last_mask` (el comportamiento nuevo se define en la Fase 3).
4. **Máscara de otra vista:** que no se pueda validar un punto con una máscara cuyo `screen_id` no coincide con el de la captura actual.

Marca estos tests como `expectedFailure` solo si hace falta para que el suite siga corriendo; el objetivo es que dejen de serlo en la Fase 3.

---

## FASE 3: Implementación del fix (solo tras mi aprobación)

Aplica **solo** lo que mi respuesta del Punto de Parada 1 apruebe. Diseño base:

### 3.1 Máscara combinada (`vision/deploy_zone_detector.py`)

- Nuevo método `build_combined_mask(shape, detections, dilate_px, per_class_dilate=None) -> np.ndarray`: `mask_unet OR (bbox de TODAS las detecciones YOLO dilatados)`. Los bbox van con clase (no solo loot) y con dilatación configurable.
- Guarda `self.last_combined_mask` y un `self.last_screen_id` (hash corto de la captura) para detectar máscara vieja.
- `is_point_deployable` y `find_closest_deployable_point` aceptan `mask=` (ya lo hacen); los llamadores pasan la combinada.
- Config nueva: `attack.deploy_zone_ai.yolo_veto: {enabled: true, dilate_px: <valor aprobado>, per_class: {}}`.

### 3.2 Regla del "centro desplegable"

En `find_closest_deployable_point`: si el centro del objetivo sale desplegable según la máscara **del U-Net sola**, trátalo como sospecha de falla del modelo:

- `log.warning("⚠️ MODEL_MISS: ...")` con clase, centro y confianza.
- Calcula el punto legal buscando desde el borde del bbox dilatado hacia afuera (no devuelvas el centro).
- Incrementa un contador `model_miss_count` por batalla, visible en el reporte final.

### 3.3 Contornos completos

- En `get_boundary_polygons` conserva el comportamiento de retorno actual (contorno mayor) pero guarda además `self.last_all_contours` (todos los componentes con área mínima configurable).
- En `_execute_single_zone_quota_deploy`, sustituye el chequeo `inside_red` basado en `pointPolygonTest` del contorno mayor por una consulta directa a la máscara combinada (`is_point_deployable(..., mask=combined)`). Mantén el `pointPolygonTest` como segunda comprobación si el contorno existe.

### 3.4 Máscara vieja

- `predict_mask` debe asociar la máscara al `screen_id` de la captura. Cuando `_get_drop_point_on_contour` o el flujo de drop usen la máscara, verifica que el `screen_id` coincide con la captura de esa vista; si no coincide, reejecuta `predict_mask` (log `🔄`) en vez de usar la caché.
- Revisa el uso de `precomputed_screen/red/green/detections`: confirma que la máscara asociada corresponde a `precomputed_screen`.

### 3.5 Fallbacks que ya no son silenciosos

- Cuando se use el rombo canónico por cualquier motivo, `log.warning` con el motivo exacto (`model_not_ready`, `mask_none`, `exception: ...`, `degenerate_contour`).
- `is_point_deployable` con máscara `None`: mantén `True` por compatibilidad pero loguea un warning (una vez por batalla).
- Config `attack.deploy_zone_ai.fail_closed_without_mask` (default `false`): si es `true` y no hay máscara, la zona no se ataca y se reporta como `NO_MASK`. Déjala en `false` hasta que yo decida.

### 3.6 Guardia final antes del click

En el bucle donde se sueltan los duendes (después de la sección `# 6. Despliegue de tropas con ranura cacheada`; ubica el `click`/`deploy_stream` exacto), antes de cada toque:

- Llama a `assert_drop_legal(x, y, combined_mask)`. Si es ilegal, reproyecta con `find_closest_deployable_point`; si no hay punto, **omite** el toque y regístralo (`🚫 DROP_VETADO`).
- Aplica lo mismo a SWB, héroes, haste y saltos solo si el punto es de drop de tropa (los hechizos pueden caer dentro de la aldea, no los valides contra la zona prohibida).

### 3.7 Debug visual útil

- `zone_<ZONA>_debug.png` debe dibujar la máscara completa combinada y todos los contornos, bboxes YOLO y cada drop con su estado y motivo.
- Guarda por ataque en `logs/yolo/attack_<timestamp>/`: captura cruda, detecciones (JSON), máscara U-Net (PNG), máscara combinada (PNG), overlay final y `decisions.json` con una fila por objetivo: `{clase, centro, drop, razón, mask_unet_en_centro, veto_yolo, fallback_usado, tropas_desplegadas}`.

---

## FASE 4: Pruebas offline tras el fix

1. Corre `python -m unittest` completo. Los tests de la Fase 2 deben pasar.
2. Re-ejecuta `tools/diagnose_drop_points.py` en modo "post-fix" sobre las mismas imágenes y entrega la comparación antes/después:
   - Drops dentro de cualquier bbox YOLO dilatado: debe ser **0**.
   - Drops en máscara combinada: **0**.
   - Fallbacks silenciosos: **0**.
   - Objetivos alcanzables antes vs. después (no debe caer sin justificación; explica cada caso que sí caiga).
   - Latencia añadida por la máscara combinada: debe ser menor a 50 ms por vista.
3. Entrega las imágenes antes/después de las 10 peores capturas.

### 🛑 PUNTO DE PARADA 2

Entrégame la comparación antes/después. **No pases a pruebas en vivo hasta que yo apruebe.**

---

## FASE 5: Pruebas en vivo (dispositivo real)

Cuidado: cada búsqueda de partida cuesta oro. Haz las pruebas en este orden y no pases al siguiente escalón si el anterior falla.

### 5.1 Dry-run (sin tocar la pantalla de batalla)

Agrega un flag `--dry-run-drops` a `main.py` (o a `tools/run_live_test_3_attacks.py`):

- Entra a una batalla real, captura, corre YOLO + máscara + cálculo completo de drops y guarda todo el paquete de la sección 3.7, **sin enviar ningún click de despliegue**.
- Termina rindiéndose por el flujo normal del bot (`allow_surrender`). **No reinicies la app** (el juego impone un temporizador de ~1.5 min si se sale en lugar de terminar la batalla).
- Corre 3 dry-runs. Entrégame los 3 paquetes.

### 🛑 PUNTO DE PARADA 3

Entrégame los 3 paquetes del dry-run con tu lectura. Yo reviso los overlays y decido si pasamos a ataques reales.

### 5.2 Ataques reales supervisados

Con mi aprobación, usa `tools/run_live_test_3_attacks.py` (o `main.py --attacks 3`) con logging completo:

- Antes y después de cada tanda de duendes captura la pantalla (a ~1.5 s del último toque) para ver dónde aparecieron realmente.
- Si `vision/army_checker.py` permite leer el contador de duendes de la carta, compara antes/después de cada drop: si el contador no baja, el toque fue ilegal (el juego lo ignoró); regístralo como `TAP_IGNORED`. Si no es viable, omite esto y dímelo.
- Si ocurre cualquier `TAP_IGNORED`, `DROP_VETADO` inesperado o un drop visible encima de una estructura, **detén el siguiente ataque** y repórtalo.

### 5.3 Reporte de hallazgos

Genera `logs/findings/drop_validation_report.md` con:

- Tabla por ataque y por objetivo: clase, drop, razón, valor del U-Net en el centro, veto YOLO sí/no, fallback usado, tropas desplegadas, `TAP_IGNORED` sí/no.
- Conteo de `MODEL_MISS`, fallbacks, vetos y reproyecciones.
- Las imágenes antes/después más reveladoras.
- Qué hipótesis (A/B/C/D) se confirmó con datos en vivo y cuáles se descartaron.
- Riesgos o dudas abiertas, y qué NO pudiste verificar.

### 🛑 PUNTO DE PARADA 4 (final)

Entrégame el reporte y espera. Revisamos los hallazgos juntos antes de cambiar defaults (`yolo_veto.dilate_px`, `fail_closed_without_mask`) o de evaluar si hace falta reentrenar/afinar el modelo con capturas propias.

---

## Criterios de éxito

- 0 drops de duendes dentro de un bbox YOLO dilatado ni dentro de la máscara combinada.
- 0 fallbacks silenciosos; todo fallback queda en el log con motivo.
- Ningún objetivo legítimo pasa a inalcanzable sin explicación.
- Overhead < 50 ms por vista.
- Tests existentes en verde y los nuevos de `tests/test_drop_validation.py` pasando.

## Qué NO hacer

- No cambiar umbrales (`yolo_conf`, `conf_thresh`, `imgsz`) para "arreglar" el síntoma sin datos que lo respalden.
- No reentrenar modelos en este plan.
- No tocar la lógica de saltos/`wall_raycast` salvo lo necesario para pasar la máscara combinada.
- No avanzar de fase sin pasar por el Punto de Parada correspondiente.
