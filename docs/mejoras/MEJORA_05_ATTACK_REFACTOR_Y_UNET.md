# MEJORA 05 — Modularización de `deployment_strategy.py` e Integración de U-Net + Beam Search

> **ADVERTENCIA DE SEGURIDAD (FASE 4):** Este módulo modifica `attack/deployment_strategy.py`.
> **NO ejecutar** sin confirmación explícita del usuario y sin crear primero una rama Git o respaldo limpio.

## Objetivo
1. Conectar tu mejor componente —`vision/wall_compartments.py` (`U-Net` + `Beam Search`)— como estrategia de saltos seleccionable (`"jump_placement": "coverage_optimizer"`) en combate real.
2. Romper el monolito `attack/deployment_strategy.py` extrayendo la percepción de escena (`VillageScene`) y la lógica de planificación de saltos (`attack/jump_planner.py`), manteniendo 100% compatibilidad con todos los tests y herramientas existentes.
3. Eliminar escrituras síncronas en combate, escaneos repetidos de la barra de batalla, inventario hardcodeado y la deriva de cámara en paneos.

## Alcance (Archivos a crear/modificar)
- **Nuevo:** `vision/village_analyzer.py` (`VillageScene` + `VillageAnalyzer`)
- **Nuevo:** `attack/jump_planner.py` (despachador unificado: `coverage_optimizer`, `wall_raycast`, `cluster_intersections`, `legacy_quadrant`)
- `attack/deployment_strategy.py`
- `config/settings.json` (añadir configuración de inventario de ejército y `debug.save_debug_images`)
- Nuevo test: `tests/test_jump_planner_integration.py`

## Requisitos previos
Haber completado y verificado `MEJORA_01` a `MEJORA_04`.

---

## Tareas Técnicas Detalladas

### 1. Crear fachada de percepción `vision/village_analyzer.py`
**Problema actual:**
En una misma vista de ataque, YOLO y `DeployZoneDetector` (U-Net) llegan a ejecutarse varias veces sobre el mismo frame o en funciones separadas sin compartir resultados.

**Solución requerida:**
1. Define un `@dataclass class VillageScene`:
   - `screen: np.ndarray`
   - `zone_name: str`
   - `detections: list[Detection]`
   - `deploy_mask: np.ndarray | None`
   - `red_contour: np.ndarray | None`
   - `green_contour: np.ndarray | None`
   - `wall_mask: np.ndarray | None` (lazy o calculada solo cuando se necesitan saltos)
2. Crea `VillageAnalyzer.analyze(screen, zone_name="CENTRO", boundary_mode=None) -> VillageScene`:
   - Ejecuta YOLO una sola vez y filtra por `vision.hud.get_battle_hud_mask()`.
   - Ejecuta `detect_village_boundary()` una sola vez reutilizando esas detecciones.
   - Guarda los resultados en `VillageScene` para que tanto la asignación de drops como el planificador de saltos consuman el mismo contexto sin re-inferir.

---

### 2. Extraer planificador de saltos en `attack/jump_planner.py` y conectar `coverage_optimizer`
**Problema actual:**
`attack/deployment_strategy.py` contiene más de 400 líneas dedicadas a 3 algoritmos de saltos y **nunca llama a `vision.wall_compartments.optimize_jumps()`** a pesar de que `settings.json` ya define todo el bloque `jump_optimizer`.

**Solución requerida:**
1. Mueve a `attack/jump_planner.py`:
   - `_compute_legacy_quadrant_jump_targets`
   - `_compute_cluster_intersection_jump_targets`
   - `_compute_wall_raycast_jump_targets` (limpiando la lectura duplicada de `tol_deg` y la variable muerta `strip`)
   - Agrega `_compute_compartment_jump_targets(screen, detections, screen_size, drop_points=None)`:
     - Invoca `vision.wall_compartments.optimize_jumps(...)` pasando `base_img=screen`, `detections=detections`, `drop_points=drop_points`, `generate_debug_overlay=SETTINGS.get("debug", {}).get("save_debug_images", False)`.
     - Devuelve `[(x, y) for x, y, _ in res.jumps]`.
     - Si `res.debug_overlay is not None`, lo guarda vía `save_debug_image_async("logs/yolo/compartments_debug.png", res.debug_overlay)`.
2. Actualiza `compute_adaptive_jump_targets_from_detections(detections, screen_size, screen=None)`:
   - Soporta `jump_placement`:
     - `"coverage_optimizer"` (U-Net + Beam Search en `wall_compartments.py`, con fallback a `wall_raycast` si `screen is None`).
     - `"wall_raycast"` (mantiene compatibilidad como default actual hasta que el usuario cambie el valor en `settings.json` tras pruebas en vivo).
     - `"cluster_intersections"`
     - `"legacy_quadrant"`
3. Re-exporta `compute_adaptive_jump_targets_from_detections`, `_compute_wall_raycast_jump_targets`, `_compute_cluster_intersection_jump_targets` y `_compute_legacy_quadrant_jump_targets` en `attack/deployment_strategy.py` para **no romper ningún test ni script en `tools/`**.

---

### 3. Limpiar `attack/deployment_strategy.py`
1. **Usar `ArmyChecker.scan_battle_bar()` una sola vez al inicio de `execute_attack()`:**
   - Guarda `self._battle_slots = self.army.scan_battle_bar(screen_centered)` para evitar tomar capturas ADB cada vez que cambias entre saltos, SWB, aceleración, héroes y duendes.
   - Si alguna ranura es `None`, haz un único re-escaneo de seguridad opcional.
2. **Eliminar hacks de test y bypass de `Device`:**
   - Reemplaza `if self.device.__class__.__name__ == "Device"` por `if not getattr(self.device, "is_mock", False)`.
   - Reemplaza `self.device.d.click(x, y)` por `self.device.raw_tap(x, y)` (manteniendo compatibilidad con mocks que solo implementan `d.click` mediante fallback o actualizando `MockDevice`).
   - En `_deploy_by_structure_quotas`, usa `self.device.pan_flank(flank)` y `self.device.pan_flank(flank, reverse=True)`; tras cada par de flancos (p. ej. Norte/Sur antes de pasar a Este/Oeste), o si se detecta deriva, ofrece opción configurable `recenter_zoom_between_axes` para evitar *camera drift* acumulado.
3. **Configurar inventario de tropas en `config/settings.json`:**
   - Añade en `settings.json -> attack.army_inventory`:
     ```json
     "army_inventory": {
       "goblins": 96,
       "swb": 8,
       "jump": 4,
       "haste": 3
     }
     ```
   - Elimina los 3 diccionarios hardcodeados repetidos en `deployment_strategy.py` y crea un método `_reset_battle_inventory()` que cargue desde `SETTINGS`.
4. **Eliminar escrituras síncronas en combate:**
   - Reemplaza todos los `cv2.imwrite(...)` en `_save_yolo_attack_debug`, `_detect_red_exclusion_boundary` y `_survey_and_analyze_village` por `save_debug_image_async(...)`, condicionados a `SETTINGS.get("debug", {}).get("save_debug_images", True)`.
5. **Reutilizar `vision.hud.get_battle_hud_mask`** (re-exportándola en `deployment_strategy.py` por compatibilidad).

---

## Criterios de Aceptación y Verificación

1. Crea `tests/test_jump_planner_integration.py` validando:
   - `compute_adaptive_jump_targets_from_detections()` con `jump_placement="coverage_optimizer"` invoca `optimize_jumps()` y devuelve objetivos válidos sin error.
   - `_reset_battle_inventory()` respeta los valores configurados en `SETTINGS["attack"]["army_inventory"]`.
   - `DeploymentStrategy` no hace llamadas síncronas a `cv2.imwrite` durante `execute_attack()`.
2. Ejecuta **toda** la suite del proyecto:
```powershell
python -m unittest discover -s tests -v
```
3. Nota final para el usuario tras pasar los tests: cambiar `"jump_placement": "wall_raycast"` a `"jump_placement": "coverage_optimizer"` en `config/settings.json` cuando desee probar la U-Net + Beam Search en vivo.