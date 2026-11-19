# MEJORA 04 — Abstracción de `Device`, Escaneo Único de Barra de Batalla y Limpieza de `EDragStrategy`

## Objetivo
Preparar una capa limpia de dispositivo y barra de batalla para eliminar:
1. Llamadas directas a `device.d.shell(...)` / `device.d.click(...)` y hacks de `self.device.__class__.__name__ == "Device"`.
2. Capturas ADB repetidas en batalla solo para re-buscar ranuras fijas de tropas/hechizos/héroes.
3. Trabajo redundante y escrituras síncronas en `attack/edrag_strategy.py`.

## Alcance (Archivos a modificar)
- `core/device.py`
- `vision/army_checker.py`
- `attack/edrag_strategy.py`
- `core/state_machine.py` (inyectar `self.army` a `EDragStrategy`)
- `tests/__init__.py` (actualizar `MockDevice`)
- `tests/test_army_checker.py`
- `tests/test_edrag_strategy.py`

## Restricciones (NO tocar)
- **NO modificar `attack/deployment_strategy.py`** en este módulo (se conectará en `MEJORA_05`).
- Mantener 100% retrocompatibilidad en los métodos existentes de `Device`, `ArmyChecker` y `MockDevice`.

---

## Tareas Técnicas Detalladas

### 1. Enriquecer la abstracción `core/device.py` (y su espejo `MockDevice`)
**Problema actual:**
Las estrategias y el `ProcessManager` acceden directamente a `self.device.d.shell(...)`, `self.device.d.click(...)` y `self.device.d.press("back")`, rompiendo el encapsulamiento y obligando a validar `self.device.__class__.__name__ == "Device"` para que los tests no duerman ni fallen.

**Solución requerida:**
1. En `core/device.py`:
   - Agrega atributo de clase/instancia `is_mock: bool = False`.
   - Agrega `press_back(self)` con manejo de reconexión igual que `tap()`.
   - Agrega `raw_tap(self, x: int | float, y: int | float)` como alias explícito de `self.tap(x, y, humanize=False)`.
   - Agrega `pan_flank(self, flank: str, reverse: bool = False, duration_ms: int = 480) -> None` que calcule las coordenadas **relativas** a `(self.width, self.height)` para `"north"`, `"south"`, `"east"` y `"west"`:
     - Referencia calibrada en `2400x1080`:
       - `north`: `(0.50, 0.222) -> (0.50, 0.759)`
       - `south`: `(0.50, 0.722) -> (0.50, 0.222)`
       - `east`: `(0.646, 0.50) -> (0.354, 0.50)`
       - `west`: `(0.354, 0.50) -> (0.646, 0.50)`
     - Si `reverse=True`, invierte `(x1, y1)` y `(x2, y2)`.
     - Ejecuta el swipe mediante `self.swipe(x1, y1, x2, y2, duration=duration_ms / 1000.0)`.
2. En `tests/__init__.py` (`MockDevice`):
   - Agrega `is_mock: bool = True`.
   - Implementa `press_back()`, `raw_tap()`, `pan_flank()`, `wait_camera_settled(timeout=...) -> True`, `zoom_half()` y `hold_tap()` registrando el historial para que los tests puedan asertar limpiamente.

---

### 2. Escaneo único de barra de batalla en `vision/army_checker.py`
**Problema actual:**
Durante una batalla, el bot toma una captura nueva por ADB cada vez que quiere seleccionar duendes, SWB, salto, aceleración o héroes, haciendo múltiples conversiones `BGR -> Gray` y búsquedas en `MITAD_INFERIOR`. La posición de las cartas en la barra inferior **no cambia** a lo largo de la batalla.

**Solución requerida:**
1. Crea un `@dataclass class BattleBarSlots`:
   ```python
   @dataclass
   class BattleBarSlots:
       goblin: tuple[int, int] | None = None
       swb: tuple[int, int] | None = None
       edrag: tuple[int, int] | None = None
       siege: tuple[int, int] | None = None
       jump: tuple[int, int] | None = None
       haste: tuple[int, int] | None = None
       rage: tuple[int, int] | None = None
       heroes: dict[str, tuple[int, int]] = field(default_factory=dict)
   ```
2. Implementa `ArmyChecker.scan_battle_bar(self, screenshot) -> BattleBarSlots`:
   - Convierte `screenshot` a escala de grises **una sola vez** y recorta `MITAD_INFERIOR` **una sola vez**.
   - Reutiliza ese único recorte en gris para localizar todos los slots (tropas, hechizos y héroes con la desduplicación espacial ya existente).
3. Mantén todos los métodos `find_*_in_battle_bar()` actuales intactos para retrocompatibilidad.

---

### 3. Limpiar y optimizar `attack/edrag_strategy.py`
**Problemas actuales:**
1. Crea su propia instancia `self.army = ArmyChecker(self.vision)` en lugar de recibir el `army_checker` compartido desde `ClashStateMachine`.
2. `compute_north_triangle_drops()` valida **10 puntos** (`5 NW + 5 NE`) contra `DeployZoneDetector` (U-Net), pero `execute_attack()` solo usa `nw_drops[1]` y `ne_drops[1]` y descarta los otros 8.
3. Guarda `edrag_attack_start_*.png` y `edrag_attack_deployed_*.png` usando `cv2.imwrite()` síncrono en pleno combate.
4. `execute_attack()` llama a 3 métodos separados sobre `screen` para localizar `edrag`, `rage` y `heroes` en vez de usar `scan_battle_bar()`.

**Solución requerida:**
1. Acepta parámetro opcional `army_checker: ArmyChecker | None = None` en `EDragStrategy.__init__()` (con fallback `army_checker or ArmyChecker(self.vision)` para no romper tests) e inyéctalo desde `ClashStateMachine.__init__()`.
2. En `execute_attack()`:
   - Usa `slots = self.army.scan_battle_bar(screen)` para detectar E-Drags, Furias y Héroes en una sola pasada (respetando los mocks existentes si en tests se parchean `find_edrag_in_battle_bar` / `find_rage_spell_in_battle_bar` / `find_all_heroes_in_battle_bar`, o actualizando el test de manera limpia).
   - Reemplaza los `cv2.imwrite(...)` por `save_debug_image_async(...)` de `core.vision`.
3. En `compute_north_triangle_drops()`:
   - Agrega un parámetro `only_anchors: bool = False` (o `validate_all: bool = True`). Cuando se llame desde `execute_attack()`, valida únicamente los 2 puntos ancla (`nw_drops[1]` y `ne_drops[1]`) contra `DeployZoneDetector` (ahorrando 8 búsquedas inútiles). Mantén la firma por defecto compatible con `test_north_triangle_drops_geometry`.

---

## Criterios de Aceptación y Verificación

1. Agrega tests en `tests/test_army_checker.py` para `scan_battle_bar()` verificando detección simultánea de tropas, hechizos y héroes en un solo frame sintético.
2. Verifica que `EDragStrategy` usa `save_debug_image_async` y no bloquea en disco.
3. Ejecuta:
```powershell
python -m unittest tests.test_army_checker tests.test_edrag_strategy tests.test_heroes_haste_and_cross_pan tests.test_ui_cache -v
```
Todos los tests deben pasar en verde.