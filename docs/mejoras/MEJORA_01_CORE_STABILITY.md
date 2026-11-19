# MEJORA 01 — Estabilidad del Core y Corrección de Bugs Críticos

## Objetivo
Eliminar cuellos de botella graves en el bucle principal del bot (`5s` de congelamiento por tick), prevenir crashes de rotación de logs en Windows, y corregir inconsistencias lógicas en `state_machine.py` y `wall_cost_table.py`.

## Alcance (Archivos a modificar)
- `core/process_manager.py`
- `core/logger.py`
- `core/state_machine.py`
- `core/wall_cost_table.py`
- `tests/test_wall_upgrader.py` (solo si se actualiza la regla `can_afford`)
- Nuevo test: `tests/test_core_stability.py`

## Restricciones (NO tocar)
- **NO modificar `attack/deployment_strategy.py`**.
- **NO alterar** la firma pública de `ProcessManager`, `ClashStateMachine` ni `get_logger()`.

---

## Tareas Técnicas Detalladas

### 1. Eliminar congelamiento de 5 segundos por tick en `core/process_manager.py`
**Problema actual:**
En `ProcessManager.handle_popup()`, el paso 7 llama a `self.handle_chest_reward_sequence(screen)`. Dentro de `handle_chest_reward_sequence()`, si no hay cofre en pantalla, el bucle entra en un `while time.monotonic() - t_wait_chest < 5.0` tomando capturas ADB cada `0.35s`. Como `ClashStateMachine.run_forever()` llama a `handle_popup()` en cada iteración del bucle principal (y `MAIN_VILLAGE` lo vuelve a llamar), cada pantalla limpia quema de 5 a 10 segundos inútilmente.

**Solución requerida:**
1. Agrega el parámetro `wait_for_chest: bool = False` a `handle_chest_reward_sequence(self, screen=None, wait_for_chest: bool = False) -> bool`.
2. Cuando `wait_for_chest=False`:
   - Revisa únicamente `cur_screen` actual (`cont_match` y `chest_match`).
   - Si no hay ninguno en la captura inicial, retorna `False` **inmediatamente** (0 segundos de espera y 0 capturas ADB adicionales).
   - Solo si se detectó y abrió un cofre (`handled_any = True`), permite esperar brevemente (máx `2.5s`) por cofres encadenados en las vueltas `cofre_idx >= 2`.
3. Cuando `wait_for_chest=True`:
   - Mantiene la espera activa de `5.0s` (usada exclusivamente justo después de pulsar `Claim` en Star Bonus o `Claim Reward` en eventos).
4. Actualiza las llamadas:
   - En `handle_popup()` paso 7: `self.handle_chest_reward_sequence(screen, wait_for_chest=False)`.
   - En `handle_popup()` pasos 6 y 8 (tras pulsar `Claim` / `Claim Reward`): `self.handle_chest_reward_sequence(wait_for_chest=True)`.
   - En `core/state_machine.py` (tras pulsar `Claim Reward`): `self.proc_mgr.handle_chest_reward_sequence(wait_for_chest=True)`.

---

### 2. Corregir rotación de logs en Windows en `core/logger.py`
**Problema actual:**
`get_logger(name)` instancia un `RotatingFileHandler` independiente sobre el mismo archivo `logs/clash_bot.log` por cada módulo del proyecto (~15 handles simultáneos). En Windows, al superar 5 MB, `os.rename` falla con `PermissionError` porque otros handlers mantienen el archivo abierto.

**Solución requerida:**
1. Crea un logger padre único `_ROOT_LOGGER_NAME = "clashbot"` inicializado una sola vez (con un lock o guard `if not root.handlers:`) que tenga **un único `StreamHandler`** y **un único `RotatingFileHandler`**.
2. `get_logger(name: str, level: int = logging.INFO)` debe:
   - Garantizar que el logger padre esté inicializado.
   - Retornar `logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")` con `propagate = True` y **sin añadir handlers duplicados** al logger hijo.
   - Mantener el formato de salida actual (`%(asctime)s | %(levelname)-7s | %(name)-16s | %(message)s`).

---

### 3. Corregir inconsistencias en `core/state_machine.py`
**Problemas actuales:**
1. `config.SEARCH_ONLY` se define en `core/config.py` y `config/settings.json`, pero `ClashStateMachine` nunca lo consulta.
2. En `WAITING_BATTLE_END` (tanto en modo `edrag_rush` como en `sneaky_farming`), el bucle solo busca `self.tpl_return_home`, ignorando `self.tpl_claim_rewards`. Cuando aparece el cofre de evento al terminar la batalla, el bot no detecta el fin natural y espera hasta timeout/estancamiento.
3. `BotState.RETURNING_HOME` es código muerto que nunca se asigna en ninguna transición.
4. En `MAIN_VILLAGE`, `handle_popup()` se llama en `run_forever()` y otra vez en la primera línea de `MAIN_VILLAGE` (doble captura ADB redundante).
5. `assets/buttons/return_home_event.png` está listado en `self.tpl_return_homes` pero no existe en `assets/buttons/`.

**Solución requerida:**
1. En `EVALUATING_LOOT`: si `loot is not None and meets`:
   - Si `config.SEARCH_ONLY` es `True`: registra en log `"🔎 [SEARCH_ONLY] Aldea apta encontrada — deteniendo ataque por modo solo búsqueda."`, no entra a `ATTACKING` y hace `break` limpio o espera controlada según diseño (mejor: añade un atributo `self._stop_requested = True` para salir limpio del bucle `run_forever()`).
2. En `WAITING_BATTLE_END` (ambos modos): reemplaza el chequeo de fin natural de batalla por:
   ```python
   if self.vision.find_any(screen, self.tpl_return_homes + self.tpl_claim_rewards, threshold=0.72, zona="MITAD_INFERIOR"):
   ```
3. Elimina la llamada duplicada `if self.proc_mgr.handle_popup(): return` al inicio de `MAIN_VILLAGE` (ya se ejecuta antes de `_tick()` en `run_forever()`).
4. Limpia `BotState.RETURNING_HOME` si no tiene referencias en tests ni herramientas (verifica con búsqueda en todo el repo antes de remover).
5. Filtra `self.tpl_return_homes` para incluir solo templates existentes o mantén solo `return_home.png` hasta que exista un asset real.

---

### 4. Corregir regla de negocio `can_afford()` en `core/wall_cost_table.py`
**Problema actual:**
En `core/wall_cost_table.py`, `can_afford(target_level, available_gold, available_elixir)` retorna `False` si falta Oro **O** si falta Elixir para niveles `>= 5` (lógica `AND`). En Clash of Clans, a partir de nivel 5 un muro se mejora pagando con **Oro O con Elixir** (lógica `OR`), nunca ambos al mismo tiempo.

**Solución requerida:**
1. En `can_afford()`:
   - Si `req["elixir"] is None` (niveles 1–4): retorna `available_gold >= req["gold"]`.
   - Si `req["elixir"] is not None` (niveles 5+): retorna `(available_gold >= req["gold"]) or (available_elixir >= req["elixir"])`.
   - Opcionalmente, agrega parámetro `require_both: bool = False` por si algún flujo requiere validar ambos explícitamente.
2. Actualiza el docstring de `can_afford()` y los tests contradictorios en `tests/test_wall_upgrader.py` (por ejemplo `test_can_afford_level_5_plus_rejects_if_missing_elixir_even_with_excess_gold`) para reflejar la regla real de Clash of Clans (acepta si alcanza Oro **o** Elixir; rechaza solo si no alcanza ninguno de los dos).

---

## Criterios de Aceptación y Verificación

1. Crea `tests/test_core_stability.py` cubriendo:
   - `ProcessManager.handle_popup()` en pantalla limpia (imagen negra mockeada) termina en **< 0.2 segundos** (antes tardaba > 5s).
   - `get_logger("mod_a")` y `get_logger("mod_b")` comparten exactamente **1 solo** `RotatingFileHandler` en el logger padre.
   - `ClashStateMachine` respeta `SEARCH_ONLY=True` sin invocar `execute_attack()`.
   - `can_afford(19, 10_000_000, 0)` retorna `True`, `can_afford(19, 0, 10_000_000)` retorna `True`, y `can_afford(19, 5_000_000, 5_000_000)` retorna `False`.
2. Ejecuta:
```powershell
python -m unittest tests.test_core_stability tests.test_wall_upgrader tests.test_storage_trigger tests.test_ui_cache -v
```
Todos los tests deben pasar en verde.