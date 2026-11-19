# Guía de Ejecución por Módulos para `agy`

Coloca estos archivos en la raíz del proyecto (`ClashFarmerbot/`) y pásale a `agy` **un solo módulo por sesión**. No avances al siguiente hasta que los tests del módulo actual pasen al 100% y hagas commit en Git.

> **Regla de seguridad del repo:** Los módulos `MEJORA_01` a `MEJORA_04` y `MEJORA_06` **no tocan `attack/deployment_strategy.py`** (respetando el congelamiento de Fase 4). Solo `MEJORA_05` lo modifica y requiere tu autorización explícita antes de lanzarlo.

---

## Orden recomendado de ejecución

| Orden | Archivo | Impacto principal | Riesgo |
| :---: | :--- | :--- | :---: |
| 1 | `MEJORA_01_CORE_STABILITY.md` | Elimina lag de 5s por tick, arregla crash de logs en Windows y bugs de `state_machine` | Bajo |
| 2 | `MEJORA_02_VISION_PERFORMANCE.md` | Arregla colisión de caché de muros, acelera U-Net drop projection (<1ms) y OCR | Bajo |
| 3 | `MEJORA_03_WALL_COMPARTMENTS_UNET.md` | Conecta `wall_seg_conf`, arregla overlay a escala y acelera Beam Search 5x | Bajo |
| 4 | `MEJORA_04_BATTLE_BAR_Y_DEVICE.md` | Un solo escaneo de barra de batalla, quita hacks de `MockDevice`, limpia E-Drag | Medio |
| 5 | `MEJORA_05_ATTACK_REFACTOR_Y_UNET.md` | Conecta U-Net + Beam Search en combate real y modulariza `deployment_strategy.py` | Alto (requiere tu OK) |
| 6 | `MEJORA_06_CONFIG_Y_REPO_HYGIENE.md` | Limpieza de `settings.json`, `.env`, `requirements` y `.gitignore` | Bajo |

---

## Prompts listos para copiar y pegar en `agy`

### Prompt 1 — Estabilidad del Core
```text
Lee completamente el archivo `MEJORA_01_CORE_STABILITY.md` en la raíz del proyecto y ejecuta el plan paso a paso usando tus herramientas internas.
Restricciones obligatorias:
- NO modifiques `attack/deployment_strategy.py`.
- Antes de editar, revisa los archivos exactos indicados y sus tests actuales.
- Al finalizar, ejecuta la suite de pruebas indicada en la sección de verificación y entrégame un resumen conciso de los cambios y el resultado de cada test.
```

### Prompt 2 — Rendimiento de Visión, ONNX y OCR
```text
Lee completamente el archivo `MEJORA_02_VISION_PERFORMANCE.md` en la raíz del proyecto y aplica todas las mejoras indicadas módulo por módulo.
Restricciones obligatorias:
- NO modifiques `attack/deployment_strategy.py` (mantén la compatibilidad de firma en `vision/wall_compartments.py` y `vision/wall_seg_detector.py` re-exportando `get_battle_hud_mask` desde `vision/hud.py`).
- Verifica con benchmarks simples de tiempo y con los tests unitarios listados al final del documento que no haya regresiones.
```

### Prompt 3 — Optimizador U-Net + Beam Search (`wall_compartments` y `wall_seg_detector`)
```text
Lee completamente el archivo `MEJORA_03_WALL_COMPARTMENTS_UNET.md` en la raíz del proyecto e implementa cada cambio con cuidado.
Restricciones obligatorias:
- NO modifiques `attack/deployment_strategy.py`.
- Asegúrate de mantener 100% la compatibilidad de los tests en `tests/test_wall_compartments.py` y `tests/test_wall_unet_pipeline.py`, y agrega los nuevos tests de regresión pedidos en el documento.
- Ejecuta la verificación completa al terminar.
```

### Prompt 4 — Abstracción de `Device`, Escaneo Único de Barra y Limpieza E-Drag
```text
Lee completamente el archivo `MEJORA_04_BATTLE_BAR_Y_DEVICE.md` en la raíz del proyecto y aplica las mejoras.
Restricciones obligatorias:
- NO modifiques `attack/deployment_strategy.py` todavía (deja listas las nuevas APIs en `core/device.py`, `vision/army_checker.py` y `tests/__init__.py` con retrocompatibilidad total).
- Ejecuta los tests de `army_checker`, `edrag_strategy`, `heroes_haste_and_cross_pan` y `ui_cache` al finalizar.
```

### Prompt 5 — Integración U-Net + Beam Search en Ataque y Refactor de `deployment_strategy.py`
*(Lanza este prompt solo cuando autorices descongelar Fase 4 y tras haber hecho commit en Git)*
```text
Autorizo explícitamente modificar `attack/deployment_strategy.py` bajo un backup previo.
Lee completamente el archivo `MEJORA_05_ATTACK_REFACTOR_Y_UNET.md` en la raíz del proyecto e impleméntalo siguiendo el orden de fases definido en el documento.
Restricciones obligatorias:
- Crea primero un respaldo o rama git antes de tocar `attack/deployment_strategy.py`.
- Mantén 100% retrocompatibilidad con las funciones públicas importadas por `tests/` y `tools/`.
- Ejecuta TODA la suite `python -m unittest discover -s tests` y repórtame cualquier fallo antes de cerrar.
```

### Prompt 6 — Higiene del Repositorio, Configuración y Dependencias
```text
Lee completamente el archivo `MEJORA_06_CONFIG_Y_REPO_HYGIENE.md` en la raíz del proyecto y ejecuta las tareas de limpieza y validación de configuración.
Restricciones obligatorias:
- NO borres modelos `.onnx`/`.pt` ni datasets anotados (`datasets/coc/jump_gt/`, `logs/yolo/test_split_unseen/`).
- Valida que `main.py --help` y la suite completa de tests sigan pasando tras los cambios.
```