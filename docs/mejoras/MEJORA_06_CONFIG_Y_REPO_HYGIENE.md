# MEJORA 06 — Higiene del Repositorio, Configuración Validada y Separación de Dependencias

## Objetivo
Dejar el repositorio limpio, seguro (sin seriales de dispositivo ni rutas absolutas hardcodeadas), con validación temprana de `settings.json` y con separación clara entre dependencias de ejecución (CPU ligero con ONNX) y dependencias de entrenamiento (PyTorch / Ultralytics).

## Alcance (Archivos a crear/modificar)
- `.gitignore`
- `.env.example`
- `core/config.py`
- `config/settings.json`
- `requirements-runtime.txt` (nuevo)
- `requirements-train.txt` (nuevo)
- `requirements.txt` (referencia o mantiene compatibilidad)

## Restricciones (NO tocar / NO borrar)
- **NO borrar** ningún peso en `models/` (`.onnx`, `.pt`).
- **NO borrar** ningún dataset ni anotación (`datasets/coc/`, `datasets/coc/jump_gt/`, `datasets/coc_walls*`, `logs/yolo/test_split_unseen/`).
- **NO borrar** `backups/` automáticamente sin confirmación del usuario (solo excluirlo de futuros commits en `.gitignore` y reportar cuánto espacio ocupa).

---

## Tareas Técnicas Detalladas

### 1. Validación de configuración y variables de entorno en `core/config.py`
**Problemas actuales:**
1. `config/settings.json` expone `"serial": "5fba8618"` en texto plano dentro del repo.
2. `TESSERACT_CMD` está hardcodeado a `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3. Hay claves duplicadas o muertas en `settings.json` (`upgrade_walls` duplicada con `enable_wall_upgrade`; `swipe_profiles`, `camera.sign_x/y`, `battle_end.*` no usadas en ejecución).
4. Si una ruta de modelo `.onnx` o un parámetro numérico en `settings.json` tiene un typo, el bot falla a mitad de batalla en lugar de avisar al arrancar.

**Solución requerida:**
1. Soporta override por variables de entorno (sin exigir librería extra; si `python-dotenv` está instalado cárgalo opcionalmente, si no lee `os.getenv`):
   - `COC_DEVICE_SERIAL` -> fallback a `SETTINGS["device"]["serial"]`
   - `TESSERACT_CMD` -> fallback a `r"C:\Program Files\Tesseract-OCR\tesseract.exe"`
2. Crea `.env.example` documentando `COC_DEVICE_SERIAL` y `TESSERACT_CMD`.
3. Añade `validate_settings(settings: dict) -> list[str]` en `core/config.py`:
   - Verifica secciones obligatorias (`device`, `attack`, `rois`, `humanizer`, `economy`).
   - Verifica rangos lógicos (`0 < storage_upgrade_trigger_threshold <= 1.0`, `0 < yolo_conf < 1.0`, `safe_drop_y_min < safe_drop_y_max`).
   - Emite `log.warning(...)` por cada archivo `.onnx` configurado que no exista en disco (sin abortar, para permitir tests offline y fallbacks).
   - Emite `log.warning(...)` si faltan templates críticos en `assets/buttons/` (ej. `attack_btn.png`, `next_btn.png`, `return_home.png`).
4. En `config/settings.json`:
   - Consolida `upgrade_walls` -> mantén solo `enable_wall_upgrade` (verifica que `core/ledger.py` ya soporta `enable_wall_upgrade`).
   - Marca claramente o elimina bloques muertos que no referencia ningún módulo de producción (verifica primero con grep en `core/`, `vision/`, `attack/` y `tests/` antes de quitar cualquier clave).

---

### 2. Separar dependencias de Runtime vs Entrenamiento
**Problema actual:**
`requirements.txt` obliga a instalar `torch`, `torchvision`, `ultralytics` y `segmentation-models-pytorch` (>2.5 GB) incluso en una máquina donde solo se corre inferencia ONNX en CPU.

**Solución requerida:**
1. Crea `requirements-runtime.txt` con lo mínimo para ejecutar el bot:
   ```text
   numpy>=1.24.0
   opencv-python>=4.8.0
   Pillow>=10.0.0
   uiautomator2>=3.0.0
   onnxruntime>=1.17.0
   ```
2. Crea `requirements-train.txt`:
   ```text
   -r requirements-runtime.txt
   torch>=2.0.0
   torchvision>=0.15.0
   ultralytics>=8.0.0
   segmentation-models-pytorch>=0.3.0
   onnx>=1.15.0
   ```
3. Mantén `requirements.txt` apuntando a `-r requirements-train.txt` para no romper flujos previos.

---

### 3. Limpiar `.gitignore`
Asegúrate de que `.gitignore` excluya archivos temporales, dumps e IDE sin tocar código ni assets:
```gitignore
# Entornos y caché Python
__pycache__/
*.py[cod]
venv/
.venv/

# IDE y secretos
.idea/
.env

# Logs, capturas temporales y experimentos
logs/
scratch/
backups/
runs/
.agents/teamwork/
*_contexto.txt
clashfarmerbot_context.md
```

---

## Criterios de Aceptación y Verificación

1. Ejecuta:
```powershell
python -c "from core.config import SETTINGS, validate_settings; print('Warnings:', validate_settings(SETTINGS))"
python main.py --help
python -m unittest discover -s tests
```
2. Confirma que `validate_settings()` reporta advertencias limpias (sin excepciones), `main.py --help` funciona y la suite completa de tests pasa.