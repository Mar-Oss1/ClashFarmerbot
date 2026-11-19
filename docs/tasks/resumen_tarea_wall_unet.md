# Pipeline U-Net (MobileNetV2) para Detección de Muros (`WallSegDetector`)

Este documento resume todos los componentes desarrollados, su verificación y la guía paso a paso para entrenar y poner en producción el modelo U-Net de segmentación de muros en **ClashFarmerbot**.

---

## 1. Resumen de lo Logrado

Se construyó e integró de forma 100% modular el pipeline de segmentación semántica de muros sin alterar la lógica de ataque existente:

### A. Conversor y Preprocesador de Máscaras (`tools/json_to_wall_mask.py`)
- **Entrada:** Lee pares `imagen.png` + `imagen.json` en formato LabelMe exportados por **X-AnyLabeling** con etiqueta única `"wall"`.
- **Soporte Geométrico:** Rasteriza primitivas `polygon` (`cv2.fillPoly`), `rectangle` (`cv2.rectangle`) y `linestrip`/`line` (con grosor configurable, default 8px).
- **Protección de Continuidad Isométrica:**
  - Aplica `cv2.dilate` (kernel 3x3, 1 iteración) **antes** de cualquier redimensionado para garantizar que tramos delgados de muro no desaparezcan al comprimir a resolución de entrenamiento.
  - Cierre morfológico `cv2.MORPH_CLOSE` (kernel 3x3) para sellar micro-fisuras de 1-2px en esquinas e intersecciones.
- **Enmascaramiento de HUD:** Zona de botones/barra inferior (`y >= round(h * 888 / 1080)`) forzada a 0.
- **Partición Train/Val:** División estratificada y reproducible (`--val-ratio 0.15`, mínimo 1 muestra de validación si hay $\ge 5$ imágenes).
- **Estructura Generada:**
  ```text
  datasets/coc_walls_unet/
    ├── train/
    │   ├── images/ (*.png)
    │   └── masks/  (*.png)
    └── val/
        ├── images/ (*.png)
        └── masks/  (*.png)
  ```
- **Auditoría Visual:** Genera imágenes compuestas en `scratch/wall_unet_mask_preview/` con overlay cian semitransparente, contornos verdes y barra superior de telemetría (split, resolución, píxeles de muro y porcentaje de cobertura).
- **Función Exportada:** `build_overlay(img_bgr, mask, text=...)` para reutilización directa en debug visual.

---

### B. Entrenamiento U-Net y Exportación ONNX (`tools/train_wall_unet.py`)
- **Arquitectura:** U-Net con encoder `mobilenet_v2` preentrenado en ImageNet (usando `segmentation-models-pytorch`).
- **Resolución de Red:** `IMG_W = 768`, `IMG_H = 352` (múltiplo exacto de 32, aspecto ~2.18:1 idóneo para 2400x1080).
- **Dataset (`WallMaskDataset`):** Carga en BGR, convierte a RGB, redimensiona imágenes con `cv2.INTER_AREA` y máscaras con `cv2.INTER_NEAREST`, y normaliza con parámetros ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- **Augmentaciones Isométricas Controladas:**
  - Espejo **horizontal únicamente** ($p=0.5$). Nunca vertical ni rotaciones para preservar los ángulos fijos de la perspectiva isométrica ($36^\circ / 144^\circ$).
  - Variación de contraste/brillo ($\alpha \in [0.80, 1.20]$, $\beta \in [-20, 20]$, $p=0.7$).
  - Variación Hue/Sat en espacio HSV ($\Delta \text{Hue} \pm 12$, $\Delta \text{Sat} \pm 25$, $p=0.5$), crucial para generalizar entre biomas (césped, nieve, invierno, muros oscuros/lava).
- **Función de Pérdida Combinada:**
  $$\mathcal{L} = 0.5 \cdot \text{BCEWithLogitsLoss} + 0.5 \cdot \text{DiceLoss}(\text{binary})$$
- **Optimizador & Scheduler:** `AdamW` con `CosineAnnealingLR`.
- **Exportación Automática a ONNX:** Al finalizar el entrenamiento, exporta el mejor checkpoint (`runs/wall_unet_best.pt`) a ONNX (`opset_version=13`, `input_names=['input']`, `output_names=['logits']`) guardándolo en `models/wall_seg.onnx`.
- **Verificación Automática:** Ejecuta inferencia dummy con `onnxruntime` (`CPUExecutionProvider`) confirmando salida de forma `(1, 1, 352, 768)`.
- **Dependencias:** Registradas en `requirements.txt` (`segmentation-models-pytorch`, `torch`, `torchvision`, `onnx`, `onnxruntime`).

---

### C. Módulo de Producción (`vision/wall_seg_detector.py`)
- **Clase `WallSegDetector`:** Singleton / detector de inferencia en tiempo real optimizado para CPU vía `onnxruntime`.
- **Manejo de Errores:** Lanza `FileNotFoundError` descriptivo y amigable si `models/wall_seg.onnx` aún no ha sido entrenado o no está en disco.
- **Pipeline de Inferencia `predict(frame_bgr)`:**
  1. Detecta resolución nativa dinámicamente (`frame_bgr.shape`).
  2. Convierte **BGR $\to$ RGB** (preservando el espacio de color del entrenamiento).
  3. Redimensiona a `(768, 352)` con `cv2.INTER_AREA`.
  4. Aplica normalización ImageNet.
  5. Ejecuta inferencia ONNX (`logits`).
  6. Aplica función sigmoide: $\sigma(z) \ge 0.5$.
  7. Redimensiona la máscara resultante de vuelta a la resolución nativa con `cv2.INTER_NEAREST` para evitar bordes borrosos.
  8. Enmascara el HUD inferior y retorna una máscara binaria `uint8` (`0` / `255`).
- **Telemetría y Debug:** Incluye método `save_debug_overlay(frame_bgr, mask, path)` para auditoría visual en `logs/yolo/wall_seg_debug.png`.

---

### D. Integración en el Optimizador de Saltos (`vision/wall_compartments.py`)
- **Configuración (`config/settings.json`):**
  Se añadió la directiva:
  ```json
  "jump_optimizer": {
    "wall_detection_mode": "unet"
  }
  ```
- **Modos de Detección Alternables:**
  - `"unet"`: Consume directamente la máscara binaria generada por `WallSegDetector.predict()` para el análisis de compartimentos y flood-fill.
  - **Fallback Transparente:** Si el modo es `"unet"` pero `models/wall_seg.onnx` no existe todavía o falla en cargar, el sistema emite un log informativo y conmuta automáticamente a `"raycast"`, impidiendo que el bot se detenga.
  - `"raycast"`: Mantiene el método tradicional de contornos y raycasting de `vision/wall_raycast.py` completamente intacto.
- **Restricciones Cumplidas:**
  - `attack/deployment_strategy.py`: **100% INTACTO Y CONGELADO** (checksum SHA256 verificado).
  - `vision/wall_raycast.py`: **100% PRESERVADO**.
  - `logs/yolo/test_split_unseen/`: **NO TOCADO**.

---

### E. Verificación y Suite de Pruebas Unitarias
Se implementó `tests/test_wall_unet_pipeline.py` (12 pruebas unitarias) cubriendo:
1. Rasterizado de geometrías LabelMe (`polygon`, `rectangle`, `linestrip`).
2. Dilatación y cierre morfológico previo al escalado.
3. Enmascaramiento de HUD de batalla.
4. Partición train/val con mínimo garantizado.
5. Tensor Dataset shapes y consistencia de aumentaciones (sin rotación).
6. Inferencia `WallSegDetector` mockeada, conversión de color y escalado.
7. Comportamiento ante modelo faltante (`FileNotFoundError`).
8. Integración de modos `"unet"` vs `"raycast"` en `wall_compartments.py` con fallback.

**Resultado de la Suite Completa:**
```bash
python -m unittest tests.test_wall_unet_pipeline tests.test_wall_compartments tests.test_wall_raycast tests.test_wall_junctions tools.test_annotate_tool
```
> **42/42 tests PASS (100% OK)** en 0.58s.

---

## 2. Guía Paso a Paso: Lo que debes hacer tú después

Una vez que tengas tus imágenes etiquetadas, sigue este orden para poner el modelo en producción:

### Paso 1: Etiquetar en X-AnyLabeling
- Abre tus capturas de aldeas en **X-AnyLabeling** (puedes usar SAM 2 para agilizar el delineado de muros).
- Utiliza la etiqueta única: `"wall"`.
- Guarda los archivos `.json` junto a sus imágenes `.png` dentro del directorio:
  `datasets/coc_walls_raw/`
  *(ejemplo: `datasets/coc_walls_raw/aldea_01.png` y `datasets/coc_walls_raw/aldea_01.json`)*.

---

### Paso 2: Generar las Máscaras de Entrenamiento
Ejecuta el conversor de máscaras:
```bash
.\.venv\Scripts\python.exe tools/json_to_wall_mask.py --input-dir datasets/coc_walls_raw --output-dir datasets/coc_walls_unet --preview-dir scratch/wall_unet_mask_preview
```
Este comando:
1. Limpia cualquier residuo anterior de `datasets/coc_walls_unet` para evitar fugas entre splits.
2. Rasteriza los polígonos aplicando dilatación y morfología.
3. Divide automáticamente el dataset en `train` (85%) y `val` (15%).
4. Genera las vistas de verificación en `scratch/wall_unet_mask_preview/`.

---

### Paso 3: Auditar las Vistas Previas
- Abre la carpeta `scratch/wall_unet_mask_preview/`.
- Revisa las imágenes generadas. Verás la aldea original con un overlay cian semitransparente sobre los muros y el contorno verde exterior.
- Verifica que las esquinas y uniones de muros estén bien conectadas y que la barra de botones inferior esté completamente limpia (en negro).

---

### Paso 4: Entrenar el Modelo U-Net y Exportar a ONNX
Inicia el entrenamiento usando el entorno virtual `.venv` (donde están instalados PyTorch y `segmentation-models-pytorch`):
```bash
.\.venv\Scripts\python.exe tools/train_wall_unet.py --data-dir datasets/coc_walls_unet --epochs 60 --batch-size 4 --onnx-out models/wall_seg.onnx
```
*Si tienes GPU disponible (CUDA), puedes agregar `--device cuda` para acelerar el proceso.*

Al finalizar las épocas (o activarse early stopping):
1. El mejor checkpoint se guardará en `runs/wall_unet_best.pt`.
2. Se exportará automáticamente a `models/wall_seg.onnx` (opset 13, input: `input`, output: `logits`).
3. El script ejecutará una inferencia de prueba validando que el ONNX sea 100% compatible con CPU en tiempo de ejecución.

---

### Paso 5: Poner en Producción
Asegúrate de que en `config/settings.json` esté configurado:
```json
{
  "jump_optimizer": {
    "wall_detection_mode": "unet"
  }
}
```
¡Listo! En cuanto el bot inicie un ataque o analice compartimentos, `vision/wall_compartments.py` utilizará automáticamente `WallSegDetector` con el nuevo modelo `models/wall_seg.onnx`. Si en cualquier momento deseas volver a la detección por raycasting clásico, simplemente cambia `"wall_detection_mode": "raycast"`.
