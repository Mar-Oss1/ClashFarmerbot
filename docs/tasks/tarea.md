# Tarea: Implementar `WallSegDetector` (U-Net) para detección de muros

## Contexto

El proyecto `ClashFarmerbot` actualmente detecta muros mediante raycasting sobre
contornos de OpenCV (`vision/wall_raycast.py`). Vamos a reemplazar/complementar
esa detección con un modelo de segmentación semántica U-Net (MobileNetV2 encoder)
entrenado específicamente para muros, siguiendo el mismo patrón que ya usa
`deploy_zone_finder.onnx` en el proyecto.

El entrenamiento del modelo (etiquetado, conversión de máscaras y entrenamiento)
lo hago yo por separado. Tu tarea es dejar listo todo el código de soporte y la
integración para que, en cuanto yo tenga `models/wall_seg.onnx` entrenado, el
bot pueda usarlo de inmediato sin más cambios de código.

## Archivos a crear

### 1. `tools/json_to_wall_mask.py`

Script que convierte anotaciones poligonales de X-AnyLabeling (formato LabelMe JSON,
label único `"wall"`) en máscaras binarias PNG para entrenar U-Net.

Requisitos funcionales:
- Lee pares `imagen.png` + `imagen.json` desde `datasets/coc_walls_raw/` (configurable
  por `--input-dir`).
- Rasteriza los polígonos (`cv2.fillPoly`), soporta también `rectangle` y `linestrip`
  (grosor fijo ~8px) como shape_type.
- **Aplica un `cv2.dilate` leve (kernel 3x3, 1 iteración) ANTES de cualquier resize**,
  para que los tramos delgados de muro no desaparezcan al escalar la imagen a
  resolución de entrenamiento.
- Aplica un cierre morfológico (`cv2.MORPH_CLOSE`, kernel configurable, default 3)
  para sellar micro-grietas de 1-2px entre polígonos adyacentes en esquinas/cruces.
- Enmascara (pone en 0) la zona del HUD inferior: `y >= round(h * 888/1080)`.
- Divide el dataset en `train`/`val` (ratio configurable, default 0.15, mínimo 1 imagen
  de val si hay >= 5 imágenes).
- Guarda estructura:
  ```
  datasets/coc_walls_unet/
    train/images/*.png
    train/masks/*.png
    val/images/*.png
    val/masks/*.png
  ```
- Genera previews de auditoría visual (overlay cian semitransparente + contorno verde)
  en `scratch/wall_unet_mask_preview/`, con texto superpuesto indicando split, nombre
  de archivo y cantidad de píxeles de muro.
- CLI con `argparse`: `--input-dir`, `--output-dir`, `--preview-dir`, `--val-ratio`,
  `--close-kernel`.
- Debe fallar con mensaje claro si no hay JSONs o no hay pares válidos imagen+JSON.

### 2. `tools/train_wall_unet.py`

Script de entrenamiento U-Net (MobileNetV2 encoder, `segmentation-models-pytorch`)
y exportación a ONNX.

Requisitos funcionales:
- Resolución de entrada: `IMG_W=768`, `IMG_H=352` (múltiplo de 32, aspecto ~2.22:1,
  cercano a 2400x1080).
- `WallMaskDataset(Dataset)`: lee `images/` y `masks/` de un split, resize con
  `INTER_AREA` para imagen e `INTER_NEAREST` para máscara.
- Augmentación (solo si `augment=True`, es decir solo en train):
  - Espejo **horizontal únicamente** (nunca vertical, nunca rotación — la cámara
    isométrica del juego tiene orientación fija), prob 0.5.
  - Variación de contraste/brillo (`alpha` 0.80-1.20, `beta` -20 a 20), prob 0.7.
  - Variación de Hue/Saturation en HSV (hue ±12, sat ±25), prob 0.5 — esto es clave
    para generalizar entre temas visuales del juego (césped, nieve/hielo con muros
    blancos/dorados, tema oscuro con muros negros, lava/desierto).
- Normalización ImageNet estándar (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).
- Modelo: `smp.Unet(encoder_name="mobilenet_v2", encoder_weights="imagenet", in_channels=3, classes=1)`.
- Loss combinado: `0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss(mode="binary", from_logits=True)`
  (el muro ocupa ~5% de los píxeles de pantalla, por lo que BCE sola generaliza mal).
- Optimizer `AdamW`, scheduler `CosineAnnealingLR`.
- Guarda el mejor checkpoint por Dice score de validación en `runs/wall_unet_best.pt`.
- Al finalizar, carga el mejor checkpoint, exporta a ONNX (`opset_version=13`,
  `input_names=["input"]`, `output_names=["logits"]`) en la ruta de `--onnx-out`
  (default `models/wall_seg.onnx`).
- Verifica el export corriendo una inferencia dummy con `onnxruntime`
  (`CPUExecutionProvider`) y reporta shape de salida y mejor Dice alcanzado.
- CLI: `--data-dir`, `--onnx-out`, `--epochs` (default 60), `--batch-size` (default 4).
- Agregar a `requirements.txt` (o indicar en un comentario al final del script):
  `segmentation-models-pytorch`, `torch`, `torchvision`, `onnx`, `onnxruntime`.

### 3. `vision/wall_seg_detector.py` (nuevo módulo, clase `WallSegDetector`)

Este es el componente que el bot usará en producción, reemplazando/complementando
`wall_raycast.py`.

Requisitos funcionales:
- Carga `models/wall_seg.onnx` con `onnxruntime` (`CPUExecutionProvider`) en el
  constructor; debe fallar con un error claro y descriptivo si el archivo no existe
  (para no romper el bot si aún no he entrenado el modelo).
- Método principal `predict(frame_bgr: np.ndarray) -> np.ndarray` que:
  1. Recibe el frame completo a resolución nativa (2400x1080, pero no asumir
     tamaño fijo — tomarlo de `frame_bgr.shape`).
  2. Convierte BGR→RGB (el dataset de entrenamiento usa RGB — **es crítico mantener
     el mismo orden de canales que en entrenamiento o la inferencia será basura
     silenciosa sin error visible**).
  3. Resize a `(768, 352)` con `INTER_AREA`.
  4. Normaliza con la misma normalización ImageNet usada en entrenamiento.
  5. Corre inferencia ONNX.
  6. Aplica `sigmoid` a los logits de salida.
  7. Umbraliza en `>= 0.5` para obtener máscara binaria.
  8. Redimensiona la máscara de vuelta a la resolución original del frame
     (`INTER_NEAREST`, para no introducir grises/bordes difusos en los bordes del muro).
  9. Devuelve la máscara binaria (`uint8`, valores 0/255) a resolución nativa.
- Mantener las constantes `IMG_W=768`, `IMG_H=352` y los valores de normalización
  como constantes de módulo, exactamente iguales a las usadas en
  `tools/train_wall_unet.py` (si es posible, importarlas de un módulo común en vez
  de duplicarlas, para evitar que diverjan con el tiempo).
- Agregar un método o flag para guardar/loggear la máscara generada (debug visual),
  reusando si es posible la lógica de overlay cian de `build_overlay` en
  `tools/json_to_wall_mask.py`.

### 4. Integración en `vision/wall_compartments.py`

- Agregar una opción de configuración (nueva clave en `config/settings.json`, por
  ejemplo `"wall_detection_mode": "raycast" | "unet"`) para poder alternar entre
  el método actual de raycasting y el nuevo `WallSegDetector` sin romper lo que ya
  funciona.
- Cuando el modo sea `"unet"`, el flood-fill de compartimentos debe consumir la
  máscara binaria devuelta por `WallSegDetector.predict()` en vez de los contornos
  de `wall_raycast.py`.
- No remover el código de raycasting existente — debe quedar como fallback
  seleccionable, no eliminado.

## Qué NO hacer

- No etiquetar ni generar el dataset de imágenes — eso lo hago yo manualmente en
  X-AnyLabeling.
- No ejecutar el entrenamiento — solo dejar los scripts listos y probados con datos
  dummy/sintéticos si hace falta validar que corren sin errores.
- No borrar ni modificar `vision/wall_raycast.py`.

## Entregable esperado

1. Los 3 archivos nuevos (`tools/json_to_wall_mask.py`, `tools/train_wall_unet.py`,
   `vision/wall_seg_detector.py`) funcionando y sin errores de sintaxis/import.
2. La integración en `vision/wall_compartments.py` con el flag de configuración.
3. Un resumen breve de los cambios y de cómo yo debo correr el pipeline una vez
   tenga las imágenes etiquetadas (orden de comandos: etiquetar → `json_to_wall_mask.py`
   → revisar previews → `train_wall_unet.py` → activar `wall_detection_mode: "unet"`
   en `config/settings.json`).
