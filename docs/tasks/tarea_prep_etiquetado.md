# Tarea: Preparar el entorno para etiquetado real (sin tocar lógica del pipeline)

## Contexto

El pipeline de `WallSegDetector` (U-Net) ya está implementado y probado con
datos sintéticos de prueba (3 imágenes dummy generadas durante el simulacro
end-to-end). Yo voy a etiquetar manualmente 25-30 capturas reales en
X-AnyLabeling. Antes de que yo empiece, necesito que limpies los residuos de
esa prueba para que no se mezclen con mi dataset real.

## Qué hacer

1. **Verifica si X-AnyLabeling está instalado** en el sistema (busca el
   ejecutable, un acceso directo, o intenta localizar la instalación en rutas
   típicas de instalación de programas en Windows). Si no está instalado,
   descárgalo e instálalo desde el repositorio oficial
   (`https://github.com/CVHub520/X-AnyLabeling`, sección de releases, versión
   para Windows). Si ya está instalado, confírmalo y no reinstales nada.

2. **Verifica y elimina** cualquier archivo de prueba sintética que haya
   quedado en:
   - `datasets/coc_walls_raw/` (las imágenes/JSON dummy del simulacro, si
     siguen ahí)
   - `datasets/coc_walls_unet/train/images/`, `train/masks/`,
     `val/images/`, `val/masks/`
   - `scratch/wall_unet_mask_preview/`
   - `runs/wall_unet_best.pt`
   - `models/wall_seg.onnx`

3. **Confirma que las carpetas de destino existen y están vacías** y listas
   para recibir:
   - `datasets/coc_walls_raw/` → aquí voy a colocar mis `.png` + `.json` reales
     de X-AnyLabeling.

4. **No ejecutes ningún script de conversión ni entrenamiento** — yo correré
   `tools/json_to_wall_mask.py` y `tools/train_wall_unet.py` manualmente una
   vez termine de etiquetar.

5. **No toques** `vision/wall_raycast.py`, `vision/wall_seg_detector.py`,
   `core/wall_upgrader.py` ni ningún otro archivo de lógica — esta tarea es
   solo limpieza de carpetas de datos e instalación/verificación de la
   herramienta de etiquetado.

6. Al terminar, dame un resumen corto confirmando: si X-AnyLabeling ya estaba
   instalado o se instaló, y que las carpetas de `datasets/coc_walls_raw/` y
   `datasets/coc_walls_unet/` quedaron vacías y listas.

## Qué NO hacer

- No generar nuevas imágenes de prueba.
- No modificar código de producción.
- No correr entrenamiento ni conversión de máscaras.
