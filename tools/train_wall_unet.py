"""
tools/train_wall_unet.py
Script de entrenamiento U-Net (MobileNetV2 encoder vía segmentation-models-pytorch)
y exportación a ONNX para la segmentación semántica de muros en Clash of Clans.

Constantes de módulo:
- IMG_W = 768, IMG_H = 352 (múltiplo de 32, aspecto ~2.18:1 adaptado a 2400x1080)
- Normalización ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

Restricciones de augmentación isométrica:
- Flip horizontal únicamente (nunca vertical ni rotación por perspectiva isométrica fija)
- Variación de contraste y brillo (alpha 0.80-1.20, beta -20 a 20)
- Variación de Hue y Saturación en HSV (hue +-12, sat +-25) para temas visuales (nieve, lava, etc.)

Exportación ONNX:
- opset_version=13, input_names=['input'], output_names=['logits'] -> models/wall_seg.onnx
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Agregar raíz del proyecto al path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(FILE_DIR) in {"tools", "scripts", "bin"}:
    PROJECT_ROOT = os.path.dirname(FILE_DIR)
else:
    PROJECT_ROOT = FILE_DIR

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _resolve_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_str))


# Logging modular con fallback transparente para ejecución standalone (ej. Google Colab)
try:
    from core.logger import get_logger

    log = get_logger("train_wall_unet")
except ImportError:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("train_wall_unet")

# ==============================================================================
# Constantes de módulo compartidas
# ==============================================================================
IMG_W: int = 768
IMG_H: int = 352
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ==============================================================================
# Dataset y Augmentación
# ==============================================================================
def apply_augmentations(
    img_rgb: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica augmentaciones respetando las restricciones isométricas de Clash of Clans:
    - Espejo horizontal únicamente (prob 0.5)
    - RandomBrightnessContrast (prob 0.7, alpha 0.70-1.30, beta -35 a 35)
    - HueSaturationValue (prob 0.7, hue +-20, sat +-40, val +-30) para sceneries y niveles de muro
    - CoarseDropout / Cutout (prob 0.5, 4-12 parches rectangulares de 12-32 px para robustez ante tropas/humo)
    NUNCA rota ni voltea verticalmente debido a la perspectiva isométrica fija.
    """
    h, w = img_rgb.shape[:2]

    # 1. Flip horizontal únicamente
    if random.random() < 0.5:
        img_rgb = cv2.flip(img_rgb, 1)
        mask = cv2.flip(mask, 1)

    # 2. RandomBrightnessContrast
    if random.random() < 0.7:
        alpha = random.uniform(0.70, 1.30)
        beta = random.uniform(-35.0, 35.0)
        img_rgb = np.clip(alpha * img_rgb.astype(np.float32) + beta, 0, 255).astype(
            np.uint8
        )

    # 3. HueSaturationValue (HSV)
    if random.random() < 0.7:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        dh = random.uniform(-20.0, 20.0)
        ds = random.uniform(-40.0, 40.0)
        dv = random.uniform(-30.0, 30.0)
        hsv[..., 0] = (hsv[..., 0] + dh) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] + ds, 0.0, 255.0)
        hsv[..., 2] = np.clip(hsv[..., 2] + dv, 0.0, 255.0)
        img_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 4. CoarseDropout (Cutout de parches pequeños sin alterar la máscara)
    if random.random() < 0.5:
        num_holes = random.randint(4, 12)
        for _ in range(num_holes):
            hole_h = random.randint(12, min(32, h))
            hole_w = random.randint(12, min(32, w))
            y1 = random.randint(0, max(0, h - hole_h))
            x1 = random.randint(0, max(0, w - hole_w))
            fill_val = random.choice([0, 128, random.randint(0, 255)])
            img_rgb[y1 : y1 + hole_h, x1 : x1 + hole_w] = fill_val

    return img_rgb, mask


class WallMaskDataset(Dataset):
    """
    Dataset para segmentación semántica de muros.
    Carga pares imagen / máscara, redimensiona a (IMG_W, IMG_H),
    aplica augmentaciones y normaliza con estadísticas ImageNet.
    """

    def __init__(
        self,
        split_dir: str,
        augment: bool = False,
        target_size: Tuple[int, int] = (IMG_W, IMG_H),
    ):
        self.split_dir = split_dir
        self.augment = augment
        self.target_w, self.target_h = target_size

        self.images_dir = os.path.join(split_dir, "images")
        self.masks_dir = os.path.join(split_dir, "masks")

        self.pairs: List[Tuple[str, str]] = []
        if os.path.isdir(self.images_dir) and os.path.isdir(self.masks_dir):
            valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            img_files = {
                os.path.splitext(f)[0]: os.path.join(self.images_dir, f)
                for f in os.listdir(self.images_dir)
                if os.path.splitext(f)[1].lower() in valid_exts
            }
            mask_files = {
                os.path.splitext(f)[0]: os.path.join(self.masks_dir, f)
                for f in os.listdir(self.masks_dir)
                if os.path.splitext(f)[1].lower() in valid_exts
            }

            common_stems = sorted(set(img_files.keys()) & set(mask_files.keys()))
            for stem in common_stems:
                self.pairs.append((img_files[stem], mask_files[stem]))

        log.info(
            "WallMaskDataset cargado desde '%s': %d pares encontrados (augment=%s)",
            split_dir,
            len(self.pairs),
            self.augment,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        # Lectura de imagen (BGR -> RGB obligatorio)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise RuntimeError(f"No se pudo cargar la imagen: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Lectura de máscara en escala de grises
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise RuntimeError(f"No se pudo cargar la máscara: {mask_path}")

        # Redimensionamiento: INTER_AREA para imagen, INTER_NEAREST para máscara
        img_resized = cv2.resize(
            img_rgb, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA
        )
        mask_resized = cv2.resize(
            mask_gray, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST
        )

        # Binarización a 0 / 1
        mask_bin = (mask_resized > 127).astype(np.uint8)

        # Augmentaciones (solo train)
        if self.augment:
            img_resized, mask_bin = apply_augmentations(img_resized, mask_bin)

        # Normalización ImageNet
        img_float = img_resized.astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        img_norm = (img_float - mean) / std

        # Transposición a BCHW: (H, W, C) -> (C, H, W)
        tensor_img = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        tensor_mask = (
            torch.from_numpy(mask_bin.astype(np.float32)).unsqueeze(0).float()
        )

        return tensor_img, tensor_mask


# ==============================================================================
# Métricas y Funciones de Pérdida
# ==============================================================================
def compute_dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """Calcula el coeficiente Dice binario a partir de logits de salida."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum().item()
    total = preds_flat.sum().item() + targets_flat.sum().item()

    if total == 0.0:
        return 1.0
    return float((2.0 * intersection + smooth) / (total + smooth))


def build_loss_criterion():
    """
    Construye el loss combinado:
    0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss(mode="binary", from_logits=True)
    """
    import segmentation_models_pytorch as smp

    bce_fn = nn.BCEWithLogitsLoss()
    dice_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def criterion(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = bce_fn(logits, targets)
        loss_dice = dice_fn(logits, targets)
        return 0.5 * loss_bce + 0.5 * loss_dice

    return criterion


# ==============================================================================
# Exportación ONNX y Validación
# ==============================================================================
def export_to_onnx(
    model: nn.Module,
    onnx_path: str,
    img_w: int = IMG_W,
    img_h: int = IMG_H,
    opset_version: int = 13,
) -> bool:
    """
    Exporta el modelo U-Net a ONNX con opset 13 y nombres estandarizados:
    input_names=['input'], output_names=['logits'].
    Mueve explícitamente el modelo y el tensor dummy a CPU para garantizar
    exportación limpia y portable tanto desde GPU CUDA como desde CPU.
    """
    model.eval()
    model_cpu = model.to("cpu")
    dummy_input = torch.randn(1, 3, img_h, img_w, device="cpu")

    dest_abs = _resolve_path(onnx_path)
    os.makedirs(os.path.dirname(dest_abs), exist_ok=True)

    log.info(
        "Exportando modelo U-Net a ONNX en '%s' (resolución: %dx%d, opset: %d)...",
        dest_abs,
        img_w,
        img_h,
        opset_version,
    )

    try:
        try:
            torch.onnx.export(
                model_cpu,
                dummy_input,
                dest_abs,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["logits"],
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                model_cpu,
                dummy_input,
                dest_abs,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["logits"],
            )

        file_size_mb = os.path.getsize(dest_abs) / (1024 * 1024)
        log.info(
            "Exportación ONNX completada exitosamente (tamaño: %.2f MB)",
            file_size_mb,
        )
        return True
    except Exception as e:
        log.exception("Fallo al exportar modelo a ONNX: %s", e)
        return False


def verify_onnx_inference(
    onnx_path: str,
    img_w: int = IMG_W,
    img_h: int = IMG_H,
) -> Tuple[bool, Optional[Tuple[int, ...]]]:
    """
    Ejecuta una inferencia dummy con onnxruntime (CPUExecutionProvider)
    para certificar la validez del artefacto generado.
    """
    dest_abs = _resolve_path(onnx_path)
    if not os.path.isfile(dest_abs):
        log.error("Archivo ONNX no encontrado para verificación: %s", dest_abs)
        return False, None

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            dest_abs, providers=["CPUExecutionProvider"]
        )
        inp_names = [i.name for i in sess.get_inputs()]
        out_names = [o.name for o in sess.get_outputs()]
        log.info(
            "InferenceSession cargada. Inputs: %s, Outputs: %s",
            inp_names,
            out_names,
        )

        dummy_inp = np.random.randn(1, 3, img_h, img_w).astype(np.float32)
        outputs = sess.run(None, {inp_names[0]: dummy_inp})
        out_shape = tuple(outputs[0].shape)
        log.info("Inferencia dummy exitosa. Forma del tensor de salida: %s", out_shape)
        return True, out_shape
    except Exception as e:
        log.exception("Error al verificar inferencia ONNX con onnxruntime: %s", e)
        return False, None


# ==============================================================================
# Función de Entrenamiento
# ==============================================================================
def train_wall_unet(
    data_dir: str = "datasets/coc_walls_unet",
    onnx_out: str = "models/wall_seg.onnx",
    epochs: int = 60,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: Optional[str] = None,
    checkpoint_out: str = "runs/wall_unet_best.pt",
    pt_out: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """
    Entrena el modelo U-Net (MobileNetV2 encoder) con MobileNetV2 y exporta a ONNX.
    Detecta automáticamente GPU CUDA o CPU si device=None.
    """
    import segmentation_models_pytorch as smp

    resolved_data_dir = _resolve_path(data_dir)
    train_split = os.path.join(resolved_data_dir, "train")
    val_split = os.path.join(resolved_data_dir, "val")

    # Validación de existencia de dataset
    train_dataset = WallMaskDataset(train_split, augment=True)
    if len(train_dataset) == 0:
        log.critical(
            "No se encontraron datos de entrenamiento en '%s'. "
            "Asegúrate de ejecutar 'python tools/json_to_wall_mask.py' una vez tengas anotaciones.",
            train_split,
        )
        return 1

    val_dataset = WallMaskDataset(val_split, augment=False)
    has_val = len(val_dataset) > 0
    if not has_val:
        log.warning(
            "No se encontraron muestras de validación en '%s'. Se evaluará sobre el split de train.",
            val_split,
        )

    # Dispositivo: auto-detección CUDA si no se especifica
    if device is None:
        target_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        target_device = torch.device(device)
    log.info("Entrenando en dispositivo: %s", target_device)

    # Dataloaders
    use_cuda = target_device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=False,
        pin_memory=use_cuda,
    )
    eval_dataset = val_dataset if has_val else train_dataset
    val_loader = DataLoader(
        eval_dataset,
        batch_size=min(batch_size, len(eval_dataset)),
        shuffle=False,
        pin_memory=use_cuda,
    )

    # Construcción de modelo
    log.info("Inicializando modelo U-Net (encoder=mobilenet_v2, classes=1)...")
    try:
        model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
    except Exception as e:
        log.warning(
            "No se pudieron descargar pesos pre-entrenados ImageNet (%s). Iniciando con pesos aleatorios.",
            e,
        )
        model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

    model.to(target_device)

    # Pérdida, Optimizador y Scheduler
    loss_fn = build_loss_criterion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_epochs = 1 if dry_run else epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs), eta_min=1e-6
    )

    best_dice = -1.0
    checkpoint_abs = _resolve_path(checkpoint_out)
    os.makedirs(os.path.dirname(checkpoint_abs), exist_ok=True)

    log.info(
        "Iniciando ciclo de entrenamiento (%d épocas, batch_size=%d, lr=%.1e)...",
        total_epochs,
        batch_size,
        lr,
    )

    for epoch in range(1, total_epochs + 1):
        # 1. Fase Train
        model.train()
        train_loss_acc = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(target_device, non_blocking=use_cuda)
            masks = masks.to(target_device, non_blocking=use_cuda)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_acc += loss.item() * imgs.size(0)

        scheduler.step()
        avg_train_loss = train_loss_acc / len(train_dataset)

        # 2. Fase Validación
        model.eval()
        val_loss_acc = 0.0
        val_dice_acc = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(target_device, non_blocking=use_cuda)
                masks = masks.to(target_device, non_blocking=use_cuda)

                logits = model(imgs)
                loss = loss_fn(logits, masks)
                val_loss_acc += loss.item() * imgs.size(0)

                dice = compute_dice_score(logits, masks)
                val_dice_acc += dice
                num_val_batches += 1

        avg_val_loss = val_loss_acc / len(eval_dataset)
        avg_val_dice = (
            val_dice_acc / num_val_batches if num_val_batches > 0 else 0.0
        )

        log.info(
            "Época [%3d/%3d] - Train Loss: %.4f | Val Loss: %.4f | Val Dice: %.4f",
            epoch,
            total_epochs,
            avg_train_loss,
            avg_val_loss,
            avg_val_dice,
        )

        # Guardar mejor modelo por Dice
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            state_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "img_size": (IMG_W, IMG_H),
            }
            torch.save(state_dict, checkpoint_abs)
            log.info(
                "⭐ Nuevo mejor checkpoint guardado en '%s' (Dice: %.4f)",
                checkpoint_abs,
                best_dice,
            )
            if pt_out:
                pt_dest_abs = _resolve_path(pt_out)
                os.makedirs(os.path.dirname(pt_dest_abs), exist_ok=True)
                torch.save(state_dict, pt_dest_abs)
                log.info("Checkpoint .pt también guardado en: '%s'", pt_dest_abs)

    # 3. Exportación a ONNX desde el mejor checkpoint
    log.info("Cargando mejor checkpoint desde '%s' para exportar...", checkpoint_abs)
    if os.path.exists(checkpoint_abs):
        ckpt = torch.load(checkpoint_abs, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        log.warning(
            "No se encontró checkpoint guardado; exportando pesos actuales del modelo."
        )

    export_ok = export_to_onnx(model, onnx_out, img_w=IMG_W, img_h=IMG_H)
    if not export_ok:
        log.error("Falló la exportación ONNX del modelo U-Net.")
        return 1

    # 4. Verificación de inferencia ONNX
    verify_ok, out_shape = verify_onnx_inference(onnx_out, img_w=IMG_W, img_h=IMG_H)
    if not verify_ok:
        log.error("La verificación de inferencia ONNX falló.")
        return 1

    log.info("=================================================================")
    log.info("Entrenamiento y exportación U-Net completados con éxito.")
    log.info("Mejor Dice Score alcanzado: %.4f", best_dice)
    log.info("Modelo ONNX guardado en: %s", onnx_out)
    if pt_out:
        log.info("Checkpoint .pt guardado en: %s", pt_out)
    log.info("Forma de salida de inferencia verificada: %s", out_shape)
    log.info("=================================================================")
    return 0


# ==============================================================================
# CLI Principal
# ==============================================================================
def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrenar modelo U-Net (MobileNetV2) para segmentación de muros y exportar a ONNX"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets/coc_walls_unet",
        help="Directorio del dataset generado con json_to_wall_mask.py (default: datasets/coc_walls_unet)",
    )
    parser.add_argument(
        "--onnx-out",
        type=str,
        default="models/wall_seg.onnx",
        help="Ruta de destino para el modelo ONNX exportado (default: models/wall_seg.onnx)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Número total de épocas de entrenamiento (default: 60)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Tamaño de lote para el entrenamiento (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Tasa de aprendizaje inicial (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo ('cpu' o 'cuda', default: auto)",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=str,
        default="runs/wall_unet_best.pt",
        help="Ruta donde guardar el mejor checkpoint PyTorch (default: runs/wall_unet_best.pt)",
    )
    parser.add_argument(
        "--pt-out",
        type=str,
        default=None,
        help="Ruta opcional para guardar una copia del mejor checkpoint .pt (ej. Google Drive)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta solo 1 época para validar el flujo completo sin entrenar a fondo",
    )
    return parser


def main() -> int:
    parser = build_cli_parser()
    args = parser.parse_args()
    return train_wall_unet(
        data_dir=args.data_dir,
        onnx_out=args.onnx_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        checkpoint_out=args.checkpoint_out,
        pt_out=args.pt_out,
        dry_run=args.dry_run,
    )


# Dependencias requeridas:
# segmentation-models-pytorch>=0.3.0
# torch>=2.0.0
# torchvision>=0.15.0
# onnx>=1.15.0
# onnxruntime>=1.17.0
# onnxscript>=0.7.0

if __name__ == "__main__":
    sys.exit(main())
