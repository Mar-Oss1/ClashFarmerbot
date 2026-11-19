"""
tools/review_prelabels.py
Visor interactivo con OpenCV para revisar, corregir y complementar
las anotaciones YOLO generadas por prelabel_hsv.py o anotaciones manuales.
"""
import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Configuración de clases y colores BGR
CLASS_NAMES = {
    0: "storage_gold",
    1: "storage_elixir",
    2: "storage_dark",
    3: "town_hall",
    4: "collector_gold",
    5: "collector_elixir",
    6: "collector_dark",
}

CLASS_COLORS = {
    0: (0, 215, 255),    # Dorado / BGR
    1: (255, 0, 255),    # Magenta
    2: (128, 0, 128),    # Púrpura oscuro
    3: (255, 100, 0),    # Azul
    4: (0, 255, 255),    # Amarillo
    5: (200, 120, 255),  # Rosa suave
    6: (80, 20, 120),    # Morado/negro extractor oscuro
}

# Variables de estado globales para el callback del ratón
g_drawing = False
g_ix, g_iy = -1, -1
g_cx, g_cy = -1, -1
g_active_cls = 0
g_current_boxes: list[dict] = []
g_dirty = False


def mouse_callback(event, x, y, flags, param):
    global g_drawing, g_ix, g_iy, g_cx, g_cy, g_current_boxes, g_dirty, g_active_cls

    if event == cv2.EVENT_LBUTTONDOWN:
        g_drawing = True
        g_ix, g_iy = x, y
        g_cx, g_cy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if g_drawing:
            g_cx, g_cy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        if g_drawing:
            g_drawing = False
            x1, x2 = min(g_ix, x), max(g_ix, x)
            y1, y2 = min(g_iy, y), max(g_iy, y)
            # Evitar clics accidentales de tamaño insignificante
            if (x2 - x1) >= 8 and (y2 - y1) >= 8:
                g_current_boxes.append({
                    "cls_id": g_active_cls,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })
                g_dirty = True


def load_yolo_labels(label_file: Path, img_w: int, img_h: int) -> list[dict]:
    """Carga cajas desde archivo formato YOLO y convierte a coordenadas en píxeles."""
    boxes = []
    if not label_file.exists():
        return boxes

    try:
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx_norm, cy_norm = float(parts[1]), float(parts[2])
                    w_norm, h_norm = float(parts[3]), float(parts[4])

                    cx = cx_norm * img_w
                    cy = cy_norm * img_h
                    bw = w_norm * img_w
                    bh = h_norm * img_h

                    x1 = int(round(cx - bw / 2.0))
                    y1 = int(round(cy - bh / 2.0))
                    x2 = int(round(cx + bw / 2.0))
                    y2 = int(round(cy + bh / 2.0))

                    # Clamping
                    x1 = max(0, min(img_w, x1))
                    y1 = max(0, min(img_h, y1))
                    x2 = max(0, min(img_w, x2))
                    y2 = max(0, min(img_h, y2))

                    boxes.append({
                        "cls_id": cls_id,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    })
    except Exception as e:
        print(f"Error al leer etiquetas de {label_file}: {e}")

    return boxes


def save_yolo_labels(label_file: Path, boxes: list[dict], img_w: int, img_h: int):
    """Guarda cajas actuales en formato YOLO normalizado."""
    lines = []
    for b in boxes:
        cls_id = b["cls_id"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = (x2 - x1)
        bh = (y2 - y1)

        cx_norm = cx / img_w
        cy_norm = cy / img_h
        w_norm = bw / img_w
        h_norm = bh / img_h

        lines.append(f"{cls_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

    label_file.parent.mkdir(parents=True, exist_ok=True)
    with open(label_file, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
    print(f"Guardado {label_file.name}: {len(lines)} cajas.")


def draw_ui(img: np.ndarray, boxes: list[dict], filename: str, idx: int, total: int) -> np.ndarray:
    """Dibuja las cajas, la caja en progreso y la barra de estado superior."""
    global g_drawing, g_ix, g_iy, g_cx, g_cy, g_active_cls, g_dirty
    disp = img.copy()
    h, w = disp.shape[:2]

    # Dibujar cajas confirmadas
    for b in boxes:
        cls_id = b["cls_id"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")

        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
        label = f"{name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bg_y1 = max(0, y1 - th - 4)
        cv2.rectangle(disp, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(disp, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Dibujar caja en arrastre actual
    if g_drawing:
        x1, x2 = min(g_ix, g_cx), max(g_ix, g_cx)
        y1, y2 = min(g_iy, g_cy), max(g_iy, g_cy)
        color = CLASS_COLORS.get(g_active_cls, (0, 255, 255))
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 1, cv2.LINE_4)

    # Banner superior
    banner_h = 50
    overlay = disp.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, disp, 0.2, 0, disp)

    active_name = CLASS_NAMES.get(g_active_cls, f"cls_{g_active_cls}")
    active_col = CLASS_COLORS.get(g_active_cls, (255, 255, 255))

    dirty_flag = " [MODIFICADO]" if g_dirty else ""
    line1 = f"[{idx + 1}/{total}] {filename}{dirty_flag} | Cajas: {len(boxes)}"
    line2 = f"Clase activa: [{g_active_cls}] {active_name} | Teclas [1-7]: 1:Gold 2:Elixir 3:Dark 4:TH 5:Mine 6:Pump 7:Drill | [s] Guardar | [n] Sig [p] Ant | [d] Undo [q] Salir"

    cv2.putText(disp, line1, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(disp, line2, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, active_col, 1, cv2.LINE_AA)

    return disp


def main():
    global g_active_cls, g_current_boxes, g_dirty

    parser = argparse.ArgumentParser(description="Revisor y anotador interactivo de prelabels YOLO")
    parser.add_argument("--images", type=str, default="datasets/coc/images/raw", help="Directorio con imágenes crudas")
    parser.add_argument("--labels", type=str, default="datasets/coc/labels/raw", help="Directorio con archivos .txt YOLO")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)

    if not images_dir.exists():
        print(f"Error: El directorio de imágenes no existe: {images_dir}")
        sys.exit(1)

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    img_files = sorted([f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions])

    if not img_files:
        print(f"No se encontraron imágenes en: {images_dir}")
        sys.exit(0)

    window_name = "Review YOLO Pre-labels"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, mouse_callback)

    idx = 0
    total = len(img_files)

    current_img_file = img_files[idx]
    current_label_file = labels_dir / f"{current_img_file.stem}.txt"
    current_img = cv2.imread(str(current_img_file))

    if current_img is None:
        print(f"Error al leer imagen inicial: {current_img_file}")
        sys.exit(1)

    h, w = current_img.shape[:2]
    g_current_boxes = load_yolo_labels(current_label_file, w, h)
    g_dirty = False

    while True:
        disp = draw_ui(current_img, g_current_boxes, current_img_file.name, idx, total)
        cv2.imshow(window_name, disp)

        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            # Guardar y salir
            if g_dirty:
                save_yolo_labels(current_label_file, g_current_boxes, w, h)
            break

        elif key == 27:  # Esc
            break

        elif key == ord('1'):
            g_active_cls = 0  # storage_gold
        elif key == ord('2'):
            g_active_cls = 1  # storage_elixir
        elif key == ord('3'):
            g_active_cls = 2  # storage_dark
        elif key == ord('4'):
            g_active_cls = 3  # town_hall
        elif key == ord('5'):
            g_active_cls = 4  # collector_gold
        elif key == ord('6'):
            g_active_cls = 5  # collector_elixir
        elif key == ord('7'):
            g_active_cls = 6  # collector_dark

        elif key == ord('d'):
            # Undo último bbox
            if g_current_boxes:
                g_current_boxes.pop()
                g_dirty = True
        elif key == ord('D'):
            # Borrar todos
            if g_current_boxes:
                g_current_boxes.clear()
                g_dirty = True

        elif key == ord('s'):
            # Guardar y avanzar
            save_yolo_labels(current_label_file, g_current_boxes, w, h)
            g_dirty = False
            if idx < total - 1:
                idx += 1
                current_img_file = img_files[idx]
                current_label_file = labels_dir / f"{current_img_file.stem}.txt"
                current_img = cv2.imread(str(current_img_file))
                h, w = current_img.shape[:2]
                g_current_boxes = load_yolo_labels(current_label_file, w, h)

        elif key == ord('n'):
            # Siguiente (guarda si está modificado)
            if g_dirty:
                save_yolo_labels(current_label_file, g_current_boxes, w, h)
                g_dirty = False
            if idx < total - 1:
                idx += 1
                current_img_file = img_files[idx]
                current_label_file = labels_dir / f"{current_img_file.stem}.txt"
                current_img = cv2.imread(str(current_img_file))
                h, w = current_img.shape[:2]
                g_current_boxes = load_yolo_labels(current_label_file, w, h)

        elif key == ord('p'):
            # Anterior
            if g_dirty:
                save_yolo_labels(current_label_file, g_current_boxes, w, h)
                g_dirty = False
            if idx > 0:
                idx -= 1
                current_img_file = img_files[idx]
                current_label_file = labels_dir / f"{current_img_file.stem}.txt"
                current_img = cv2.imread(str(current_img_file))
                h, w = current_img.shape[:2]
                g_current_boxes = load_yolo_labels(current_label_file, w, h)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
