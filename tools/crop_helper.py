
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import PROJECT_ROOT


def main():
    if len(sys.argv) < 2:
        print("Uso: python tools/crop_helper.py <ruta_imagen.png>")
        return

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"No se pudo abrir la imagen: {img_path}")
        return

    os.makedirs(os.path.join(PROJECT_ROOT, "assets", "digits"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "assets", "buttons"), exist_ok=True)

    print("=== ASISTENTE DE RECORTE CLASH BOT ===")
    print("1. Dibuja un rectángulo con el mouse y presiona ENTER o ESPACIO.")
    print("2. Presiona 'c' o ESC en la ventana para salir.")

    while True:
        r = cv2.selectROI("Recortar Template (ENTER=Guardar, ESC=Salir)", img, fromCenter=False, showCrosshair=True)
        x, y, w, h = [int(v) for v in r]
        if w == 0 or h == 0:
            break

        crop = img[y:y + h, x:x + w]
        nombre = input("Guardar como (ej: digits/8.png o buttons/next_btn.png): ").strip()
        if nombre:
            out_path = os.path.join(PROJECT_ROOT, "assets", nombre)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, crop)
            print(f"✅ Guardado en: {out_path} ({w}x{h} px)")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()