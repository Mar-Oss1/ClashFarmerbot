import cv2
import os
import glob

# Ruta de tus plantillas
CARPETA = "assets/buildings"
SALIDA = "logs/template_check"
os.makedirs(SALIDA, exist_ok=True)

archivos = glob.glob(os.path.join(CARPETA, "*.png"))
print(f"Revisando {len(archivos)} plantillas...")

for f in archivos:
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] < 4:
        continue

    alpha = img[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None: continue

    # Recortar igual que en el script principal
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]

    # Escalar al 35% (el tamaño medio que usa el script)
    new_w, new_h = int(w * 0.35), int(h * 0.35)
    if new_w < 5 or new_h < 5: continue

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Guardar para que lo veas
    nombre_salida = os.path.basename(f)
    cv2.imwrite(os.path.join(SALIDA, nombre_salida), resized)

print(f"✅ Listo. Abre la carpeta: {SALIDA}")
print("Revisa si las imágenes son edificios 3D grandes o íconos planos pequeños.")