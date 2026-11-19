import os
from PIL import Image

# Asegúrate de que esta sea la ruta correcta
CARPETA = r"C:\ClashFarmerbot\assets\buildings"


def renombrar_y_convertir():
    if not os.path.exists(CARPETA):
        print(f"❌ La carpeta {CARPETA} no existe. Créala y pon tus imágenes ahí.")
        return

    contador = 0
    for archivo in os.listdir(CARPETA):
        # Ignorar si ya está en el formato correcto
        if archivo.startswith("storage_") or archivo.startswith("collector_"):
            continue

        # Obtener el nombre sin extensión
        nombre_base, extension = os.path.splitext(archivo)
        nuevo_nombre = None

        # --- MAPEO DE ALMACENES ---
        if "Dark_Elixir_Storage" in nombre_base:
            nivel = nombre_base.replace("Dark_Elixir_Storage", "").strip()
            nuevo_nombre = f"storage_dark_lvl{nivel}.png"
        elif "Elixir_Storage" in nombre_base:
            nivel = nombre_base.replace("Elixir_Storage", "").strip()
            nuevo_nombre = f"storage_elixir_lvl{nivel}.png"
        elif "Gold_Storage" in nombre_base:
            nivel = nombre_base.replace("Gold_Storage", "").strip()
            nuevo_nombre = f"storage_gold_lvl{nivel}.png"

        # --- MAPEO DE RECOLECTORES ---
        elif "Dark_Elixir_Drill" in nombre_base:
            nivel = nombre_base.replace("Dark_Elixir_Drill", "").strip()
            nuevo_nombre = f"collector_dark_lvl{nivel}.png"
        elif "Elixir_Collector" in nombre_base:
            nivel = nombre_base.replace("Elixir_Collector", "").strip()
            nuevo_nombre = f"collector_elixir_lvl{nivel}.png"
        elif "Gold_Mine" in nombre_base:
            nivel = nombre_base.replace("Gold_Mine", "").strip()
            nuevo_nombre = f"collector_gold_lvl{nivel}.png"

        # Si encontramos una coincidencia, procesamos
        if nuevo_nombre:
            ruta_vieja = os.path.join(CARPETA, archivo)
            ruta_nueva = os.path.join(CARPETA, nuevo_nombre)

            # Si el archivo es .webp, lo convertimos a .png real
            if extension.lower() == '.webp':
                try:
                    img = Image.open(ruta_vieja).convert("RGBA")
                    img.save(ruta_nueva, "PNG")
                    os.remove(ruta_vieja)  # Borramos el .webp original
                    print(f"✅ Convertido y renombrado: {archivo} -> {nuevo_nombre}")
                except Exception as e:
                    print(f"❌ Error convirtiendo {archivo}: {e}")
            else:
                # Si ya es png (o similar), solo lo renombramos
                os.rename(ruta_vieja, ruta_nueva)
                print(f"✅ Renombrado: {archivo} -> {nuevo_nombre}")

            contador += 1

    print(f"\n🎉 Proceso terminado. {contador} archivos procesados.")


if __name__ == "__main__":
    renombrar_y_convertir()