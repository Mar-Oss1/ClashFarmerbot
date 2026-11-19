"""
tools/split_dataset.py
Distribuye capturas y etiquetas anotadas desde 'raw' hacia train/val de YOLO.
IDEMPOTENTE POR CONTRATO: purga train/val y sus .cache ANTES de copiar, y
valida invariantes DESPUÉS. Correrlo N veces siempre deja el disco limpio.
"""
import argparse
import glob
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import get_logger

log = get_logger("split_dataset")


def _dirs(dest):
    return {
        "img_train": os.path.join(dest, "images", "train"),
        "img_val":   os.path.join(dest, "images", "val"),
        "lbl_train": os.path.join(dest, "labels", "train"),
        "lbl_val":   os.path.join(dest, "labels", "val"),
    }


def purge_split_dirs(dest):
    """[FIX #1 y #2] Vaciado INCONDICIONAL de train/val + caché de Ultralytics.
    La caché vive en <dest>/labels/{train,val}.cache (hermana de las carpetas,
    no hija), por eso rmtree de las carpetas NO la alcanza: hay que borrarla a mano."""
    d = _dirs(dest)
    for path in d.values():
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    for cache in (os.path.join(dest, "labels", "train.cache"),
                  os.path.join(dest, "labels", "val.cache")):
        if os.path.exists(cache):
            os.remove(cache)
            log.info("Purgada caché obsoleta: %s", cache)


def validate_label_format(lbl_path, bad):
    """[FIX #4] Atrapa .txt truncados/corruptos (crash del revisor a mitad de 's')
    ANTES de quemar horas de CPU. Valida estructura, no semántica de clase
    (el rango clase<nc lo vela YOLO en train, para no acoplar split a data.yaml)."""
    stem = Path(lbl_path).stem
    with open(lbl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            tok = line.split()
            if len(tok) != 5:
                bad.append(f"{stem}.txt:{i} -> {len(tok)} campos (espera 5)")
                continue
            try:
                cls = int(tok[0])
                vals = [float(x) for x in tok[1:]]
            except ValueError:
                bad.append(f"{stem}.txt:{i} -> campos no numéricos: {line!r}")
                continue
            if cls < 0 or any(v < 0.0 or v > 1.0 for v in vals):
                bad.append(f"{stem}.txt:{i} -> fuera de rango: {line!r}")


def collect_valid_pairs(images_src, labels_src, labels_pre, allow_prelabel):
    images = sorted(
        glob.glob(os.path.join(images_src, "*.png")) +
        glob.glob(os.path.join(images_src, "*.jpg"))
    )
    pairs, using_pre, bad = [], [], []
    for img_p in images:
        stem = Path(img_p).stem
        lbl_p = os.path.join(labels_src, f"{stem}.txt")
        src_kind = "raw"
        if not (os.path.exists(lbl_p) and os.path.getsize(lbl_p) > 0):
            # [FIX #3] fallback a labels_pre SOLO si el humano lo autoriza, y loggeado.
            alt = os.path.join(labels_pre, f"{stem}.txt")
            if allow_prelabel and os.path.exists(alt) and os.path.getsize(alt) > 0:
                lbl_p, src_kind = alt, "PRELABEL(sucia)"
                using_pre.append(stem)
            else:
                continue
        validate_label_format(lbl_p, bad)
        pairs.append((img_p, lbl_p, src_kind))
    if bad:
        log.error("Etiquetas corruptas/truncadas detectadas (re-guarda esas aldeas):")
        for b in bad:
            log.error("   %s", b)
        sys.exit(1)  # fail-fast: no entreno con veneno geométrico
    if using_pre:
        log.warning("⚠️  %d imágenes ENTRENAN con label PRE-ETIQUETADA SIN REVISAR: %s",
                    len(using_pre), ", ".join(using_pre))
    return pairs


def assert_invariants(dest):
    """[FIX #4] Contrato de salida: el disco REAL debe cumplir paridad y cero solape.
    Esto es lo que habría gritado '43/9' en el segundo que ocurrió."""
    d = _dirs(dest)
    sets = {}
    for key, path in d.items():
        ext = "*.png" if key.startswith("img") else "*.txt"
        sets[key] = {Path(p).stem for p in glob.glob(os.path.join(path, ext))}
    if sets["img_train"] != sets["lbl_train"]:
        raise RuntimeError(f"Paridad rota TRAIN: imgs-lbls={sets['img_train']-sets['lbl_train']}, "
                           f"lbls-imgs={sets['lbl_train']-sets['img_train']}")
    if sets["img_val"] != sets["lbl_val"]:
        raise RuntimeError(f"Paridad rota VAL: imgs-lbls={sets['img_val']-sets['lbl_val']}, "
                           f"lbls-imgs={sets['lbl_val']-sets['img_val']}")
    overlap = sets["img_train"] & sets["img_val"]
    if overlap:
        raise RuntimeError(f"DATA LEAKAGE: {len(overlap)} aldeas en train Y val: "
                           f"{sorted(overlap)[:5]}")
    return len(sets["img_train"]), len(sets["img_val"])


def main():
    ap = argparse.ArgumentParser(description="Dividir dataset CoC en train/val (idempotente)")
    ap.add_argument("--images-src", default="datasets/coc/images/raw")
    ap.add_argument("--labels-src", default="datasets/coc/labels/raw")
    ap.add_argument("--dest", default="datasets/coc")
    ap.add_argument("--val-count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-prelabel", action="store_true",
                    help="Permitir que imágenes sin label revisada usen el borrador de labels_pre (loggeado)")
    args = ap.parse_args()

    random.seed(args.seed)
    labels_pre = os.path.join(args.dest, "labels_pre")  # [FIX #3] ruta derivada, no hardcodeada

    purge_split_dirs(args.dest)  # [FIX #1 y #2] SIEMPRE, sin flag

    pairs = collect_valid_pairs(args.images_src, args.labels_src, labels_pre, args.allow_prelabel)
    if not pairs:
        log.error("No hay pares válidos en %s / %s. Etiqueta primero con review_prelabels.py.",
                  args.images_src, args.labels_src)
        sys.exit(1)
    log.info("Pares válidos encontrados: %d", len(pairs))

    random.shuffle(pairs)
    cap = max(1, int(len(pairs) * 0.25))
    val_n = min(args.val_count, cap)
    if args.val_count > cap:  # [FIX] techo del 25% ya NO es silencioso
        log.warning("--val-count %d capado a %d (25%% de %d pares).", args.val_count, val_n, len(pairs))
    val_pairs, train_pairs = pairs[:val_n], pairs[val_n:]
    log.info("Partición deseada: TRAIN=%d, VAL=%d", len(train_pairs), len(val_pairs))

    d = _dirs(args.dest)
    for img_p, lbl_p, _ in train_pairs:
        shutil.copy2(img_p, os.path.join(d["img_train"], os.path.basename(img_p)))
        shutil.copy2(lbl_p, os.path.join(d["lbl_train"], os.path.basename(lbl_p)))
    for img_p, lbl_p, _ in val_pairs:
        shutil.copy2(img_p, os.path.join(d["img_val"], os.path.basename(img_p)))
        shutil.copy2(lbl_p, os.path.join(d["lbl_val"], os.path.basename(lbl_p)))

    n_train, n_val = assert_invariants(args.dest)  # [FIX #4] valida el disco REAL
    log.info("✅ Dataset limpio y verificado: Train=%d, Val=%d (solape=0, paridad img==lbl, caché purgada).",
             n_train, n_val)
    log.info("   python tools/train_yolo.py --epochs 80 --batch 8 --imgsz 1280")


if __name__ == "__main__":
    main()