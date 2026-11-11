"""Copia la cache dei modelli HuggingFace senza duplicarne i file.

La cache tiene i pesi una volta sola in `blobs/` e li espone in `snapshots/` con
dei collegamenti. Una copia normale li risolve, e 3 GB diventano 6. Qui i
collegamenti vengono ricreati come hardlink: nessun privilegio speciale
richiesto, e i pesi restano scritti una volta sola.

Uso:  python scripts/copy_model_cache.py <sorgente> <destinazione>
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def copy_cache(src: Path, dst: Path) -> tuple[int, int, int]:
    copied = linked = skipped = 0
    for root, dirs, files in os.walk(src):
        dirs.sort()  # "blobs" prima di "snapshots": i link trovano il bersaglio
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            s = Path(root) / name
            d = dst / rel / name
            if d.exists():
                skipped += 1
                continue
            if s.is_symlink():
                # il bersaglio e' gia' stato copiato in blobs/: ci si aggancia
                target = (s.parent / os.readlink(s)).resolve()
                try:
                    rel_target = target.relative_to(src)
                except ValueError:
                    shutil.copy2(s, d)
                    copied += 1
                    continue
                dest_target = dst / rel_target
                if dest_target.exists():
                    os.link(dest_target, d)
                    linked += 1
                else:
                    shutil.copy2(s, d)
                    copied += 1
            else:
                shutil.copy2(s, d)
                copied += 1
    return copied, linked, skipped


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if not src.is_dir():
        print(f"sorgente inesistente: {src}")
        return 1
    copied, linked, skipped = copy_cache(src, dst)
    print(f"copiati {copied} file, {linked} agganciati agli originali, {skipped} gia' presenti")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
