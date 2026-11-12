"""Comando unico di verifica: esegue TUTTI i test, comunque siano scritti.

Perche' esiste: `unittest discover` raccoglie solo le classe TestCase, quindi i
moduli scritti a funzioni con runner manuale risultavano "passati" solo perche'
non venivano mai eseguiti. Qui si esegue discover e, in piu', ogni modulo di
test come modulo Python, cosi' nessun file resta fuori dal conteggio.

Uso (dalla root, ambiente attivo):
    python -m scripts.verify           # test
    python -m scripts.verify --full    # test + ambiente + vincoli di versione
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TESTS_DIR = ROOT / "tests"


def _modules() -> list[str]:
    return sorted(
        f"tests.{path.stem}"
        for path in TESTS_DIR.glob("test_*.py")
        if path.is_file()
    )


def _run(label: str, args: list[str]) -> tuple[str, bool, float, str]:
    started = time.time()
    result = subprocess.run(
        [sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    output = (result.stdout or "") + (result.stderr or "")
    return label, result.returncode == 0, time.time() - started, output.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Esegue la verifica completa del progetto")
    parser.add_argument("--full", action="store_true",
                        help="aggiunge diagnostica ambiente e vincoli di versione")
    parser.add_argument("--verbose", action="store_true",
                        help="stampa l'output completo anche dei passi riusciti")
    args = parser.parse_args()

    try:  # accenti corretti in console su Windows
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    steps: list[list[str]] = [["-m", "unittest", "discover", "-s", "tests"]]
    steps += [["-m", module] for module in _modules()]
    failures: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="tessitura-verify-") as folder:
        if args.full:
            baseline = str(Path(folder) / "runtime-baseline.json")
            steps.append(["scripts/check_env.py"])
            # Match runtime_health's CLI and compare the environment before
            # its offline model/decoder/Qt smoke test. Keep this transient
            # snapshot separate from any saved update or accuracy baseline.
            steps.append(["scripts/runtime_health.py", "--snapshot", baseline])
            steps.append(["scripts/runtime_health.py", "--verify", baseline])

        for step in steps:
            label = " ".join(step)
            name, ok, elapsed, output = _run(label, step)
            print(f"[{'OK  ' if ok else 'FAIL'}] {name}  ({elapsed:.1f}s)", flush=True)
            if args.verbose or not ok:
                print(output)
            if not ok:
                failures.append((name, output))

    print()
    if failures:
        print(f"{len(failures)} passi falliti su {len(steps)}:")
        for name, _output in failures:
            print(f"  - {name}")
        return 1
    print(f"Verifica completata: {len(steps)} passi, tutti superati.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
