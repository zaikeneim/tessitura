"""Entry point della GUI Tessitura."""

from __future__ import annotations

import json
import os
import sys

from .bootstrap import (
    add_conda_dll_dirs,
    ensure_std_streams,
    install_system_certs,
    silence_noisy_warnings,
    use_bundled_models,
)
from .version import APP_NAME, APP_PUBLISHER, get_app_version


def _run_smoke_test() -> int:
    """Esegue un controllo diagnostico headless dell'installazione e dei percorsi."""
    from . import paths
    from .migrations import validate_and_migrate_startup
    from .processing.first_run import check_first_run_status

    migration = validate_and_migrate_startup()
    first_run = check_first_run_status()
    payload = {
        "app_name": APP_NAME,
        "version": get_app_version(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "installed_mode": paths.is_installed_mode(),
        "app_root": str(paths.app_root()),
        "config_dir": str(paths.config_dir()),
        "data_dir": str(paths.data_dir()),
        "models_dir": str(paths.models_dir()),
        "cache_dir": str(paths.cache_dir()),
        "logs_dir": str(paths.logs_dir()),
        "migration_status": migration.status,
        "device": first_run.device,
    }
    output = json.dumps(payload, indent=2, ensure_ascii=False)
    out_file = os.environ.get("TESSITURA_SMOKE_OUT")
    if out_file:
        from pathlib import Path

        Path(out_file).write_text(output, encoding="utf-8")
    try:
        sys.stdout.write(output + "\n")
        sys.stdout.flush()
    except OSError:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--version" in args:
        ensure_std_streams()
        out = f"{APP_NAME} {get_app_version()}"
        out_file = os.environ.get("TESSITURA_VERSION_OUT")
        if out_file:
            from pathlib import Path

            Path(out_file).write_text(out, encoding="utf-8")
        try:
            sys.stdout.write(out + "\n")
            sys.stdout.flush()
        except OSError:
            pass
        return 0

    ensure_std_streams()      # con pythonw stdout/stderr sono None: tqdm & co. crashano
    add_conda_dll_dirs()      # DLL di Qt (conda-forge / PyInstaller) trovabili senza attivazione
    use_bundled_models()      # modelli nella cartella modelli configurata
    install_system_certs()    # certificati di sistema (AVG intercetta HTTPS)
    silence_noisy_warnings()  # avvisi innocui di torch/pyannote

    if "--smoke-test" in args:
        return _run_smoke_test()
    if "--timeline-smoke" in args:
        from .timeline_validation import validate_timeline_runtime
        return validate_timeline_runtime(args[args.index("--timeline-smoke") + 1], inference="--inference" in args)

    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication, QMessageBox

    from . import paths
    from .gui.main_window import MainWindow
    from .migrations import validate_and_migrate_startup

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(get_app_version())
    app.setOrganizationName(APP_PUBLISHER)

    from .gui.theme import apply_theme

    apply_theme(app)
    icon_file = paths.app_root() / "assets" / "tessitura.ico"
    if icon_file.is_file():
        app.setWindowIcon(QIcon(str(icon_file)))

    migration_report = validate_and_migrate_startup()
    if migration_report.status == "incompatible_newer_schema":
        QMessageBox.warning(
            None,
            "Configurazione creata da versione più recente",
            migration_report.message,
        )

    window = MainWindow()
    if "--session" in args:
        from pathlib import Path
        from .models.storage import load_session
        session_path = Path(args[args.index("--session") + 1])
        window.session = load_session(session_path)
        window.session_path = session_path
        window._display_transcript()
        window.speakers.set_session(window.session)
        window._load_audio(window.session)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
