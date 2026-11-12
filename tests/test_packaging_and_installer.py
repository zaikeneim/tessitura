"""Tests for Windows packaging specs, Inno Setup script, GitHub Actions workflow, and CLI smoke modes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.app import main as app_main
from src.version import APP_NAME, INNO_APP_ID, get_app_version


class PackagingAndInstallerSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent

    def test_pyinstaller_spec_exists_and_configures_onedir_tessitura(self) -> None:
        spec = self.project_root / "packaging" / "windows" / "tessitura.spec"
        self.assertTrue(spec.is_file())
        content = spec.read_text(encoding="utf-8")
        self.assertIn('name="Tessitura"', content)
        self.assertIn("COLLECT(", content)
        self.assertIn("tessitura.ico", content)
        # Ensure legacy branding is not used
        self.assertNotIn("KrispLike", content)

    def test_inno_setup_script_exists_and_enforces_per_user_and_safe_uninstall(self) -> None:
        iss = self.project_root / "packaging" / "windows" / "tessitura.iss"
        self.assertTrue(iss.is_file())
        content = iss.read_text(encoding="utf-8")
        self.assertIn(INNO_APP_ID, content)
        self.assertIn("PrivilegesRequired=lowest", content)
        self.assertIn("DefaultDirName={localappdata}\\Programs\\{#MyAppName}", content)
        self.assertIn("OutputBaseFilename=Tessitura-Setup-{#MyAppVersion}", content)
        self.assertIn("CloseApplications=yes", content)
        self.assertIn(".installed", content)
        self.assertIn("InitializeSetup", content)
        # Ensure user data and models are never listed in [UninstallDelete]
        self.assertNotIn("{userappdata}\\Tessitura", content)
        self.assertNotIn("{localappdata}\\Tessitura\\models", content)
        self.assertNotIn("{localappdata}\\Tessitura\\data", content)

    def test_github_actions_release_workflow_enforces_least_privilege_and_quality_gate(self) -> None:
        wf = self.project_root / ".github" / "workflows" / "release-windows.yml"
        self.assertTrue(wf.is_file())
        content = wf.read_text(encoding="utf-8")
        self.assertIn("contents: read", content)
        self.assertIn("contents: write", content)
        self.assertIn("needs: verify", content)
        self.assertIn("SHA256SUMS.txt", content)
        self.assertIn("Tessitura-Setup-", content)

    def test_release_jobs_install_gui_and_require_the_complete_gate(self) -> None:
        import yaml

        workflow = self.project_root / ".github" / "workflows" / "release-windows.yml"
        jobs = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]
        for name in ("verify", "build-windows"):
            with self.subTest(job=name):
                commands = "\n".join(step.get("run", "") for step in jobs[name]["steps"])
                self.assertRegex(commands, r"(?m)^\s*(?:python -m )?pip install [^\n]*PySide6")
                self.assertFalse(jobs[name].get("continue-on-error", False))
        checks = jobs["verify"]["steps"]
        gate = next(step for step in checks if "python -m scripts.verify" in step.get("run", ""))
        self.assertIn("python -m scripts.verify", gate["run"].splitlines())
        self.assertNotIn("--full", gate["run"])
        self.assertFalse(gate.get("continue-on-error", False))
        self.assertEqual(jobs["build-windows"]["needs"], "verify")
        self.assertIn("build-windows", jobs["publish-release"]["needs"])

    def test_app_cli_version_and_smoke_test_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ver_out = root / "version.txt"
            smoke_out = root / "smoke.json"
            env = {
                "TESSITURA_VERSION_OUT": str(ver_out),
                "TESSITURA_SMOKE_OUT": str(smoke_out),
                "TESSITURA_CONFIG_DIR": str(root / "config"),
                "TESSITURA_DATA_DIR": str(root / "data"),
                "TESSITURA_MODELS_DIR": str(root / "models"),
                "TESSITURA_CACHE_DIR": str(root / "cache"),
                "TESSITURA_LOGS_DIR": str(root / "logs"),
            }
            with patch.dict(os.environ, env, clear=False):
                rc_ver = app_main(["--version"])
                self.assertEqual(rc_ver, 0)
                self.assertEqual(ver_out.read_text(encoding="utf-8").strip(), f"{APP_NAME} {get_app_version()}")

                rc_smoke = app_main(["--smoke-test"])
                self.assertEqual(rc_smoke, 0)
                payload = json.loads(smoke_out.read_text(encoding="utf-8"))
                self.assertEqual(payload["app_name"], APP_NAME)
                self.assertEqual(payload["migration_status"], "ok")


if __name__ == "__main__":
    unittest.main()

