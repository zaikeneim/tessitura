"""The documented CLI (`python -m scripts.run_pipeline`) must at least parse.

Regression: `--speakers` and `--no-denoise` were declared twice, so argparse
raised `ArgumentError: conflicting option string` before any work started and
the README command could not run at all (since commit 23bc2b4, 2026-09-21).

Esecuzione: python -m tests.test_run_pipeline_cli
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class RunPipelineCliTests(unittest.TestCase):
    def test_help_parses_and_lists_documented_options(self):
        result = subprocess.run([sys.executable, "-m", "scripts.run_pipeline", "--help"], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for option in ("--profile", "--speakers", "--no-denoise", "--review", "--quality-retries", "--out"):
            self.assertIn(option, result.stdout)
        self.assertNotIn("conflicting option string", result.stdout + result.stderr)

    def test_profile_is_mandatory(self):
        result = subprocess.run([sys.executable, "-m", "scripts.run_pipeline", "audio.mp3"], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--profile", result.stderr)


if __name__ == "__main__":
    unittest.main()
