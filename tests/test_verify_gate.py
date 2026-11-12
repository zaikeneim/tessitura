"""Regression tests for the complete verification command and its runtime CLI."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from scripts import runtime_health, verify


class VerificationGateTests(unittest.TestCase):
    def test_function_module_failure_fails_the_gate(self):
        attempted = []

        def run(label, args):
            attempted.append(args)
            return label, args != ["-m", "tests.test_merge"], 0.0, "regression failure"

        with patch.object(sys, "argv", ["verify"]), patch.object(verify, "_run", side_effect=run), \
                contextlib.redirect_stdout(io.StringIO()):
            result = verify.main()

        self.assertEqual(result, 1)
        self.assertIn(["-m", "tests.test_merge"], attempted)
        self.assertEqual(len(attempted), len(verify._modules()) + 1)

    def test_full_gate_uses_the_real_runtime_parser_and_a_temporary_snapshot(self):
        baseline = {"versions": {"torch": "test"}, "cuda": False, "cuda_build": None}
        runtime_args = []

        def run(label, args):
            if args[0] != "scripts/runtime_health.py":
                return label, True, 0.0, ""
            runtime_args.append(args)
            with patch.object(sys, "argv", args), contextlib.redirect_stderr(io.StringIO()):
                try:
                    runtime_health.main()
                except SystemExit as error:
                    return label, error.code == 0, 0.0, "runtime CLI rejected arguments"
            return label, True, 0.0, ""

        with patch.object(sys, "argv", ["verify", "--full"]), \
                patch.object(verify, "_run", side_effect=run), \
                patch.object(runtime_health, "add_conda_dll_dirs"), \
                patch.object(runtime_health, "use_bundled_models"), \
                patch.object(runtime_health, "snapshot", return_value=baseline) as snapshot, \
                patch.object(runtime_health, "verify") as health_check, \
                contextlib.redirect_stdout(io.StringIO()):
            result = verify.main()

        self.assertEqual(result, 0)
        snapshot.assert_called_once_with()
        health_check.assert_called_once_with(baseline)
        self.assertEqual([args[1] for args in runtime_args], ["--snapshot", "--verify"])
        self.assertEqual(runtime_args[0][2], runtime_args[1][2])
        self.assertFalse(Path(runtime_args[0][2]).exists(), "temporary baseline must be removed")


if __name__ == "__main__":
    unittest.main()
