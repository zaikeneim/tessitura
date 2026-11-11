"""Fault injection against the real updater, using a tiny fake environment."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("powershell"), "Windows PowerShell required")
class UpdateSafetyTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="tessitura-update-test-")
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / "scripts").mkdir()
        (self.root / "env").mkdir()
        (self.root / "env/python.exe").write_bytes(b"known-good")
        for name in ("runtime.ps1", "update.ps1"):
            shutil.copy2(ROOT / "scripts" / name, self.root / "scripts" / name)
        # Override external commands only; exercise the real transaction and
        # pointer management. No package manager is run by these tests.
        with (self.root / "scripts/runtime.ps1").open("a", encoding="utf-8") as file:
            file.write('''
function Find-TessituraConda { return 'fake-conda' }
$script:calls = 0
function Invoke-Checked([string] $Exe, [string[]] $Arguments) {
    $script:calls++
    if ($script:calls -eq 1) {
        $index = [Array]::IndexOf($Arguments, '--prefix')
        $target = $Arguments[$index + 1]
        $null = New-Item -ItemType Directory -Path $target -Force
        [IO.File]::WriteAllText((Join-Path $target 'python.exe'), 'candidate')
    }
    if ($script:calls -eq [int]$env:TESSITURA_TEST_FAIL_STEP) { throw 'Injected failure' }
    if ($env:TESSITURA_TEST_PAUSE -eq '1' -and $script:calls -eq 1) {
        [IO.File]::WriteAllText((Join-Path $root 'paused'), 'ready')
        Start-Sleep -Seconds 30
    }
}
''')

    def run_update(self, fail=0, *args):
        import os
        env = dict(os.environ, TESSITURA_TEST_FAIL_STEP=str(fail))
        return subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                               "-File", str(self.root / "scripts/update.ps1"), *args],
                              env=env, capture_output=True, timeout=30)

    def test_every_failed_stage_leaves_active_environment_untouched(self):
        for step in range(1, 11):
            with self.subTest(step=step):
                result = self.run_update(step)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(b"Injected failure", b" ".join(result.stderr.split()))
                self.assertFalse((self.root / ".runtime.json").exists())
                self.assertEqual((self.root / "env/python.exe").read_bytes(), b"known-good")

    def test_activation_and_rollback(self):
        result = self.run_update()
        self.assertEqual(result.returncode, 0, result.stderr)
        state = json.loads((self.root / ".runtime.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(state["previous"], "env")
        self.assertTrue(state["active"].startswith(".runtimes/"))
        failed = self.run_update(5)
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(json.loads((self.root / ".runtime.json").read_text()), state)
        result = self.run_update(0, "-Rollback")
        self.assertEqual(result.returncode, 0, result.stderr)
        after = json.loads((self.root / ".runtime.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(after["active"], "env")
        self.assertEqual(after["previous"], state["active"])
        self.assertEqual((self.root / "env/python.exe").read_bytes(), b"known-good")

    def test_pointer_rejects_directory_escape(self):
        (self.root / ".runtime.json").write_text(json.dumps({"active": "../outside", "previous": None}))
        result = self.run_update()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / ".runtimes").exists())

    def test_corrupted_pointer_can_be_rolled_back(self):
        (self.root / ".runtime.json").write_text("incomplete JSON")
        result = self.run_update(0, "-Rollback")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((self.root / ".runtime.json").read_text())["active"], "env")

    def test_interruption_and_concurrent_update_keep_original(self):
        import os
        import time
        env = dict(os.environ, TESSITURA_TEST_FAIL_STEP="0", TESSITURA_TEST_PAUSE="1")
        process = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                                    "-File", str(self.root / "scripts/update.ps1")],
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic()+15
            while not (self.root/"paused").exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    self.fail(f"Updater exited before pause: {process.communicate()}")
                time.sleep(.05)
            self.assertTrue((self.root/"paused").exists())
            competing = self.run_update()
            self.assertNotEqual(competing.returncode, 0)
        finally:
            process.terminate()
            process.communicate(timeout=10)
        self.assertFalse((self.root/".runtime.json").exists())
        self.assertEqual((self.root/"env/python.exe").read_bytes(), b"known-good")
        # The OS released the lock, so a new update can succeed after the crash.
        result = self.run_update()
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
