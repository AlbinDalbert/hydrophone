from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cli_wrapper_can_run_status():
    result = subprocess.run([str(ROOT / "bin" / "gemma-flow"), "status"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode in {0, 1}
    assert result.stderr == ""
    assert result.stdout.strip() in {"idle", "stale"} or result.stdout.lstrip().startswith("{")
