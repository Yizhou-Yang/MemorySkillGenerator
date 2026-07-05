"""Environment-probe verifier for Grounded Patch Resolution (Method C).

The "verifier" is the environment itself — we run a patch's applicability probe
(a shell command) in the live task container and read the result. This is ground
truth, not a new LLM judge, which is exactly why C can claim a verified signal
without switching the evaluation judge.

The verifier takes an injectable ``exec_fn`` so it is fully testable offline
with a fake shell, and wires to ``docker exec`` (or any executor) in a live run.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Awaitable, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.latest._vgr import Patch, VerificationResult  # noqa: E402

# exec_fn(cmd: str) -> (stdout: str, returncode: int)
ExecFn = Callable[[str], Awaitable[tuple[str, int]]]


class EnvProbeVerifier:
    """Verifies a patch's applicability by executing its probe in the environment.

    Applicability decision:
      - probe empty               -> applies = None  (UNKNOWN; carried, not dropped)
      - expected_signal given     -> applies = (expected_signal in stdout)
      - no expected_signal        -> applies = (returncode == 0)

    Confidence is 1.0 for a clean observation because the environment is ground
    truth; it drops to 0.0 only when the probe itself errors.
    """

    def __init__(self, exec_fn: ExecFn, max_obs: int = 400) -> None:
        self._exec = exec_fn
        self._max_obs = max_obs

    async def verify(self, patch: Patch, context: dict) -> VerificationResult:
        if not patch.probe:
            return VerificationResult(applies=None, confidence=0.0,
                                      observation="", probe="", method="no_probe")
        try:
            stdout, rc = await self._exec(patch.probe)
        except Exception as e:
            return VerificationResult(applies=None, confidence=0.0,
                                      observation=f"exec_error: {str(e)[:160]}",
                                      probe=patch.probe, method="env_probe_error")
        stdout = stdout or ""
        if patch.expected_signal:
            applies = patch.expected_signal in stdout
        else:
            applies = (rc == 0)
        return VerificationResult(applies=applies, confidence=1.0,
                                  observation=stdout[: self._max_obs],
                                  probe=patch.probe, method="env_probe")


# ── Live executors ──────────────────────────────────────────────────────────

def make_docker_exec_fn(container: str, workdir: str = "/app") -> ExecFn:
    """Run probes via ``docker exec`` in the task's container (live runs).

    Probes must be cheap and read-only (stat/grep/test/cat/run-tests). They
    observe the current environment version; they must not mutate it.
    """
    import asyncio

    async def _exec(cmd: str) -> tuple[str, int]:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "-w", workdir, container, "bash", "-lc", cmd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        return out.decode("utf-8", "replace"), proc.returncode

    return _exec


def make_local_exec_fn(workdir: str = ".") -> ExecFn:
    """Run probes in a local shell (for non-Docker terminal tasks / debugging)."""
    import asyncio

    async def _exec(cmd: str) -> tuple[str, int]:
        proc = await asyncio.create_subprocess_shell(
            cmd, cwd=workdir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        return out.decode("utf-8", "replace"), proc.returncode

    return _exec
