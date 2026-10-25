"""Restartable stage runner with a source-hash log (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 22.4-22.5, 25).

Each stage is bound to a hash of its declared inputs and source hashes. A stage
with a matching, intact checkpoint is not re-run (idempotent restart); a
checkpoint whose inputs changed fails closed instead of being silently reused or
overwritten (a correction is a new run id, spec 25.2). Every attempt, resume and
failure is appended to ``stages.jsonl`` - the log is never rewritten.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil

from .common import tree_sha256

try:
    import fcntl
except ImportError:
    fcntl = None
    import msvcrt
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..contracts.errors import ErrorCode, PodPlatformError
from .common import (
    atomic_write_json,
    canonical_json,
    digest,
    fail,
    file_sha256,
    now_utc,
    read_json,
    safe_id,
)

STAGE_STATUSES = ("started", "completed", "resumed", "failed")


class StageRunner:
    def __init__(
        self,
        root: "str | Path",
        run_id: str,
        *,
        clock: Callable[[], dt.datetime] = now_utc,
    ):
        self.run_id = safe_id(run_id, "run id")
        self.root = Path(root)
        self.dir = self.root / "application" / "runs" / self.run_id
        self.checkpoint_dir = self.dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "stages.jsonl"
        self._clock = clock

    # ------------------------------------------------------------------ log
    def _log(self, entry: dict[str, Any]) -> None:
        stamp = self._clock()
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("stage clock must be timezone-aware")
        line = {
            "at": stamp.astimezone(dt.timezone.utc).isoformat(),
            "run_id": self.run_id,
            **entry,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(line) + "\n")

    def records(self) -> list[dict[str, Any]]:
        if not self.log_path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def completed(self) -> list[str]:
        """Stage names with a checkpoint on disk, in log order."""
        names: list[str] = []
        for record in self.records():
            if (
                record.get("status") in ("completed", "resumed")
                and record["stage"] not in names
            ):
                names.append(record["stage"])
        return names

    # ------------------------------------------------------------------ run
    def run(
        self,
        name: str,
        fn: Callable[[], Mapping[str, Any]],
        *,
        inputs: Mapping[str, Any],
        source_hashes: Optional[Mapping[str, str]] = None,
        artifacts_root: Optional["str | Path"] = None,
        owned_outputs: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Run (or resume) one stage. ``fn`` must return a JSON-serialisable mapping.

        A stage output may contain ``"artifacts": {relative_path: sha256}``; with
        ``artifacts_root`` those files are re-verified on resume, and a missing or
        altered file invalidates the checkpoint (fail closed).
        """
        safe_id(name, "stage name")
        sources = dict(sorted((source_hashes or {}).items()))
        input_hash = digest(
            {"stage": name, "inputs": dict(inputs), "sources": sources}
        )
        # Bind even an interrupted first attempt: changed inputs cannot reuse a run.
        binding_dir = self.dir / "bindings"
        binding_dir.mkdir(exist_ok=True)
        binding = binding_dir / f"{name}.json"
        if binding.exists() and read_json(binding)["input_hash"] != input_hash:
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"stage {name!r}: inputs/source hashes changed after an attempted run",
                "Use a new run id for changed inputs, including after interruption.",
            )
        if not binding.exists():
            atomic_write_json(
                binding,
                {
                    "input_hash": input_hash,
                    "owned_outputs": list(owned_outputs),
                },
            )
        elif read_json(binding).get("owned_outputs", []) != list(owned_outputs):
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                "stage output ownership changed",
                "Use a new run id.",
            )
        checkpoint = self.checkpoint_dir / f"{name}.json"
        if checkpoint.is_file():
            stored = read_json(checkpoint)
            if stored.get("input_hash") != input_hash:
                self._log(
                    {
                        "stage": name,
                        "status": "failed",
                        "input_hash": input_hash,
                        "error_code": ErrorCode.SCHEMA_MISMATCH.value,
                        "error": "stage inputs or source hashes changed since the checkpoint",
                    }
                )
                fail(
                    ErrorCode.SCHEMA_MISMATCH,
                    f"stage {name!r}: inputs/source hashes differ from the stored checkpoint of run {self.run_id}",
                    "Use a new run id for changed inputs; checkpoints are never overwritten.",
                )
            output = stored.get("output")
            if digest(output) != stored.get("output_hash"):
                self._log(
                    {
                        "stage": name,
                        "status": "failed",
                        "input_hash": input_hash,
                        "error_code": ErrorCode.ACCOUNTING_RECONCILIATION_FAILED.value,
                        "error": "checkpoint output failed its hash",
                    }
                )
                fail(
                    ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                    f"stage {name!r}: stored checkpoint output failed its content hash",
                    "A damaged checkpoint is not trusted; start a new run id.",
                )
            self._verify_artifacts(name, output, artifacts_root)
            self._log(
                {
                    "stage": name,
                    "status": "resumed",
                    "input_hash": input_hash,
                    "output_hash": stored["output_hash"],
                    "source_hashes": sources,
                }
            )
            return output
        attempt_dir = self.dir / "attempts" / name
        attempt_dir.mkdir(parents=True, exist_ok=True)
        lock = (attempt_dir / ".lock").open("a")
        try:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                lock.seek(0)
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as _exc:
                    raise BlockingIOError(str(_exc)) from _exc
        except BlockingIOError:
            lock.close()
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                "stage already has an active attempt",
                "Concurrent retries are denied.",
            )
        if checkpoint.exists():
            lock.close()
            return self.run(
                name,
                fn,
                inputs=inputs,
                source_hashes=source_hashes,
                artifacts_root=artifacts_root,
                owned_outputs=owned_outputs,
            )
        try:
            attempts = sorted(
                p for p in attempt_dir.glob("attempt_*") if p.is_dir()
            )
            attempt_id = f"attempt_{len(attempts) + 1:06d}"
            current = attempt_dir / attempt_id
            current.mkdir()
            atomic_write_json(
                current / "started.json",
                {
                    "stage": name,
                    "attempt_id": attempt_id,
                    "input_hash": input_hash,
                    "source_hashes": sources,
                    "owned_outputs": list(owned_outputs),
                },
            )
            if attempts:
                for previous in attempts:
                    if not (previous / "terminal.json").exists():
                        atomic_write_json(
                            previous / "terminal.json",
                            {
                                "status": "interrupted",
                                "recovered_by": attempt_id,
                            },
                        )
                        self._log(
                            {
                                "stage": name,
                                "status": "interrupted",
                                "attempt_id": previous.name,
                                "recovered_by": attempt_id,
                            }
                        )
                self._quarantine(name, current, artifacts_root, owned_outputs)
            self._log(
                {
                    "stage": name,
                    "status": "started",
                    "attempt_id": attempt_id,
                    "input_hash": input_hash,
                    "source_hashes": sources,
                }
            )
            try:
                raw = fn()
                output = json.loads(canonical_json(dict(raw)))
            except PodPlatformError as exc:
                atomic_write_json(
                    current / "terminal.json",
                    {"status": "failed", "error": exc.response.summary[:500]},
                )
                lock.close()
                self._log(
                    {
                        "stage": name,
                        "status": "failed",
                        "input_hash": input_hash,
                        "error_code": exc.response.error.code.value,
                        "error": exc.response.summary[:500],
                    }
                )
                raise
            except BaseException as exc:
                atomic_write_json(
                    current / "terminal.json",
                    {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                    },
                )
                lock.close()
                self._log(
                    {
                        "stage": name,
                        "status": "failed",
                        "input_hash": input_hash,
                        "error_code": "UNTYPED_EXCEPTION",
                        "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                    }
                )
                raise
            output_hash = digest(output)
            atomic_write_json(
                checkpoint,
                {
                    "stage": name,
                    "run_id": self.run_id,
                    "input_hash": input_hash,
                    "source_hashes": sources,
                    "output": output,
                    "output_hash": output_hash,
                    "completed_at": self._clock()
                    .astimezone(dt.timezone.utc)
                    .isoformat(),
                },
            )
            atomic_write_json(
                current / "terminal.json",
                {"status": "completed", "output_hash": output_hash},
            )
            lock.close()
            self._log(
                {
                    "stage": name,
                    "status": "completed",
                    "attempt_id": attempt_id,
                    "input_hash": input_hash,
                    "output_hash": output_hash,
                    "source_hashes": sources,
                }
            )
            return output
        finally:
            lock.close()

    def _quarantine(self, name, attempt, artifacts_root, owned_outputs):
        if not owned_outputs:
            return
        if artifacts_root is None:
            raise ValueError("owned stage outputs need an artifacts root")
        root = Path(artifacts_root).resolve()
        protected = set()
        for checkpoint in self.checkpoint_dir.glob("*.json"):
            record = read_json(checkpoint)
            if record["stage"] != name:
                protected.update(record.get("output", {}).get("artifacts", {}))
        plan = []
        for rel in owned_outputs:
            target = root / rel
            if (
                not target.resolve().is_relative_to(root)
                or Path(rel).is_absolute()
                or ".." in Path(rel).parts
            ):
                raise ValueError("stage output escapes declared root")
            if any(
                p == rel or p.startswith(rel.rstrip("/") + "/")
                for p in protected
            ):
                raise ValueError(
                    "cannot quarantine evidence committed by a completed stage"
                )
            if target.exists():
                plan.append(
                    {"relative_path": rel, "sha256": tree_sha256(target)}
                )
        atomic_write_json(
            attempt / "quarantine_manifest.json",
            {"stage": name, "outputs": plan},
        )
        for item in plan:
            source = root / item["relative_path"]
            destination = attempt / "quarantine" / item["relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        self._log(
            {
                "stage": name,
                "status": "quarantined",
                "attempt_id": attempt.name,
                "outputs": plan,
            }
        )

    def _verify_artifacts(
        self, name: str, output: Any, artifacts_root: Optional["str | Path"]
    ) -> None:
        if artifacts_root is None or not isinstance(output, dict):
            return
        artifacts = output.get("artifacts")
        if not isinstance(artifacts, dict):
            return
        root = Path(artifacts_root)
        for rel, expected in sorted(artifacts.items()):
            target = root / rel
            if not target.is_file() or file_sha256(target) != expected:
                self._log(
                    {
                        "stage": name,
                        "status": "failed",
                        "error_code": ErrorCode.ACCOUNTING_RECONCILIATION_FAILED.value,
                        "error": f"artifact {rel} missing or altered since the checkpoint",
                    }
                )
                fail(
                    ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                    f"stage {name!r}: artifact {rel} is missing or no longer matches its checkpoint hash",
                    "Historical artifacts are never silently regenerated; start a new run id.",
                )
