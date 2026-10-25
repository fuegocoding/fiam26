"""Run manifest contract (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 22.4)."""

from __future__ import annotations

import re

from pydantic import Field, model_validator

from ..contracts.base import StrictModel

_FORBIDDEN_VERSION_TOKENS = re.compile(
    r"(^|[/\\])latest($|[/\\.])", re.IGNORECASE
)


class RunManifest(StrictModel):
    run_id: str = Field(min_length=1)
    code_commit_or_tree_hash: str = Field(min_length=1)
    environment_lock_hash: str = Field(min_length=1)
    config_hashes: dict[str, str] = Field(min_length=1)
    source_snapshot_ids: dict[str, str] = Field(min_length=1)
    security_master_version: str = Field(min_length=1)
    economic_graph_version: str = Field(min_length=1)
    model_versions: dict[str, str] = Field(default_factory=dict)
    bridge_versions: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    cache_versions: dict[str, str] = Field(default_factory=dict)
    calendar_policy: str = Field(min_length=1)
    timezone_policy: str = Field(min_length=1)
    seeds: dict[str, int] = Field(default_factory=dict)
    solver_settings: dict[str, str] = Field(default_factory=dict)
    data_access_policy_version: str = Field(min_length=1)
    evaluation_policy_version: str = Field(min_length=1)
    initial_accounting_state: dict[str, str] = Field(default_factory=dict)
    expected_outputs: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_no_floating_versions(self) -> "RunManifest":
        versioned = {
            "code_commit_or_tree_hash": self.code_commit_or_tree_hash,
            "environment_lock_hash": self.environment_lock_hash,
            "security_master_version": self.security_master_version,
            "economic_graph_version": self.economic_graph_version,
        }
        versioned.update(self.config_hashes)
        versioned.update(self.source_snapshot_ids)
        versioned.update(self.model_versions)
        versioned.update(self.bridge_versions)
        for name, value in versioned.items():
            if _FORBIDDEN_VERSION_TOKENS.search(value):
                raise ValueError(
                    f"{name} must not rely on a floating reference such as 'latest'"
                )
        return self

    @model_validator(mode="after")
    def _check_hash_shapes(self) -> "RunManifest":
        for name in ("code_commit_or_tree_hash", "environment_lock_hash"):
            value = getattr(self, name)
            if len(value) < 7:
                raise ValueError(f"{name} is too short to identify an artifact")
        for map_name in (
            "config_hashes",
            "source_snapshot_ids",
            "expected_outputs",
        ):
            for key, value in getattr(self, map_name).items():
                if not value:
                    raise ValueError(f"{map_name}[{key!r}] must not be empty")
        return self
