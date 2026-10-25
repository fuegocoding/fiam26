"""Partition plan, label maturity, purge and embargo (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 13.2-13.4)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

from pydantic import Field, model_validator

from ..contracts.base import StrictModel, require_aware

DESIGN = "design"
CHECK = "check"
REPLAY = "replay"
PROSPECTIVE = "prospective"

PARTITION_ORDER = (DESIGN, CHECK, REPLAY, PROSPECTIVE)
PROTECTED_PARTITIONS = frozenset({CHECK, REPLAY, PROSPECTIVE})


class Partition(StrictModel):
    name: str = Field(min_length=1)
    start: dt.datetime
    end: dt.datetime
    label: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "Partition":
        require_aware(self.start, "start")
        require_aware(self.end, "end")
        if self.end <= self.start:
            raise ValueError("partition end must lie after start")
        return self

    def contains(self, when: dt.datetime) -> bool:
        when = require_aware(when, "when")
        return self.start <= when <= self.end


class PartitionPlan(StrictModel):
    partitions: list[Partition] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "PartitionPlan":
        names = [p.name for p in self.partitions]
        if len(set(names)) != len(names):
            raise ValueError("partition names must be unique")
        ordered = sorted(self.partitions, key=lambda p: p.start)
        for earlier, later in zip(ordered, ordered[1:]):
            if later.start < earlier.end:
                raise ValueError(
                    f"partitions {earlier.name!r} and {later.name!r} overlap"
                )
        return self

    def partition(self, name: str) -> Partition:
        for part in self.partitions:
            if part.name == name:
                return part
        raise KeyError(name)

    def partition_of(self, when: dt.datetime) -> Optional[str]:
        for part in self.partitions:
            if part.contains(when):
                return part.name
        return None


@dataclass(frozen=True)
class LabelWindow:
    observation_at: dt.datetime
    resolves_at: dt.datetime

    def __post_init__(self) -> None:
        require_aware(self.observation_at, "observation_at")
        require_aware(self.resolves_at, "resolves_at")
        if self.resolves_at < self.observation_at:
            raise ValueError("a label cannot resolve before its observation")


def matured_labels(
    labels: list[LabelWindow], training_cutoff: dt.datetime
) -> list[LabelWindow]:
    cutoff = require_aware(training_cutoff, "training_cutoff")
    return [label for label in labels if label.resolves_at <= cutoff]


def purge_between(
    labels: list[LabelWindow],
    protected_start: dt.datetime,
    protected_end: dt.datetime,
) -> tuple[list[LabelWindow], list[LabelWindow]]:
    protected_start = require_aware(protected_start, "protected_start")
    protected_end = require_aware(protected_end, "protected_end")
    kept: list[LabelWindow] = []
    purged: list[LabelWindow] = []
    for label in labels:
        overlaps = (
            label.observation_at <= protected_end
            and label.resolves_at >= protected_start
        )
        (purged if overlaps else kept).append(label)
    return kept, purged


def embargo_after(
    labels: list[LabelWindow],
    last_training_observation: dt.datetime,
    embargo_end: dt.datetime,
) -> tuple[list[LabelWindow], list[LabelWindow]]:
    last_training_observation = require_aware(
        last_training_observation, "last_training_observation"
    )
    embargo_end = require_aware(embargo_end, "embargo_end")
    if embargo_end < last_training_observation:
        raise ValueError(
            "embargo_end cannot precede the last training observation"
        )
    kept = [
        label
        for label in labels
        if label.observation_at <= last_training_observation
    ]
    embargoed = [
        label
        for label in labels
        if label.observation_at > last_training_observation
    ]
    embargoed = [
        label for label in embargoed if label.observation_at <= embargo_end
    ]
    rest = [label for label in labels if label.observation_at > embargo_end]
    return kept + rest, embargoed


class LateStartPolicy(StrictModel):
    source_id: str = Field(min_length=1)
    first_eligible_at: dt.datetime
    rule: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "LateStartPolicy":
        require_aware(self.first_eligible_at, "first_eligible_at")
        return self

    def eligible(self, when: dt.datetime) -> bool:
        return require_aware(when, "when") >= self.first_eligible_at
