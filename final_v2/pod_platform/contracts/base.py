"""Shared envelope for durable records (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 21.1)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from decimal import Decimal
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"
DIGEST_FIELD = "content_hash"


def canonical_digest(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def require_aware(value: dt.datetime, field_name: str) -> dt.datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(dt.timezone.utc)


class DurableModel(BaseModel):
    ID_FIELD: ClassVar[str] = ""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: str = SCHEMA_VERSION
    owner: str = Field(min_length=1)
    created_at: dt.datetime
    parent_id: Optional[str] = None
    content_hash: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {value!r}")
        return value

    @field_validator("created_at")
    @classmethod
    def _check_created_at(cls, value: dt.datetime) -> dt.datetime:
        return require_aware(value, "created_at")

    @model_validator(mode="after")
    def _check_id_present(self) -> "DurableModel":
        if not self.ID_FIELD:
            raise ValueError("DurableModel subclass must declare ID_FIELD")
        if not str(getattr(self, self.ID_FIELD, "") or ""):
            raise ValueError(f"{self.ID_FIELD} must be a non-empty stable id")
        return self

    @property
    def stable_id(self) -> str:
        return str(getattr(self, self.ID_FIELD))

    def digest(self) -> str:
        payload = self.model_dump(mode="json", exclude={DIGEST_FIELD})
        return canonical_digest(payload)

    def sealed(self) -> "DurableModel":
        return self.model_copy(update={DIGEST_FIELD: self.digest()})

    def verify_seal(self) -> bool:
        return self.content_hash is not None and self.content_hash == self.digest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def finite_decimal(value: Decimal, field_name: str) -> Decimal:
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return value
