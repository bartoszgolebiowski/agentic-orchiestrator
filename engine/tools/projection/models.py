from __future__ import annotations

from typing import Any, TypeAlias
from collections.abc import Mapping, Sequence

ProjectionSelector: TypeAlias = str | Sequence[str]
ProjectionSpec: TypeAlias = str | Mapping[str, Any]
