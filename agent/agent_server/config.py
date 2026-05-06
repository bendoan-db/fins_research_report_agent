"""Single point of access for agent_config.yaml. Cached so each process reads
the file once."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "agent_config.yaml"


@lru_cache(maxsize=1)
def load_agent_config() -> dict[str, Any]:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)
