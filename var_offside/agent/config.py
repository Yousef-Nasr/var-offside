"""Auto-review settings, read from the environment or ``var_offside/.env``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from dotenv import dotenv_values

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_COST_USD = 0.50
DEFAULT_MAX_CALLS = 20
DEFAULT_STAGE = 1


@dataclass(frozen=True)
class AgentConfig:
    """Connection settings and per-run limits. The key is left out of ``repr``."""

    api_key: Optional[str] = field(default=None, repr=False)
    model: Optional[str] = None
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_cost_usd: float = DEFAULT_MAX_COST_USD
    max_calls: int = DEFAULT_MAX_CALLS
    auto_review_stage: int = DEFAULT_STAGE  # Rollout stage, 1 to 4. At 1, every result needs checking.

    @property
    def vision_available(self) -> bool:
        """Without a key and model, only the rule checks can run."""
        return bool(self.api_key and self.model)


def load_config(
    env_path: Path = ENV_PATH, environ: Optional[Mapping[str, str]] = None
) -> AgentConfig:
    """Build the config. Process environment values win over the ``.env`` file."""
    values = dict(dotenv_values(env_path)) if Path(env_path).is_file() else {}
    values.update(os.environ if environ is None else environ)

    def text(name: str) -> Optional[str]:
        value = (values.get(name) or "").strip()
        return value or None

    return AgentConfig(
        api_key=text("OPENROUTER_API_KEY"),
        model=text("OPENROUTER_MODEL_ID"),
        max_cost_usd=float(text("OPENROUTER_MAX_COST_USD") or DEFAULT_MAX_COST_USD),
        max_calls=int(text("OPENROUTER_MAX_CALLS") or DEFAULT_MAX_CALLS),
        auto_review_stage=int(text("AUTO_REVIEW_STAGE") or DEFAULT_STAGE),
    )
