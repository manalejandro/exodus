"""Runtime configuration for an exodus node.

Parameters are chosen so that a small, non-profit network converges quickly and
is robust to casual misbehaviour without any monetary stake.  Every parameter
can be overridden through ``EXODUS_*`` environment variables, which makes the
whole protocol tunable in production without code changes.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _env(name: str, default: str) -> str:
    return os.environ.get(f"EXODUS_{name}", default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name, "true" if default else "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _default_data_dir() -> Path:
    """XDG-style per-user data directory for the exodus node."""

    base = os.environ.get("XDG_DATA_HOME", "~/.local/share")
    return Path(base).expanduser() / "exodus"


@dataclass(frozen=True)
class ExodusConfig:
    """Immutable tunables for one exodus node.

    Attributes
    ----------
    data_dir:
        Directory holding the identity and the SQLite ledger.
    node_name:
        Optional human-readable label for this node.
    epoch_seconds:
        How often the sealer proposes a checkpoint.
    election_timeout_seconds:
        How long a node waits for a checkpoint from the sealer before treating
        the sealer as faulty and rotating leadership.
    byzantine:
        Use a BFT quorum (``2f+1`` out of ``N``) instead of a simple majority.
        Safe default for a permissionless network.
    max_faulty:
        Number of faulty/Byzantine nodes to tolerate (``f``).  If ``None`` it
        is derived from the observed peer count at each proposal.
    claim_dedup_window:
        Keep claim ids for deduplication this many checkpoints back.
    flops_tolerance:
        Allowed relative deviation between the claimed token/memory figures and
        the FLOPS sanity estimate.
    credits_per_cu:
        Compute credits awarded per verified Compute Unit.
    reward_diminishing:
        Exponent < 1 flattens the credit curve so huge contributions saturate.
    credit_halflife_seconds:
        Halving period for credit decay (anti-hoarding).
    free_quota_seconds:
        Free inference budget every node starts each day with (base AI time).
    seconds_per_credit:
        How many "extra AI time" seconds a credit is worth.
    max_priority_levels:
        Number of scheduling priority tiers derived from credits.
    """

    data_dir: Path = field(default_factory=_default_data_dir)
    node_name: str = "exodus-node"

    epoch_seconds: float = _env_float("EPOCH_SECONDS", 30.0)
    election_timeout_seconds: float = _env_float(
        "ELECTION_TIMEOUT_SECONDS", 90.0
    )
    byzantine: bool = _env_bool("BYZANTINE", True)
    max_faulty: int | None = None
    claim_dedup_window: int = _env_int("CLAIM_DEDUP_WINDOW", 256)
    active_peer_window: int = _env_int("ACTIVE_PEER_WINDOW", 5)
    heartbeat_seconds: float = _env_float("HEARTBEAT_SECONDS", 10.0)

    flops_tolerance: float = _env_float("FLOPS_TOLERANCE", 0.5)
    credits_per_cu: float = _env_float("CREDITS_PER_CU", 0.01)
    reward_diminishing: float = _env_float("REWARD_DIMINISHING", 0.85)
    credit_halflife_seconds: float = _env_float(
        "CREDIT_HALFLIFE_SECONDS", 30 * 24 * 3600
    )
    free_quota_seconds: float = _env_float("FREE_QUOTA_SECONDS", 300.0)
    seconds_per_credit: float = _env_float("SECONDS_PER_CREDIT", 60.0)
    max_priority_levels: int = _env_int("MAX_PRIORITY_LEVELS", 5)

    @classmethod
    def from_env(cls) -> ExodusConfig:
        return cls()

    @property
    def identity_path(self) -> Path:
        return self.data_dir / "identity.key"

    @property
    def ledger_path(self) -> Path:
        return self.data_dir / "ledger.sqlite3"


def config_help() -> str:
    """Human-readable summary of the effective configuration (for the CLI)."""

    config = ExodusConfig.from_env()
    lines = [
        "Exodus runtime configuration",
        "===========================",
        f"  data dir                    : {config.data_dir}",
        f"  epoch (checkpoint period)   : {config.epoch_seconds}s",
        f"  election timeout            : {config.election_timeout_seconds}s",
        "  quorum model                : "
        + ("byzantine (2f+1)" if config.byzantine else "majority"),        f"  credits per compute unit    : {config.credits_per_cu}",
        f"  reward curve exponent       : {config.reward_diminishing}",
        f"  credit half-life            : {config.credit_halflife_seconds}s",
        f"  free AI-time quota          : {config.free_quota_seconds}s/day",
        f"  seconds of AI time / credit : {config.seconds_per_credit}",
    ]
    return "\n".join(lines)


def tunable_parameters() -> dict[str, Any]:
    """Flatten every tunable into a plain mapping (used by the API)."""

    config = ExodusConfig.from_env()
    return {
        key: value for key, value in vars(config).items() if not key.endswith("_path")
    }


def _getenv(name: str) -> str | None:
    return os.environ.get(name)


# Kept around so tooling can discover which environment variables matter.
ENV_VARIABLES: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("EXODUS_EPOCH_SECONDS", lambda: _env_float("EPOCH_SECONDS", 30.0)),
    (
        "EXODUS_ELECTION_TIMEOUT_SECONDS",
        lambda: _env_float("ELECTION_TIMEOUT_SECONDS", 90.0),
    ),
    ("EXODUS_BYZANTINE", lambda: _env_bool("BYZANTINE", True)),
    ("EXODUS_MAX_FAULTY", lambda: _env_int("MAX_FAULTY", -1)),
    ("EXODUS_CREDITS_PER_CU", lambda: _env_float("CREDITS_PER_CU", 0.01)),
    (
        "EXODUS_REWARD_DIMINISHING",
        lambda: _env_float("REWARD_DIMINISHING", 0.85),
    ),
    (
        "EXODUS_CREDIT_HALFLIFE_SECONDS",
        lambda: _env_float("CREDIT_HALFLIFE_SECONDS", 30 * 24 * 3600),
    ),
    (
        "EXODUS_FREE_QUOTA_SECONDS",
        lambda: _env_float("FREE_QUOTA_SECONDS", 300.0),
    ),
    (
        "EXODUS_SECONDS_PER_CREDIT",
        lambda: _env_float("SECONDS_PER_CREDIT", 60.0),
    ),
)


def env_variables() -> list[str]:
    return [name for name, _ in ENV_VARIABLES]
