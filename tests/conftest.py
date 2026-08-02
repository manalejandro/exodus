"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path

import loguru
import pytest

from exodus.config import ExodusConfig


@pytest.fixture(autouse=True)
def _quiet_loguru():
    loguru.logger.remove()
    yield
    loguru.logger.remove()


@pytest.fixture
def config() -> ExodusConfig:
    return ExodusConfig(
        data_dir=Path("/tmp/exodus-tests"),
        epoch_seconds=1.0,
        election_timeout_seconds=10.0,
        heartbeat_seconds=0.5,
    )
