"""Pluggable, license-respecting forward curve sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from bess_optimization.data.schemas import validate_forward_quotes


class ForwardCurveProvider(ABC):
    """Interface for authorized forward data providers."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load one or several successive forward snapshots."""


class CSVForwardCurveProvider(ForwardCurveProvider):
    """Load EEX (or other authorized) quotes supplied by the user as CSV."""

    def __init__(self, paths: str | Path | list[str | Path], timezone: str = "Europe/Paris"):
        self.paths = [paths] if isinstance(paths, (str, Path)) else paths
        self.timezone = timezone

    def load(self) -> pd.DataFrame:
        frames = [pd.read_csv(path) for path in self.paths]
        if not frames:
            raise ValueError("At least one CSV path is required")
        return validate_forward_quotes(pd.concat(frames, ignore_index=True), self.timezone)


class APIForwardCurveProvider(ForwardCurveProvider):
    """Extension point for a future licensed API implementation."""

    def load(self) -> pd.DataFrame:
        raise NotImplementedError("Configure an authorized market-data API implementation")
