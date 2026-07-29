"""Typed configuration models and YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BESSConfig(BaseModel):
    """Physical and economic battery parameters.

    Power is in MW, energy in MWh, costs in EUR/MWh and time in hours.
    Efficiencies apply respectively when energy enters and leaves the stored-energy
    state. Round-trip efficiency is therefore ``charge_efficiency * discharge_efficiency``.
    """

    model_config = ConfigDict(extra="forbid")

    max_charge_mw: float = Field(50.0, gt=0)
    max_discharge_mw: float = Field(50.0, gt=0)
    energy_capacity_mwh: float = Field(100.0, gt=0)
    charge_efficiency: float = Field(0.9434, gt=0, le=1)
    discharge_efficiency: float = Field(0.9434, gt=0, le=1)
    min_soc_mwh: float = Field(0.0, ge=0)
    max_soc_mwh: float = Field(100.0, gt=0)
    initial_soc_mwh: float = Field(50.0, ge=0)
    final_soc_mwh: float | None = Field(50.0, ge=0)
    min_mode_duration_hours: int = Field(0, ge=0)
    variable_charge_cost_eur_mwh: float = Field(0.0, ge=0)
    variable_discharge_cost_eur_mwh: float = Field(0.5, ge=0)
    degradation_cost_eur_mwh: float = Field(3.0, ge=0)
    max_equivalent_cycles_per_day: float | None = Field(1.5, gt=0)
    unavailable_timestamps: list[str] = Field(default_factory=list)
    availability_factor: float = Field(1.0, ge=0, le=1)
    timestep_hours: float = Field(1.0, gt=0)

    @model_validator(mode="after")
    def validate_soc(self) -> BESSConfig:
        if self.max_soc_mwh > self.energy_capacity_mwh:
            raise ValueError("max_soc_mwh cannot exceed energy_capacity_mwh")
        if self.min_soc_mwh >= self.max_soc_mwh:
            raise ValueError("min_soc_mwh must be below max_soc_mwh")
        for name, value in (
            ("initial_soc_mwh", self.initial_soc_mwh),
            ("final_soc_mwh", self.final_soc_mwh),
        ):
            if value is not None and not self.min_soc_mwh <= value <= self.max_soc_mwh:
                raise ValueError(f"{name} must lie within SOC bounds")
        return self

    @property
    def round_trip_efficiency(self) -> float:
        return self.charge_efficiency * self.discharge_efficiency


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["bootstrap", "factor"] = "bootstrap"
    n_scenarios: int = Field(20, ge=1)
    seed: int = 42
    block_days: int = Field(7, ge=1)


class HedgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    products: Literal["baseload", "peakload", "both"] = "both"
    objective: Literal["min_variance", "mean_variance"] = "min_variance"
    risk_aversion_lambda: float = Field(0.0, ge=0)
    min_volume_mw: float = -50.0
    max_volume_mw: float = 50.0
    volume_step_mw: float | None = Field(None, gt=0)
    regularization: float = Field(1e-6, ge=0)


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: str = "FR"
    timezone: str = "Europe/Paris"
    start: str = "2027-01-01"
    end: str = "2027-01-15"
    historical_start: str = "2022-01-01"
    historical_end: str = "2025-12-31 23:00"
    dispatch_mode: Literal["perfect_foresight", "expected"] = "expected"
    dispatch_quantile: float | None = Field(None, ge=0, le=1)
    minimum_revenue_target_eur: float = 0.0
    output_dir: str = "data/processed/demo"
    battery: BESSConfig = Field(default_factory=BESSConfig)
    scenarios: ScenarioConfig = Field(default_factory=ScenarioConfig)
    hedge: HedgeConfig = Field(default_factory=HedgeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AnalysisConfig:
        with Path(path).open(encoding="utf-8") as stream:
            return cls.model_validate(yaml.safe_load(stream))
