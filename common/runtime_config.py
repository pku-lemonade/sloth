import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator

TRACE_END_CYCLE_DEFAULT = int((1 << 31) - 1)


class SimulatorInputConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    workload: Path = Path("data/workload_example.json")
    arch: Path = Path("data/arch_example.json")
    failslow: Path = Path("data/fail_example.json")

    @field_validator("workload", "arch", "failslow", mode="before")
    @classmethod
    def normalize_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser()

    @field_validator("workload", "arch", "failslow")
    @classmethod
    def validate_file_exists(cls, value: Path, info: ValidationInfo) -> Path:
        if not value.is_file():
            raise ValueError(f"{info.field_name} file does not exist: {value}")
        return value


class ProbeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    fragment: Literal["Exec", "Route", "Mem"]
    kind: Literal["Comm", "Comp", "IO"]
    location: Literal["Post", "Pre", "Surround"] = "Surround"
    level: Literal["Inst", "Stage"] = "Stage"
    structure: Literal["List", "Sketch"] = "Sketch"

    def as_runtime_list(self) -> list[str]:
        return [self.fragment, self.kind, self.location, self.level, self.structure]


class SimulatorRuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    noc_model: Literal["basic", "packet"] = "basic"
    inference_count: int = 1

    @field_validator("inference_count")
    @classmethod
    def validate_inference_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("inference_count must be at least 1")
        return value


class RecorderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    hash: int = 5
    bucket: int = 1024
    size: int = 8192
    threshold: int = 10

    @field_validator("hash", "bucket", "size", "threshold")
    @classmethod
    def validate_positive(cls, value: int, info: ValidationInfo) -> int:
        if value < 1:
            raise ValueError(f"{info.field_name} must be positive")
        return value


class MonitoringConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    trace_start_cycle: int = 0
    trace_end_cycle: int = TRACE_END_CYCLE_DEFAULT
    enable_flow_trace: bool = False

    @model_validator(mode="after")
    def validate_cycle_range(self) -> "MonitoringConfig":
        if self.trace_start_cycle > self.trace_end_cycle:
            raise ValueError("trace_start_cycle must be less than or equal to trace_end_cycle")
        return self

    def should_record(self, cycle: int | float) -> bool:
        return self.trace_start_cycle <= cycle <= self.trace_end_cycle


class LoggingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    log_file: Path = Path("logging/simulation.log")
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "debug"

    @field_validator("log_file", mode="before")
    @classmethod
    def normalize_log_file(cls, value: str | Path) -> Path:
        return Path(value).expanduser()

    def python_level(self) -> int:
        return getattr(logging, self.log_level.upper())


class SimulatorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    inputs: SimulatorInputConfig
    probe: ProbeConfig
    runtime: SimulatorRuntimeConfig
    recorder: RecorderConfig
    monitoring: MonitoringConfig
    logging: LoggingConfig
