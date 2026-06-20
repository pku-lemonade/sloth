from dataclasses import dataclass

from pydantic import BaseModel, Field

from recorder.compression_types import CompressedTrace, CompressionResult

DEFAULT_COMM_TRACE_SIZE_MB = 2.861328125
DEFAULT_COMP_TRACE_SIZE_MB = 1.767578125
BYTES_PER_MEGABYTE = 1024 * 1024


def size_of_int_bytes(value) -> int:
    if value < 256:
        return 1
    if value < 65536:
        return 2
    if value < 4294967296:
        return 4
    return 8


def size_of_str_bytes(value: str | None) -> int:
    if value is None:
        return 16
    return len(value.encode("utf-8"))


class AttributeFootprint(BaseModel):
    layer_id: int = 1
    pe_id: int = 1
    start_time: int = 1
    end_time: int = 1
    inference_time: int = 1
    flops: int = 1
    data_size: int = 1
    src_id: int = 1
    dst_id: int = 1
    duration: int = 1

    def merge_max(self, other: "AttributeFootprint") -> None:
        for field_name in self.model_fields:
            setattr(self, field_name, max(getattr(self, field_name), getattr(other, field_name)))

    def update_from_trace(self, attr: CompressedTrace) -> None:
        self.layer_id = max(self.layer_id, size_of_int_bytes(attr.layer_id))
        self.pe_id = max(self.pe_id, size_of_int_bytes(attr.pe_id))
        self.start_time = max(self.start_time, size_of_int_bytes(attr.start_time))
        self.end_time = max(self.end_time, size_of_int_bytes(attr.end_time))
        self.inference_time = max(self.inference_time, size_of_int_bytes(attr.inference_time))
        self.flops = max(self.flops, size_of_int_bytes(attr.flops))
        self.data_size = max(self.data_size, size_of_int_bytes(attr.data_size))
        self.src_id = max(self.src_id, size_of_int_bytes(attr.src_id))
        self.dst_id = max(self.dst_id, size_of_int_bytes(attr.dst_id))
        self.duration = max(self.duration, size_of_int_bytes(attr.duration))

    def total_units(self) -> int:
        return sum(self.model_dump().values())


class Stage1Footprint(BaseModel):
    d: int = 1
    m: int = 1
    H: int = 1
    tables_key: int = 0
    tables_count: int = 1

    def merge_max(self, other: "Stage1Footprint") -> None:
        for field_name in self.model_fields:
            setattr(self, field_name, max(getattr(self, field_name), getattr(other, field_name)))


class Stage2Footprint(BaseModel):
    max_size: int = 1
    table_key: int = 0
    table_value_s_time: int = 1
    table_value_e_time: int = 1
    table_value_count: int = 1
    table_value_attr: AttributeFootprint = Field(default_factory=AttributeFootprint)

    def merge_max(self, other: "Stage2Footprint") -> None:
        self.max_size = max(self.max_size, other.max_size)
        self.table_key = max(self.table_key, other.table_key)
        self.table_value_s_time = max(self.table_value_s_time, other.table_value_s_time)
        self.table_value_e_time = max(self.table_value_e_time, other.table_value_e_time)
        self.table_value_count = max(self.table_value_count, other.table_value_count)
        self.table_value_attr.merge_max(other.table_value_attr)

    def total_units(self) -> int:
        return (
            self.table_key
            + self.table_value_s_time
            + self.table_value_e_time
            + self.table_value_count
            + self.table_value_attr.total_units()
        )


class SummaryFootprint(BaseModel):
    pe_id: int = 1
    layer_id: int = 1
    start_time: int = 1
    end_time: int = 1
    inference_time: int = 1
    count: int = 1
    trace_sum: int = 0

    def merge_max(self, other: "SummaryFootprint") -> None:
        for field_name in self.model_fields:
            setattr(self, field_name, max(getattr(self, field_name), getattr(other, field_name)))

    def update_common(self, item) -> None:
        self.pe_id = max(self.pe_id, size_of_int_bytes(item.pe_id))
        self.layer_id = max(self.layer_id, size_of_int_bytes(item.layer_id))
        self.start_time = max(self.start_time, size_of_int_bytes(item.start_time))
        self.end_time = max(self.end_time, size_of_int_bytes(item.end_time))
        self.inference_time = max(self.inference_time, size_of_int_bytes(item.inference_time))
        self.count = max(self.count, size_of_int_bytes(item.count))

    def per_trace_units(self) -> int:
        values = self.model_dump().copy()
        values.pop("trace_sum", None)
        return sum(values.values())

    def size_mb(self) -> float:
        return self.per_trace_units() * self.trace_sum / BYTES_PER_MEGABYTE


class CommSummaryFootprint(SummaryFootprint):
    avg_time: int = 4
    data_size: int = 1
    src_id: int = 1
    dst_id: int = 1

    @classmethod
    def from_summary(cls, summary) -> "CommSummaryFootprint":
        footprint = cls()
        for item in summary.trace:
            footprint.update_common(item)
            footprint.data_size = max(footprint.data_size, size_of_int_bytes(item.data_size))
            footprint.src_id = max(footprint.src_id, size_of_int_bytes(item.src_id))
            footprint.dst_id = max(footprint.dst_id, size_of_int_bytes(item.dst_id))
        footprint.trace_sum = max(footprint.trace_sum, len(summary.trace))
        return footprint


class CompSummaryFootprint(SummaryFootprint):
    flops: int = 4

    @classmethod
    def from_summary(cls, summary) -> "CompSummaryFootprint":
        footprint = cls()
        for item in summary.trace:
            footprint.update_common(item)
        footprint.trace_sum = max(footprint.trace_sum, len(summary.trace))
        return footprint


class CompressionStorageResult(BaseModel):
    stage1: Stage1Footprint
    stage2: Stage2Footprint
    compressed_comm: CommSummaryFootprint
    compressed_comp: CompSummaryFootprint


@dataclass(frozen=True)
class CompressionRunResult:
    compression: CompressionResult
    storage_model: CompressionStorageResult
    effective_stage2_size: int


class CompressionStorageTracker:
    def __init__(self, num_hashes: int, num_buckets: int, threshold: int, stage2_size: int | None):
        self.stage1 = Stage1Footprint(
            d=size_of_int_bytes(num_hashes),
            m=size_of_int_bytes(num_buckets),
            H=size_of_int_bytes(threshold),
        )
        initial_stage2_size = 1 if stage2_size is None else size_of_int_bytes(stage2_size)
        self.stage2 = Stage2Footprint(max_size=initial_stage2_size)

    def observe_stage1_bucket(self, key: str | None, count: int) -> None:
        self.stage1.tables_key = max(self.stage1.tables_key, size_of_str_bytes(key))
        self.stage1.tables_count = max(self.stage1.tables_count, size_of_int_bytes(count))

    def observe_stage2_insert(self, key: str, start_time: int, end_time: int, attr: CompressedTrace) -> None:
        self.stage2.table_key = max(self.stage2.table_key, size_of_str_bytes(key))
        self.stage2.table_value_s_time = max(self.stage2.table_value_s_time, size_of_int_bytes(start_time))
        self.stage2.table_value_e_time = max(self.stage2.table_value_e_time, size_of_int_bytes(end_time))
        self.stage2.table_value_attr.update_from_trace(attr)

    def observe_stage2_update(self, count: int, end_time: int, attr: CompressedTrace) -> None:
        self.stage2.table_value_count = max(self.stage2.table_value_count, size_of_int_bytes(count))
        self.stage2.table_value_e_time = max(self.stage2.table_value_e_time, size_of_int_bytes(end_time))
        self.stage2.table_value_attr.update_from_trace(attr)

    def observe_stage2_capacity(self, entry_count: int) -> None:
        self.stage2.max_size = max(self.stage2.max_size, size_of_int_bytes(max(entry_count, 1)))

    def finalize(self, compression_result: CompressionResult) -> CompressionStorageResult:
        return CompressionStorageResult(
            stage1=self.stage1.model_copy(deep=True),
            stage2=self.stage2.model_copy(deep=True),
            compressed_comm=CommSummaryFootprint.from_summary(compression_result.comm),
            compressed_comp=CompSummaryFootprint.from_summary(compression_result.comp),
        )


class CompressionStorageRecord(BaseModel):
    stage1: Stage1Footprint = Field(default_factory=Stage1Footprint)
    stage2: Stage2Footprint = Field(default_factory=Stage2Footprint)
    compressed_comm: CommSummaryFootprint = Field(default_factory=CommSummaryFootprint)
    compressed_comp: CompSummaryFootprint = Field(default_factory=CompSummaryFootprint)

    def merge(self, storage_result: CompressionStorageResult) -> None:
        self.stage1.merge_max(storage_result.stage1)
        self.stage2.merge_max(storage_result.stage2)
        self.compressed_comm.merge_max(storage_result.compressed_comm)
        self.compressed_comp.merge_max(storage_result.compressed_comp)

    def structure_overhead_mb(self, num_hashes: int, num_buckets: int, stage2_size: int) -> float:
        total = self.stage1.d + self.stage1.m + self.stage1.H
        total += (self.stage1.tables_key + self.stage1.tables_count) * num_hashes * num_buckets
        total += self.stage2.max_size
        total += self.stage2.total_units() * stage2_size
        return total / BYTES_PER_MEGABYTE

    def compressed_sizes_mb(self) -> tuple[float, float]:
        return self.compressed_comm.size_mb(), self.compressed_comp.size_mb()

    def to_record_payload(self) -> dict:
        return {
            "data_structure": {
                "stage1": self.stage1.model_dump(),
                "stage2": self.stage2.model_dump(),
            },
            "compressed_comm": self.compressed_comm.model_dump(),
            "compressed_comp": self.compressed_comp.model_dump(),
        }

    @classmethod
    def from_record_payload(cls, payload: dict) -> "CompressionStorageRecord":
        data_structure = payload.get("data_structure", {})
        return cls(
            stage1=Stage1Footprint.model_validate(data_structure.get("stage1", {})),
            stage2=Stage2Footprint.model_validate(data_structure.get("stage2", {})),
            compressed_comm=CommSummaryFootprint.model_validate(payload.get("compressed_comm", {})),
            compressed_comp=CompSummaryFootprint.model_validate(payload.get("compressed_comp", {})),
        )


class CompressionAggregateMetrics(BaseModel):
    overhead: float = 0
    comm_trace_size: float = DEFAULT_COMM_TRACE_SIZE_MB
    comp_trace_size: float = DEFAULT_COMP_TRACE_SIZE_MB
    compressed_comm_size: float = 0
    compressed_comp_size: float = 0
    rate: float = 1

    def update(self, record: CompressionStorageRecord, num_hashes: int, num_buckets: int, stage2_size: int) -> None:
        self.overhead = max(self.overhead, record.structure_overhead_mb(num_hashes, num_buckets, stage2_size))
        compressed_comm_size, compressed_comp_size = record.compressed_sizes_mb()
        self.compressed_comm_size = max(self.compressed_comm_size, compressed_comm_size)
        self.compressed_comp_size = max(self.compressed_comp_size, compressed_comp_size)
        total_trace_size = self.comm_trace_size + self.comp_trace_size
        if total_trace_size > 0:
            self.rate = (self.compressed_comm_size + self.compressed_comp_size) / total_trace_size

    def to_output_payload(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_output_payload(cls, payload: dict) -> "CompressionAggregateMetrics":
        return cls.model_validate(payload)
