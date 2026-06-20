import json
from dataclasses import dataclass
from pathlib import Path

from evaluater.sim_type import INST_OFFSET, TaskType
from recorder.compression_projection import IO_TASK_TYPES, trace_to_key_attr
from recorder.compression_types import CompressionResult
from recorder.sketch_compressor import (
    EXACT_RETENTION_MODE,
    NO_EVICTION_POLICY,
    OLDEST_EVICTION_POLICY,
    SKETCH_RETENTION_MODE,
    FailSlowCompressor,
)
from recorder.storage_model import CompressionAggregateMetrics, CompressionStorageRecord
from recorder.trace_format import CommInst, CompInst

TRACE_FORMAT_COMPRESSED = "compressed"


@dataclass(frozen=True)
class OnlineRecorderConfig:
    num_hashes: int = 5
    num_buckets: int = 1024
    stage2_size: int = 8192
    threshold: int = 10
    comm_retention_mode: str = SKETCH_RETENTION_MODE
    stage2_eviction_policy: str = OLDEST_EVICTION_POLICY
    comm_stage2_size: int | None = None

    @property
    def effective_stage2_size(self) -> int | None:
        if self.comm_retention_mode != EXACT_RETENTION_MODE:
            return self.stage2_size
        if self.stage2_eviction_policy == NO_EVICTION_POLICY and self.comm_stage2_size is None:
            return None
        if self.comm_stage2_size is not None:
            return self.comm_stage2_size
        return self.stage2_size * 4

    def to_metadata(self) -> dict:
        return {
            "format": TRACE_FORMAT_COMPRESSED,
            "compression": {
                "hash": self.num_hashes,
                "bucket": self.num_buckets,
                "size": self.stage2_size,
                "effective_size": self.effective_stage2_size,
                "threshold": self.threshold,
                "comm_retention_mode": self.comm_retention_mode,
                "stage2_eviction_policy": self.stage2_eviction_policy,
                "comm_stage2_size": self.comm_stage2_size,
            },
        }


class OnlineTraceRecorder:
    def __init__(self, config: OnlineRecorderConfig | None = None):
        self.config = config or OnlineRecorderConfig()
        self.compressor = FailSlowCompressor(
            num_hashes=self.config.num_hashes,
            num_buckets=self.config.num_buckets,
            stage2_size=self.config.effective_stage2_size,
            threshold=self.config.threshold,
            stage2_eviction_policy=self.config.stage2_eviction_policy,
        )
        self._recorded_compute: set[int] = set()
        self._recorded_communication: set[int] = set()
        self._pending_send_metrics: dict[int, dict] = {}
        self._pending_recv_metrics: dict[int, dict] = {}

    def observe_probe_metrics(self, inst_index: int, metrics: dict) -> None:
        instruction_type = metrics.get("instruction_type")
        if instruction_type is None:
            return

        if instruction_type in (TaskType.CONV, TaskType.POOL, TaskType.FC, TaskType.ELEM, TaskType.GCONV, TaskType.PTP, TaskType.TRANS):
            self._try_record_compute(inst_index, metrics)
            return

        if instruction_type == TaskType.SEND:
            self._pending_send_metrics[inst_index] = dict(metrics)
            self._try_record_communication(inst_index)
            return

        if instruction_type == TaskType.RECV:
            self._pending_recv_metrics[inst_index] = dict(metrics)
            self._try_record_communication(inst_index)

    def _try_record_compute(self, inst_index: int, metrics: dict) -> None:
        if inst_index in self._recorded_compute:
            return
        required = ("instruction_id", "instruction_type", "layer_id", "pe_id", "start_time", "end_time", "flops")
        if any(metrics.get(field) is None for field in required):
            return

        self.record_compute(
            CompInst(
                instruction_id=metrics["instruction_id"],
                instruction_type=metrics["instruction_type"],
                layer_id=metrics["layer_id"],
                pe_id=metrics["pe_id"],
                start_time=metrics["start_time"],
                end_time=metrics["end_time"],
                inference_time=metrics["instruction_id"] // INST_OFFSET,
                flops=metrics["flops"],
            )
        )
        self._recorded_compute.add(inst_index)

    def _try_record_communication(self, inst_index: int) -> None:
        if inst_index in self._recorded_communication:
            return
        send_metrics = self._pending_send_metrics.get(inst_index)
        recv_metrics = self._pending_recv_metrics.get(inst_index)
        if not send_metrics or not recv_metrics:
            return

        send_required = ("end_time", "data_size", "src_id")
        recv_required = ("instruction_type", "layer_id", "pe_id", "start_time", "dst_id")
        if any(send_metrics.get(field) is None for field in send_required):
            return
        if any(recv_metrics.get(field) is None for field in recv_required):
            return

        instruction_id = recv_metrics.get("instruction_id", inst_index)
        self.record_communication(
            CommInst(
                instruction_id=inst_index,
                instruction_type=recv_metrics["instruction_type"],
                layer_id=recv_metrics["layer_id"],
                pe_id=recv_metrics["pe_id"],
                start_time=send_metrics["end_time"],
                end_time=recv_metrics["start_time"],
                inference_time=instruction_id // INST_OFFSET,
                data_size=send_metrics["data_size"],
                src_id=send_metrics["src_id"],
                dst_id=recv_metrics["dst_id"],
            )
        )
        self._recorded_communication.add(inst_index)

    def record_compute(self, trace: CompInst) -> None:
        self._record(trace, always_promote=False)

    def record_communication(self, trace: CommInst) -> None:
        always_promote = self.config.comm_retention_mode == EXACT_RETENTION_MODE
        self._record(trace, always_promote=always_promote)

    def _record(self, trace, always_promote: bool) -> None:
        if trace.instruction_type in IO_TASK_TYPES:
            return
        key, attr = trace_to_key_attr(trace)
        if key is None or attr is None:
            return
        self.compressor.insert(key, attr.start_time, attr.end_time, attr, always_promote=always_promote)

    def result(self):
        return self.compressor.result()

    def write_outputs(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        run_result = self.result()
        compression: CompressionResult = run_result.compression

        with (output_path / "comm_trace.json").open("w", encoding="utf-8") as file:
            print(compression.comm.model_dump_json(indent=4), file=file)
        with (output_path / "comp_trace.json").open("w", encoding="utf-8") as file:
            print(compression.comp.model_dump_json(indent=4), file=file)

        storage_record = CompressionStorageRecord()
        storage_record.merge(run_result.storage_model)
        with (output_path / "record.json").open("w", encoding="utf-8") as file:
            json.dump(storage_record.to_record_payload(), file, indent=4)

        aggregate_metrics = CompressionAggregateMetrics()
        aggregate_metrics.update(
            storage_record,
            self.config.num_hashes,
            self.config.num_buckets,
            run_result.effective_stage2_size or 0,
        )
        with (output_path / "overall.json").open("w", encoding="utf-8") as file:
            json.dump(aggregate_metrics.to_output_payload(), file, indent=4)

        with (output_path / "trace_meta.json").open("w", encoding="utf-8") as file:
            json.dump(self.config.to_metadata(), file, indent=4)
