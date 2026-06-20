import hashlib

from recorder.compression_projection import COMMUNICATION_TASK_TYPES, IO_TASK_TYPES, trace_to_key_attr
from recorder.compression_types import (
    CompressedComp,
    CompressedComm,
    CompressedTrace,
    CompSummary,
    CommSummary,
    CompressionResult,
)
from recorder.storage_model import (
    CompressionRunResult,
    CompressionStorageTracker,
)

SKETCH_RETENTION_MODE = "sketch"
EXACT_RETENTION_MODE = "exact"

OLDEST_EVICTION_POLICY = "oldest"
NO_EVICTION_POLICY = "none"
OVERREPRESENTED_TRIPLET_EVICTION_POLICY = "overrepresented-triplet"

DEFAULT_EXACT_STAGE2_SIZE_FACTOR = 4


def available_retention_modes() -> tuple[str, ...]:
    return (SKETCH_RETENTION_MODE, EXACT_RETENTION_MODE)


def available_stage2_eviction_policies() -> tuple[str, ...]:
    return (OLDEST_EVICTION_POLICY, NO_EVICTION_POLICY, OVERREPRESENTED_TRIPLET_EVICTION_POLICY)


class Stage1Bucket:
    def __init__(self):
        self.key = None
        self.count = 0


class RunningTrack:
    def __init__(self, num_hashes=3, num_buckets=128, threshold=10, storage_tracker: CompressionStorageTracker | None = None):
        self.d = num_hashes
        self.m = num_buckets
        self.H = threshold
        self.tables = [[Stage1Bucket() for _ in range(self.m)] for _ in range(self.d)]
        self.storage_tracker = storage_tracker or CompressionStorageTracker(num_hashes, num_buckets, threshold, stage2_size=128)

    def _hashes(self, key: str):
        return [int(hashlib.md5((key + str(i)).encode()).hexdigest(), 16) % self.m for i in range(self.d)]

    def insert(self, key: str) -> bool:
        promoted = False
        for table_id, idx in enumerate(self._hashes(key)):
            bucket = self.tables[table_id][idx]
            if bucket.key == key:
                bucket.count += 1
                self.storage_tracker.observe_stage1_bucket(bucket.key, bucket.count)
                if bucket.count >= self.H:
                    promoted = True
            elif bucket.key is None:
                bucket.key = key
                bucket.count = 1
                self.storage_tracker.observe_stage1_bucket(bucket.key, bucket.count)
            else:
                bucket.count -= 1
                if bucket.count <= 0:
                    bucket.key = None
                    bucket.count = 0
        return promoted


class FailSlowPattern:
    def __init__(self, key: str, start_time: int, end_time: int, attr: CompressedTrace):
        self.key = key
        self.start_time = start_time
        self.end_time = end_time
        self.count = 1
        self.merged_attr = attr
        self.merged_attr.duration = attr.end_time - attr.start_time
        if attr.flops != -1 and self.merged_attr.duration > 0:
            self.merged_attr.flops = attr.flops / self.merged_attr.duration

    def update(self, start_time: int, end_time: int, attr: CompressedTrace):
        self.count += 1
        self.start_time = min(self.start_time, start_time)
        self.end_time = max(self.end_time, end_time)
        self.merged_attr.merge(attr)
        return self.count, self.end_time, self.merged_attr

    def summary(self):
        if self.merged_attr.flops != -1:
            return CompressedComp(
                pe_id=self.merged_attr.pe_id,
                layer_id=self.merged_attr.layer_id,
                start_time=self.start_time,
                end_time=self.end_time,
                inference_time=self.merged_attr.inference_time,
                count=self.count,
                flops=self.merged_attr.flops / self.count,
            )
        return CompressedComm(
            pe_id=self.merged_attr.pe_id,
            layer_id=self.merged_attr.layer_id,
            start_time=self.start_time,
            end_time=self.end_time,
            inference_time=self.merged_attr.inference_time,
            count=self.count,
            avg_time=self.merged_attr.duration / self.count,
            data_size=self.merged_attr.data_size,
            src_id=self.merged_attr.src_id,
            dst_id=self.merged_attr.dst_id,
        )

    def communication_family(self):
        if self.merged_attr.data_size == -1:
            return None
        return (self.merged_attr.src_id, self.merged_attr.dst_id, self.merged_attr.data_size)


class SnapshotTable:
    def __init__(
        self,
        max_size=128,
        eviction_policy: str = OLDEST_EVICTION_POLICY,
        storage_tracker: CompressionStorageTracker | None = None,
    ):
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.table = {}
        self.storage_tracker = storage_tracker or CompressionStorageTracker(num_hashes=3, num_buckets=128, threshold=10, stage2_size=max_size)

    def insert(self, key: str, start_time: int, end_time: int, attr: CompressedTrace):
        if key not in self.table:
            if self.max_size is not None and len(self.table) >= self.max_size and self.eviction_policy != NO_EVICTION_POLICY:
                self._evict()
            self.table[key] = FailSlowPattern(key, start_time, end_time, attr)
            self.storage_tracker.observe_stage2_insert(key, start_time, end_time, attr)
            self.storage_tracker.observe_stage2_capacity(len(self.table))
            return

        new_count, new_end_time, new_attr = self.table[key].update(start_time, end_time, attr)
        self.storage_tracker.observe_stage2_update(new_count, new_end_time, new_attr)

    def _evict(self):
        if not self.table:
            return
        if self.eviction_policy == OVERREPRESENTED_TRIPLET_EVICTION_POLICY:
            self._evict_overrepresented_triplet()
            return
        self._evict_oldest()

    def _evict_oldest(self):
        oldest_key = min(self.table.keys(), key=lambda key: self.table[key].start_time)
        del self.table[oldest_key]

    def _evict_overrepresented_triplet(self):
        family_to_keys = {}
        for key, pattern in self.table.items():
            family = pattern.communication_family()
            if family is None:
                continue
            family_to_keys.setdefault(family, []).append((key, pattern))

        if not family_to_keys:
            self._evict_oldest()
            return

        max_family_size = max(len(items) for items in family_to_keys.values())
        if max_family_size <= 1:
            self._evict_oldest()
            return

        candidate_members = []
        for members in family_to_keys.values():
            if len(members) == max_family_size:
                candidate_members.extend(members)

        victim_key, _ = min(candidate_members, key=lambda item: (item[1].start_time, item[1].end_time))
        del self.table[victim_key]

    def get_summaries(self):
        comm_compressed_trace = []
        comp_compressed_trace = []
        for pattern in self.table.values():
            summary = pattern.summary()
            if isinstance(summary, CompressedComp):
                comp_compressed_trace.append(summary)
            else:
                comm_compressed_trace.append(summary)
        return comm_compressed_trace, comp_compressed_trace


class FailSlowCompressor:
    def __init__(
        self,
        num_hashes=3,
        num_buckets=128,
        stage2_size=128,
        threshold=10,
        stage2_eviction_policy: str = OLDEST_EVICTION_POLICY,
    ):
        self.configured_stage2_size = stage2_size
        self.storage_tracker = CompressionStorageTracker(num_hashes, num_buckets, threshold, stage2_size)
        self.stage1 = RunningTrack(num_hashes, num_buckets, threshold, storage_tracker=self.storage_tracker)
        self.stage2 = SnapshotTable(stage2_size, eviction_policy=stage2_eviction_policy, storage_tracker=self.storage_tracker)

    def insert(self, key: str, start_time: int, end_time: int, attr: CompressedTrace, always_promote: bool = False):
        if always_promote:
            self.stage2.insert(key, start_time, end_time, attr)
            return
        if self.stage1.insert(key):
            self.stage2.insert(key, start_time, end_time, attr)

    def result(self) -> CompressionRunResult:
        comm, comp = self.stage2.get_summaries()
        compression = CompressionResult(
            comm=CommSummary(trace=comm),
            comp=CompSummary(trace=comp),
        )
        storage_model = self.storage_tracker.finalize(compression)
        effective_stage2_size = self.configured_stage2_size
        if self.configured_stage2_size is None or self.stage2.eviction_policy == NO_EVICTION_POLICY:
            effective_stage2_size = max(len(self.stage2.table), self.configured_stage2_size or 0)
        return CompressionRunResult(
            compression=compression,
            storage_model=storage_model,
            effective_stage2_size=effective_stage2_size,
        )


def compress_traces(
    comm_trace,
    comp_trace,
    num_hashes=5,
    num_buckets=1024,
    stage2_size=8192,
    threshold=10,
    comm_retention_mode: str = SKETCH_RETENTION_MODE,
    stage2_eviction_policy: str = OLDEST_EVICTION_POLICY,
    comm_stage2_size: int | None = None,
) -> CompressionRunResult:
    resolved_stage2_size = stage2_size
    if comm_retention_mode == EXACT_RETENTION_MODE:
        if stage2_eviction_policy == NO_EVICTION_POLICY and comm_stage2_size is None:
            resolved_stage2_size = None
        elif comm_stage2_size is not None:
            resolved_stage2_size = comm_stage2_size
        else:
            resolved_stage2_size = stage2_size * DEFAULT_EXACT_STAGE2_SIZE_FACTOR

    compressor = FailSlowCompressor(
        num_hashes=num_hashes,
        num_buckets=num_buckets,
        stage2_size=resolved_stage2_size,
        threshold=threshold,
        stage2_eviction_policy=stage2_eviction_policy,
    )

    for trace in comm_trace.trace:
        if trace.instruction_type in IO_TASK_TYPES:
            continue
        key, attr = trace_to_key_attr(trace)
        if key is None or attr is None:
            continue
        always_promote = (
            comm_retention_mode == EXACT_RETENTION_MODE
            and trace.instruction_type in COMMUNICATION_TASK_TYPES
        )
        compressor.insert(key, attr.start_time, attr.end_time, attr, always_promote=always_promote)

    for trace in comp_trace.trace:
        if trace.instruction_type in IO_TASK_TYPES:
            continue
        key, attr = trace_to_key_attr(trace)
        if key is None or attr is None:
            continue
        compressor.insert(key, attr.start_time, attr.end_time, attr)

    return compressor.result()
