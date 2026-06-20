from evaluater.sim_type import TaskType

from recorder.compression_types import CompressedTrace

COMPUTE_TASK_TYPES = (
    TaskType.CONV,
    TaskType.POOL,
    TaskType.FC,
    TaskType.ELEM,
    TaskType.GCONV,
    TaskType.PTP,
    TaskType.TRANS,
)
COMMUNICATION_TASK_TYPES = (
    TaskType.SEND,
    TaskType.RECV,
)
IO_TASK_TYPES = (
    TaskType.READ,
    TaskType.WRITE,
)


def trace_to_key_attr(trace):
    if trace.instruction_type in COMPUTE_TASK_TYPES:
        key = f"pe{trace.pe_id}-flops{trace.flops}-layer{trace.layer_id}-inf{trace.inference_time}"
        attr = CompressedTrace(
            layer_id=trace.layer_id,
            pe_id=trace.pe_id,
            start_time=trace.start_time,
            end_time=trace.end_time,
            inference_time=trace.inference_time,
            flops=trace.flops,
        )
        return key, attr

    if trace.instruction_type in COMMUNICATION_TASK_TYPES:
        key = f"src{trace.src_id}-dst{trace.dst_id}-ds{trace.data_size}-inf{trace.inference_time}"
        attr = CompressedTrace(
            layer_id=trace.layer_id,
            pe_id=trace.pe_id,
            start_time=trace.start_time,
            end_time=trace.end_time,
            inference_time=trace.inference_time,
            data_size=trace.data_size,
            src_id=trace.src_id,
            dst_id=trace.dst_id,
        )
        return key, attr

    return None, None
