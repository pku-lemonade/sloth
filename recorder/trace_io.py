import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from recorder.compression_types import CommSummary, CompSummary
from recorder.trace_format import CommTrace, CompTrace, LayerGroupsInfo, LinksData

T = TypeVar("T", bound=BaseModel)


def load_json_model(filename: str, model_type: type[T]) -> T:
    with Path(filename).open("r", encoding="utf-8") as file:
        data = json.load(file)
        try:
            return model_type.model_validate(data)
        except ValidationError as exc:
            print(exc.json())
            raise


def load_comp_trace(filename: str) -> CompTrace:
    return load_json_model(filename, CompTrace)


def load_comm_trace(filename: str) -> CommTrace:
    return load_json_model(filename, CommTrace)


def load_comp_summary(filename: str) -> CompSummary:
    return load_json_model(filename, CompSummary)


def load_comm_summary(filename: str) -> CommSummary:
    return load_json_model(filename, CommSummary)


def load_links_data(filename: str) -> LinksData:
    return load_json_model(filename, LinksData)


def load_layer_groups(filename: str) -> LayerGroupsInfo:
    return load_json_model(filename, LayerGroupsInfo)


comp_analyzer = load_comp_trace
comm_analyzer = load_comm_trace
link_analyzer = load_links_data
layer_group_analyzer = load_layer_groups
