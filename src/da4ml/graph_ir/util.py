from typing import Any, Tuple
from dataclasses import dataclass

import keras
import numpy as np

from da4ml.codegen.rtl.rtl_model import get_io_kifs
from da4ml.cmvm.types import CombLogic

@dataclass
class OpRepr:
    operation: keras.Operation
    args: list
    kwargs: dict
    produces: Tuple[keras.KerasTensor, ...]
    requires: Tuple[keras.KerasTensor, ...]


def _strip_batch(shape: Any) -> tuple[int, ...]:
    """
    Convert KerasTensor shape into tuple[int,...] with batch removed.
    Typically KerasTensor.shape looks like (None, d1, d2, ...)
    """
    if shape is None:
        return tuple()
    # shape may be TensorShape-like; tuple() makes it concrete
    shp = tuple(shape)
    if len(shp) == 0:
        return tuple()
    if shp[0] is None:
        shp = shp[1:]
    # Ensure all remaining dims are ints or None; keep None if present
    return tuple(shp)


def _ensure_tuple_ints(shape: tuple[Any, ...]) -> tuple[int, ...]:
    """
    Best-effort conversion to tuple[int,...] where possible.
    If a dim is None, we keep None out by raising (because your schedules assume ints).
    """
    out: list[int] = []
    for d in shape:
        if d is None:
            raise ValueError(f"Encountered dynamic/None dim in shape {shape}. "
                             f"Your scheduling/min-shape logic assumes static ints.")
        out.append(int(d))
    return tuple(out)

def _strip_batch_and_ensure_ints(shape: Any) -> tuple[int, ...]: 
    return _ensure_tuple_ints(_strip_batch(shape))


def _flatten_ops(parsed: list[list[OpRepr]]) -> list[OpRepr]:
    ops: list[OpRepr] = []
    for group in parsed:
        ops.extend(group)
    return ops

def _short_tid(tid: int, n: int = 6) -> str:
    s = str(tid)
    return s[-n:] if len(s) > n else s

def short_tid(tid: int, n: int = 6) -> str:
    return _short_tid(tid, n)

def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')

def get_bitwidth_from_cl(cs: CombLogic):
    inp_kif, output_kif = get_io_kifs(cs)
    inp_bitwidth = sum(np.max(arr) for arr in inp_kif)
    output_bitwidth = sum(np.max(arr) for arr in output_kif)
    return int(inp_bitwidth), int(output_bitwidth)



def parse_model(model: keras.Model) -> list[list[OpRepr]]:
    if isinstance(model, keras.Sequential):
        model = model._functional

    operators: dict[int, list[OpRepr]] = {}
    for depth, nodes in model._nodes_by_depth.items():
        _oprs: list[OpRepr] = []
        for node in nodes:
            assert isinstance(node.operation, keras.Operation)
            opr = OpRepr(
                operation=node.operation,
                args=node.arguments.args,
                kwargs=node.arguments.kwargs,
                produces=tuple(node.outputs),
                requires=tuple(node.arguments.keras_tensors),
            )
            _oprs.append(opr)
        operators[depth] = _oprs
    return [operators[i] for i in range(max(operators.keys()), -1, -1)]
