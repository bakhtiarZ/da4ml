from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Set

import keras

from da4ml.cmvm.types import CombLogic
from da4ml.trace import FixedVariableArrayInput, comb_trace
from da4ml.converter import trace_model

from .scheduling import DataSchedule, _SCHEDULE_REGISTRY


@dataclass
class OpRepr:
    operation: keras.Operation
    args: list
    kwargs: dict
    produces: Tuple[keras.KerasTensor, ...]
    requires: Tuple[keras.KerasTensor, ...]


def parse_model(model: keras.Model) -> List[List[OpRepr]]:
    if isinstance(model, keras.Sequential):
        model = model._functional

    operators: Dict[int, List[OpRepr]] = {}
    for depth, nodes in model._nodes_by_depth.items():
        _oprs: List[OpRepr] = []
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

@dataclass
class PureLogic:
    empty_logic: bool = True  # marker type

@dataclass
class RoutingLogic:
    to_hardware: Callable[[Any], Any]
    schedule: Optional[Any] = None
    buffer_type: str = "default"
    buffer_shape: int = 0


@dataclass
class RoutingEdge:
    tid: int
    tensor: Optional[keras.KerasTensor] = None
    routing_logic: Optional[RoutingLogic] = None
    semantic_shape: Optional[Tuple[int, ...]] = None
    from_node: Optional[int] = None
    from_compute_shape: Optional[Tuple[int, ...]] = None
    to_nodes: Set[int] = field(default_factory=set)
    to_compute_shapes: Dict[int, Tuple[int, ...]] = field(default_factory=dict)

@dataclass
class LogicNode:
    op_id: int
    operation: Optional[keras.Operation]
    op_repr: Optional[OpRepr]

    logic_impl: CombLogic | PureLogic

    input_tids: List[int]
    output_tids: List[int]

    input_shapes: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    output_shapes: Dict[int, Tuple[int, ...]] = field(default_factory=dict)


@dataclass
class LRGraph:
    logic_nodes: Dict[int, LogicNode]
    routing_edges: Dict[int, RoutingEdge]
    model_input_tids: List[int] = field(default_factory=list)
    model_output_tids: List[int] = field(default_factory=list)


def _strip_batch(shape: Any) -> Tuple[int, ...]:
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


def _ensure_tuple_ints(shape: Tuple[Any, ...]) -> Tuple[int, ...]:
    """
    Best-effort conversion to tuple[int,...] where possible.
    If a dim is None, we keep None out by raising (because your schedules assume ints).
    """
    out: List[int] = []
    for d in shape:
        if d is None:
            raise ValueError(f"Encountered dynamic/None dim in shape {shape}. "
                             f"Your scheduling/min-shape logic assumes static ints.")
        out.append(int(d))
    return tuple(out)

def _strip_batch_and_ensure_ints(shape: Any) -> Tuple[int, ...]: 
    return _ensure_tuple_ints(_strip_batch(shape))

def _min_shapes_for_op(
    opr: OpRepr,
    schedule: Optional[DataSchedule],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Decide minimal input/output shapes for this op, including batch=1.
    Uses the first input/output tensor shapes as representative
    """
    if opr.operation.__class__ is keras.layers.InputLayer or not opr.requires:
        out_no_batch_int = _strip_batch_and_ensure_ints(opr.produces[0].shape) 
        return out_no_batch_int, out_no_batch_int

    in_no_batch_int = _strip_batch_and_ensure_ints(opr.requires[0].shape)
    out_no_batch_int = _strip_batch_and_ensure_ints(opr.produces[0].shape)
    
    if schedule is None:
        return in_no_batch_int, out_no_batch_int

    min_in_no_batch = schedule.minimum_input_shape(in_no_batch_int)
    min_out_no_batch = schedule.minimum_output_shape(out_no_batch_int)

    min_in = tuple(min_in_no_batch)
    min_out = tuple(min_out_no_batch)
    return min_in, min_out


def create_comb_logic_from_oprepr(
    opr: OpRepr,
    schedule: Optional[DataSchedule],
) -> Tuple[CombLogic | PureLogic, Tuple[int, ...], Tuple[int, ...]]:
    # InputLayer special-case
    if opr.operation.__class__ is keras.layers.InputLayer:
        out_no_batch_int = _strip_batch_and_ensure_ints(opr.produces[0].shape)
        shp = out_no_batch_int
        return PureLogic(), shp, shp

    min_in_shape, min_out_shape = _min_shapes_for_op(opr, schedule)

    # Build a minimal model around this operation
    min_inp = keras.Input(shape=min_in_shape) # Keras Input excludes batch dimension, but our shapes include batch=1, so we use min_in_shape[1:] here
    out = opr.operation(min_inp)
    min_model = keras.Model(min_inp, out)

    # Trace
    fx_inp = FixedVariableArrayInput(shape=min_in_shape)  # includes batch=1
    tr_in, tr_out = trace_model(min_model, inputs=fx_inp)
    logic = comb_trace(tr_in, tr_out)

    # Validate output shape if possible (remove batch from out.shape)
    out_no_batch_int = _strip_batch_and_ensure_ints(out.shape)
    if out_no_batch_int != min_out_shape:
        raise AssertionError(
            f"Output shape mismatch for {opr.operation} "
            f"got={out_no_batch_int} expected={min_out_shape}"
        )

    return logic, min_in_shape, min_out_shape


def _flatten_ops(parsed: List[List[OpRepr]]) -> List[OpRepr]:
    ops: List[OpRepr] = []
    for group in parsed:
        ops.extend(group)
    return ops


def build_lr_graph_from_parsed(
    parsed: List[List[OpRepr]],
    *,
    model_inputs: Optional[Sequence[keras.KerasTensor]] = None,
    model_outputs: Optional[Sequence[keras.KerasTensor]] = None,
) -> LRGraph:
    """
    Build LRGraph directly from parsed ops.
    - Assigns op_id in traversal order
    - Creates one RoutingEdge per tensor tid (id(tensor)), storing tensor
    - Adds a pure output node that consumes model outputs
    """
    lr = LRGraph(logic_nodes={}, routing_edges={})

    if model_inputs is not None:
        lr.model_input_tids = [id(t) for t in model_inputs]
    if model_outputs is not None:
        lr.model_output_tids = [id(t) for t in model_outputs]

    ops = _flatten_ops(parsed)

    def get_edge(t: keras.KerasTensor) -> RoutingEdge:
        tid = id(t)
        e = lr.routing_edges.get(tid)
        if e is None:
            e = RoutingEdge(tid=tid, tensor=t)
            lr.routing_edges[tid] = e
        else:
            if e.tensor is None:
                e.tensor = t

        if e.tensor is not None and e.semantic_shape is None:
            e.semantic_shape = _strip_batch_and_ensure_ints(e.tensor.shape)

        return e

    for op_id, opr in enumerate(ops):
        schedule: Optional[DataSchedule] = _SCHEDULE_REGISTRY.get(type(opr.operation))

        in_ts = list(opr.requires or ())
        out_ts = list(opr.produces or ())

        in_tids = [id(t) for t in in_ts]
        out_tids = [id(t) for t in out_ts]

        logic_impl, min_in_shape, min_out_shape = create_comb_logic_from_oprepr(opr, schedule)

        node = LogicNode(
            op_id=op_id,
            operation=opr.operation,
            op_repr=opr,
            logic_impl=logic_impl,
            input_tids=in_tids,
            output_tids=out_tids,
            input_shapes={tid: min_in_shape for tid in in_tids},
            output_shapes={tid: min_out_shape for tid in out_tids},
        )
        lr.logic_nodes[op_id] = node

        # Wire inputs (tensor -> this op)
        for t in in_ts:
            e = get_edge(t)
            # fan-out supported: same tensor may feed multiple ops
            e.to_nodes.add(op_id)
            e.to_compute_shapes[op_id] = node.input_shapes[id(t)]

        # Wire outputs (this op -> tensor)
        for t in out_ts:
            e = get_edge(t)
            if e.from_node is not None and e.from_node != op_id:
                raise ValueError(
                    f"Tensor {getattr(t, 'name', e.tid)} has multiple producers: "
                    f"{e.from_node} and {op_id}"
                )
            e.from_node = op_id
            e.from_compute_shape = node.output_shapes[id(t)]

    # Append a pure output node that consumes model outputs
    if lr.model_output_tids:
        append_pure_output_node(lr, output_tids=lr.model_output_tids)

    return lr


def build_lr_graph_from_model(model: keras.Model) -> LRGraph:
    """
    Convenience wrapper that calls parse_model(model) then builds LRGraph.
    """
    if isinstance(model, keras.Sequential):
        model = model._functional

    parsed = parse_model(model)
    return build_lr_graph_from_parsed(
        parsed,
        model_inputs=model.inputs,
        model_outputs=model.outputs,
    )


def append_pure_output_node(lr: LRGraph, *, output_tids: List[int]) -> int:
    """
    Create a final PureLogic node that consumes the model outputs.
    """
    if not output_tids:
        raise ValueError("append_pure_output_node: output_tids was empty")

    out_node_id = (max(lr.logic_nodes.keys()) + 1) if lr.logic_nodes else 0

    input_shapes: Dict[int, Tuple[int, ...]] = {}

    for tid in output_tids:
        e = lr.routing_edges.get(tid)
        if e is None:
            raise ValueError(f"Output tid {tid} not found in routing_edges.")
        if e.from_node is None or e.from_compute_shape is None:
            raise ValueError(
                f"Output tid {tid} has no producer/shape. "
                f"Have you built LRGraph before appending output?"
            )

        e.to_nodes.add(out_node_id)
        to_shape = e.semantic_shape if e.semantic_shape is not None else e.from_compute_shape
        e.to_compute_shapes[out_node_id] = to_shape
        input_shapes[tid] = to_shape

    lr.logic_nodes[out_node_id] = LogicNode(
        op_id=out_node_id,
        operation=None,
        op_repr=None,
        logic_impl=PureLogic(),
        input_tids=list(output_tids),
        output_tids=[],
        input_shapes=input_shapes,
        output_shapes={},
    )
    return out_node_id



class LRGraphAPI:
    def __init__(self, lr_graph: LRGraph):
        self.lr_graph = lr_graph

    def print(self) -> None:
        print("\nLogic Nodes:")
        for node_id, n in self.lr_graph.logic_nodes.items():
            op_name = None
            if n.operation is not None:
                op_name = f"{n.operation.__class__.__name__}({getattr(n.operation, 'name', '')})"
            else:
                op_name = "PureOutput"
            print(
                f"Node {node_id}: {op_name} | "
                f"in={len(n.input_tids)} out={len(n.output_tids)}"
            )

        print("\nRouting Edges:")
        for tid, e in self.lr_graph.routing_edges.items():
            tname = getattr(e.tensor, "name", str(tid)) if e.tensor is not None else str(tid)
            print(
                f"Edge {tname}: "
                f"from={e.from_node} -> to={sorted(e.to_nodes)} "
                f"| from_shape={e.from_compute_shape} "
                f"| to_shapes={{{', '.join(f'{k}:{v}' for k,v in e.to_compute_shapes.items())}}}"
            )
        print("")


def _short_tid(tid: int, n: int = 6) -> str:
    s = str(tid)
    return s[-n:] if len(s) > n else s

def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')

def lr_to_dot(lr: LRGraph) -> str:
    """
    Graphviz DOT with:
      - Green  = InputLayer
      - Blue   = PureOutput
      - Yellow = All other logic nodes
      - Structured table layout
    """

    def esc(s: str) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    lines = [
        "digraph LR {",
        "rankdir=LR;",
        'graph [fontsize=10];',
        'node  [shape=plain];',
        'edge  [fontsize=9];',
    ]

    # -------- Nodes --------
    for node_id, n in lr.logic_nodes.items():

        # ----- Determine node type & colour -----
        if n.operation is None:
            # Pure output node
            node_color = "#cfe2ff"  # light blue
            op_type = "PureOutput"
            op_name = ""
        elif isinstance(n.operation, keras.layers.InputLayer):
            node_color = "#d9ead3"  # light green
            op_type = "InputLayer"
            op_name = getattr(n.operation, "name", "")
        else:
            node_color = "#fff2cc"  # light yellow
            op_type = n.operation.__class__.__name__
            op_name = getattr(n.operation, "name", "")

        rows = []

        # Header row
        rows.append(
            f'<TR><TD BGCOLOR="{node_color}"><B>op_id: {node_id}</B></TD></TR>'
        )
        rows.append(
            f'<TR><TD ALIGN="left">op: {esc(op_type)}</TD></TR>'
        )
        if op_name:
            rows.append(
                f'<TR><TD ALIGN="left">name: {esc(op_name)}</TD></TR>'
            )

        # Inputs section
        rows.append(
            '<TR><TD ALIGN="left" BGCOLOR="#eeeeee"><B>Inputs</B></TD></TR>'
        )
        if n.input_tids:
            for tid in n.input_tids:
                shp = n.input_shapes.get(tid)
                rows.append(
                    f'<TR><TD ALIGN="left">tid:{str(tid)[-6:]} '
                    f'shape:{esc(shp)}</TD></TR>'
                )
        else:
            rows.append('<TR><TD ALIGN="left">(none)</TD></TR>')

        # Outputs section
        rows.append(
            '<TR><TD ALIGN="left" BGCOLOR="#eeeeee"><B>Outputs</B></TD></TR>'
        )
        if n.output_tids:
            for tid in n.output_tids:
                shp = n.output_shapes.get(tid)
                rows.append(
                    f'<TR><TD ALIGN="left">tid:{str(tid)[-6:]} '
                    f'shape:{esc(shp)}</TD></TR>'
                )
        else:
            rows.append('<TR><TD ALIGN="left">(none)</TD></TR>')

        table = (
            '<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">'
            + "".join(rows) +
            "</TABLE>>"
        )

        lines.append(f'node_{node_id} [label={table}];')

    # -------- Edges --------
    for tid, e in lr.routing_edges.items():
        if e.from_node is None:
            continue

        tname = getattr(e.tensor, "name", None)
        tname = tname if tname else f"tensor_{str(tid)[-6:]}"

        for to in sorted(e.to_nodes):
            to_shape = e.to_compute_shapes.get(to)
            lbl = (
                f"{esc(tname)}\\n"
                f"{esc(e.from_compute_shape)} → {esc(to_shape)}"
            )
            lines.append(
                f'node_{e.from_node} -> node_{to} [label="{lbl}"];'
            )

    lines.append("}")
    return "\n".join(lines)
