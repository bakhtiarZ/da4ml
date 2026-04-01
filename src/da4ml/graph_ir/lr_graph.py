from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Set

import keras
from pathlib import Path

from da4ml.cmvm.types import CombLogic
from da4ml.trace import FixedVariableArrayInput, comb_trace
from da4ml.converter import trace_model
from da4ml.codegen.rtl.rtl_model import RTLModel, get_io_kifs

from .hardware_types import PortConnection, HWInterface, PureLogic, CustomLogic, RoutingLogic
from .schedules.scheduling import DataSchedule, _SCHEDULE_REGISTRY
from .util import _strip_batch_and_ensure_ints, parse_model, OpRepr, _flatten_ops, short_tid, _short_tid, _strip_batch


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

    logic_impl: CombLogic | CustomLogic | PureLogic

    input_tids: List[int]
    output_tids: List[int]

    input_shapes: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    output_shapes: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    logic_wrapper: Optional[CustomLogic] = None


@dataclass
class LRGraph:
    logic_nodes: Dict[int, LogicNode]
    routing_edges: Dict[int, RoutingEdge]
    model_input_tids: List[int] = field(default_factory=list)
    model_output_tids: List[int] = field(default_factory=list)
    parallelism: int = 1

def _min_shapes_for_op(
    opr: OpRepr,
    schedule: Optional[DataSchedule],
    parallelism: int = 1,
    **kwargs: Any,
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
    min_in_no_batch = schedule.minimum_input_shape(in_no_batch_int, **kwargs)
    min_out_no_batch = schedule.minimum_output_shape(out_no_batch_int, **kwargs)
    min_in = tuple(min_in_no_batch)
    min_out = tuple(min_out_no_batch)

    streamed_in_no_batch = (min_in[0] * parallelism, min_in[1:])
    streamed_out_no_batch = (min_out[0] * parallelism, min_out[1:])

    return min_in, min_out


def create_comb_logic_from_oprepr(
    opr: OpRepr,
    schedule: Optional[DataSchedule],
) -> Tuple[CombLogic | PureLogic, Tuple[int, ...], Tuple[int, ...]]:
    
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

def create_logic_impl_from_oprepr(
    opr: OpRepr,
    schedule: Optional[DataSchedule],
    parallelism: int = 1,
) -> CombLogic | PureLogic | CustomLogic:
    
    if opr.operation.__class__ is keras.layers.InputLayer:
        out_no_batch_int = _strip_batch_and_ensure_ints(opr.produces[0].shape)
        shp = out_no_batch_int
        return PureLogic(opr=opr), shp, shp
    
    elif issubclass(schedule.hardware_type, CustomLogic):
        logic_impl = schedule.hardware_type(opr)
        axes_without_batch = tuple(axis - 1 for axis in opr.operation.axes) if opr.operation.axes is not None else None
        min_in_shape, min_out_shape = _min_shapes_for_op(opr, schedule, axes=axes_without_batch)
        return logic_impl, min_in_shape, min_out_shape
    
    elif schedule.hardware_type == CombLogic:
        return create_comb_logic_from_oprepr(opr, schedule)
    
    else:
        raise AssertionError(f"Hardware type is not recognised opr class: {opr.operation.__class__}, schedule hardware type: {schedule.hardware_type}")
        

def build_lr_graph_from_parsed(
    parsed: List[List[OpRepr]],
    *,
    model_inputs: Optional[Sequence[keras.KerasTensor]] = None,
    model_outputs: Optional[Sequence[keras.KerasTensor]] = None,
    parallelism: int = 1,
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
            routing_logic = RoutingLogic(
                buffer_type="default", buffer_shape=(-1, -1), # default routing logic for now; can be customized later
            )
            e = RoutingEdge(tid=tid, tensor=t, routing_logic=routing_logic)
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

        logic_impl, min_in_shape, min_out_shape = create_logic_impl_from_oprepr(opr, schedule, parallelism=parallelism)

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
            
            # if edge has is now fully connected (from_node, to_nodes, from_compute_shape, to_compute_shapes), we can fill the routing logic properly
            if e.from_node is not None and e.from_compute_shape is not None:
                unique_to_shapes = set(e.to_compute_shapes.values())
                if len(unique_to_shapes) > 1:
                    raise ValueError(
                        f"Edge tid={e.tid} has multiple consumer compute shapes {unique_to_shapes}. This is not supported right now. "
                        f"Either enforce single shape or store routing logic per consumer."
                    )
                to_shape = e.to_compute_shapes[op_id]
                if tuple(e.from_compute_shape) == tuple(to_shape):
                    buffer_size = 1
                else:
                    buffer_size = math.prod(e.semantic_shape[:-1])  # leading dims only
                item_size = e.from_compute_shape[-1] # this may not always be -1, need to fix later
                e.routing_logic = RoutingLogic(
                    buffer_type=schedule.buffer_type if op_id != 1 else "input_buffer", 
                    buffer_shape=(buffer_size, item_size),
                )
                    
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


def build_lr_graph_from_model(model: keras.Model, parallelism: int = 1) -> LRGraph:
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
        parallelism=parallelism,
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
        if tuple(e.from_compute_shape) == tuple(to_shape):
            buffer_size = 1
        else:
            buffer_size = math.prod(e.semantic_shape[:-1])  # leading dims only
        item_size = e.from_compute_shape[-1] # this may not always be -1, need to fix later
        e.routing_logic = RoutingLogic(
            buffer_type="output_buffer", # hardcoded for now; can be customized later, should be a memory or something.
            buffer_shape=(buffer_size, item_size),
        )
        input_shapes[tid] = to_shape

    lr.logic_nodes[out_node_id] = LogicNode(
        op_id=out_node_id,
        operation=None,
        op_repr=None,
        logic_impl=PureLogic(opr=None),
        input_tids=list(output_tids),
        output_tids=[],
        input_shapes=input_shapes,
        output_shapes={},
    )
    return out_node_id

def create_logic_node_hw(node_id, node, project_dir):
    if (type(node.logic_impl) == CombLogic):
        return create_comb_logic_node_hw(node_id, node, project_dir)
    elif (issubclass(type(node.logic_impl), CustomLogic)):
        module_file_path, lines, input_port_conns, output_port_conns = node.logic_impl.generate_hw(project_dir=project_dir)
        return lines, input_port_conns, output_port_conns
    elif (type(node.logic_impl) == PureLogic):
        print("PureLogic node, no hardware to generate, what happened here? is this supposed to be called?")
    else:
        raise AssertionError(f"Unknown logic_impl type {type(node.logic_impl)} for node_id {node_id}")

def create_comb_logic_node_hw(node_id, node, project_dir):
    lines = []
    lines.append(f"// Logic node {node_id} for operation {node.operation}")
    hw_interface = HWInterface(node)

    in_bw, in_sz = hw_interface.get_input_bw_is()
    input_port_conns = PortConnection(
        data=(f"inp_to_op_{node_id}", in_bw * in_sz),
        valid=f"in_valid_to_{node_id}",
        ready=f"out_ready_to_{node_id}"
    )
    input_port_decls = input_port_conns.get_intermediate_decls()
    lines.append("\n".join(input_port_decls))

    out_bw, out_sz = hw_interface.get_output_bw_is()
    output_port_conns = PortConnection(
        data=(f"out_from_op_{node_id}", out_bw * out_sz),
        valid=f"out_valid_from_{node_id}",
        ready=f"in_ready_from_{node_id}"
    )

    output_port_decls = output_port_conns.get_intermediate_decls()
    lines.append("\n".join(output_port_decls))

    op_name = node.operation.__class__.__name__ if node.operation else "PureOutput"
    instance_name = f"op_{node_id}__{op_name}"
    rtl_model = RTLModel(
        solution=node.logic_impl,
        prj_name=f"mod_{instance_name}",
        path=project_dir,
        flavor="verilog",
    )
    rtl_model.write()
    
    port_conns = f".model_inp({input_port_conns.data[0]}), .model_out({output_port_conns.data[0]})" # rn its unclocked with no rst
    #since the comb logic has no ready and valid, just assign it to the previous one as passthrough for now
    lines.append(f"assign {output_port_conns.valid} = {input_port_conns.valid}; // passthrough valid")
    lines.append(f"assign {output_port_conns.ready} = {input_port_conns.ready}; // passthrough ready")
    lines.append(f"mod_{instance_name} {instance_name} ({port_conns});")
    lines.append(f"// End of logic node {node_id}")
    return lines, input_port_conns, output_port_conns


def create_buffer(packed_bitwidth, r_edge):
    lines = []
    lines.append(f"// Buffer for edge tid={r_edge.tid} with routing logic {r_edge.routing_logic}")
    # inst_name = f"buffer_{short_tid(r_edge.tid)}"
    inst_name = f"buffer_edge__from_op{r_edge.from_node}_to_op{next(iter(r_edge.to_nodes))}"
    buffer_size, item_size = r_edge.routing_logic.buffer_shape
    input_port_conns = PortConnection(
        data=(f"inp_to_{inst_name}", packed_bitwidth),
        valid=f"in_valid_to_{inst_name}",
        ready=f"out_ready_to_{inst_name}"
    )
    lines.append("\n".join(input_port_conns.get_intermediate_decls()))
    output_port_conns = PortConnection(
        data=(f"out_from_{inst_name}", packed_bitwidth),
        valid=f"out_valid_from_{inst_name}",
        ready=f"in_ready_from_{inst_name}"
    )
    lines.append("\n".join(output_port_conns.get_intermediate_decls()))
    inst_params = f".DEPTH({buffer_size}), .DATA_WIDTH({packed_bitwidth})"
    lines.append(
        f"fifo_rv #({inst_params}) {inst_name} (.clk(clk), .rst(rst), .in_data({input_port_conns.data[0]}), .in_valid({input_port_conns.valid}), .out_ready({input_port_conns.ready}), .out_data({output_port_conns.data[0]}), .out_valid({output_port_conns.valid}), .in_ready({output_port_conns.ready}), /* verilator lint_off PINCONNECTEMPTY */ .full(), .empty(), .count() /* verilator lint_on PINCONNECTEMPTY */);"
    )
    lines.append(f"// End of buffer for edge tid={r_edge.tid}")
    return lines, input_port_conns, output_port_conns


def get_top_level_interface(lrg: LRGraph):
    first_node = lrg.logic_nodes[1] # skip input
    fnhwi = HWInterface(first_node)
    packed_input_bw = math.prod(fnhwi.get_input_bw_is())
    last_node = lrg.logic_nodes[max(lrg.logic_nodes.keys()) - 1] # skip output
    lnhwi = HWInterface(last_node)
    packed_output_bw = math.prod(lnhwi.get_output_bw_is())
    return packed_input_bw, packed_output_bw

def create_preamble(name, lr):
    packed_in_width, packed_out_width = get_top_level_interface(lr)
    module_declaration = f"""module {name} (
        input logic clk, 
        input logic rst, 
        input logic [{packed_in_width-1}:0] data_in, 
        input logic data_in_valid,
        input logic data_out_ready,
        output logic [{packed_out_width-1}:0] data_out,
        output logic data_out_valid,
        output logic data_in_ready 
    );
    """ 
    return module_declaration
    
def assign_inputs_with_previous(input_port_conns_previous: PortConnection, input_port_conns_next: PortConnection, output_port_conns_previous: PortConnection, output_port_conns_next: PortConnection):
    lines = []
    lines.append(f"assign {input_port_conns_previous.ready} = {output_port_conns_next.ready};")
    lines.append(f"assign {input_port_conns_next.data[0]} = {output_port_conns_previous.data[0]};") 
    lines.append(f"assign {input_port_conns_next.valid} = {output_port_conns_previous.valid};")
    return "\n".join(lines)

def configure_custom_logic_nodes(lr: LRGraph):
    for node_id, node in lr.logic_nodes.items():
        if issubclass(type(node.logic_impl), CustomLogic):
            node.logic_impl.configure(node)

def lr_graph_to_hardware(lr: LRGraph, project_dir: str | Path, debug=False) -> int:
    
    # injecting node and making hw configs for custom logics
    configure_custom_logic_nodes(lr)
    
    lines = []
    os.makedirs(project_dir, exist_ok=True)
    #copy the src file for fifo_rv
    
    preamble = create_preamble("top_module", lr)
    lines.append(preamble)
    include_output_buffer = False # temp
    input_of_top = PortConnection(data=(f"data_in", get_top_level_interface(lr)[0]), valid="data_in_valid", ready="data_out_ready")
    output_of_top = PortConnection(data=(f"data_out", get_top_level_interface(lr)[1]), valid="data_out_valid", ready="data_in_ready")
    input_ports_of_previous = output_of_top
    output_ports_of_previous = input_of_top 
    for node_id, node in lr.logic_nodes.items():
        if node_id == 0:
            continue
        if node_id == max(lr.logic_nodes.keys()):
            assignments = assign_inputs_with_previous(input_ports_of_previous, output_of_top, output_ports_of_previous, input_of_top)
            lines.append(assignments) 
            continue
        ln_lines, l_input_port_connections, l_output_port_connections = create_logic_node_hw(node_id, node, project_dir)
        if (type(ln_lines) == str):
            ln_lines = [ln_lines]
        lines.extend(ln_lines)
        assignments = assign_inputs_with_previous(input_ports_of_previous, l_input_port_connections, output_ports_of_previous, l_output_port_connections)
        lines.append(f"\n// Connecting intermediate signals of node {node_id} to previous intermediate signals\n{assignments}\n// End of connections for node {node_id}\n")
        if (not include_output_buffer and node_id == max(lr.logic_nodes.keys()) - 1):
            output_ports_of_previous = l_output_port_connections
            input_ports_of_previous = l_input_port_connections
            continue
        edge_for_buffer = lr.routing_edges[node.output_tids[0]]
        buffer_lines, buffer_input_port_conn, buffer_output_port_conn = create_buffer(l_output_port_connections.data[1], edge_for_buffer)
        lines.extend(buffer_lines)
        assignments = assign_inputs_with_previous(l_input_port_connections, buffer_input_port_conn, l_output_port_connections, buffer_output_port_conn)
        lines.append(f"\n// Connecting buffer for edge tid={edge_for_buffer.tid} to logic node {node_id}\n{assignments}\n// End of connections for buffer for edge tid={edge_for_buffer.tid}\n")
        input_ports_of_previous = buffer_input_port_conn
        output_ports_of_previous = buffer_output_port_conn
    
    lines.append("\nendmodule")
    if debug:
        print("\n".join(lines))
    shutil.copy("/homes/bm920/workspace/da4ml/src/da4ml/codegen/rtl/verilog/source/fifo_rv.sv", f"{project_dir}/src/static/fifo_rv.sv")
    with open(f"{project_dir}/top_module.sv", "w") as f:
        f.write("\n".join(lines))
    return len(lines)

        
def lr_to_dot(lr: LRGraph) -> str:
    """
    Graphviz DOT with:
      - Node colors:
          * InputLayer  = green
          * PureOutput  = blue
          * Other ops   = yellow
      - Structured HTML-table node labels
      - Edge labels include:
          * tensor name + tid(short)
          * compute: from_compute_shape -> to_compute_shape(for that consumer)
          * semantic_shape
          * routing logic: buffer_type + buffer_shape
    """

    def esc(s: Any) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def short_tid(tid: int, n: int = 6) -> str:
        s = str(tid)
        return s[-n:] if len(s) > n else s

    def edge_color(e: RoutingEdge, to: int) -> str:
        # simple heuristic colouring (optional but helpful)
        to_shape = e.to_compute_shapes.get(to)
        if e.from_compute_shape is None or to_shape is None:
            return "gray60"
        if tuple(e.from_compute_shape) == tuple(to_shape):
            return "forestgreen"
        return "darkorange"

    lines = [
        "digraph LR {",
        "rankdir=LR;",
        'graph [fontsize=10];',
        'node  [shape=plain];',
        'edge  [fontsize=9];',
    ]

    # -------- Nodes --------
    for node_id, n in lr.logic_nodes.items():
        if n.operation is None:
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

        rows: List[str] = []
        rows.append(f'<TR><TD BGCOLOR="{node_color}"><B>op_id: {node_id}</B></TD></TR>')
        rows.append(f'<TR><TD ALIGN="left">op: {esc(op_type)}</TD></TR>')
        if op_name:
            rows.append(f'<TR><TD ALIGN="left">name: {esc(op_name)}</TD></TR>')
        if issubclass(type(n.logic_impl), CustomLogic):
            rows.append(f'<TR><TD ALIGN="left">logic: {esc(n.logic_impl)}</TD></TR>')
         
        rows.append('<TR><TD ALIGN="left" BGCOLOR="#eeeeee"><B>Inputs</B></TD></TR>')
        if n.input_tids:
            for tid in n.input_tids:
                shp = n.input_shapes.get(tid)
                rows.append(f'<TR><TD ALIGN="left">tid:{short_tid(tid)}  shape:{esc(shp)}</TD></TR>')
        else:
            rows.append('<TR><TD ALIGN="left">(none)</TD></TR>')

        rows.append('<TR><TD ALIGN="left" BGCOLOR="#eeeeee"><B>Outputs</B></TD></TR>')
        if n.output_tids:
            for tid in n.output_tids:
                shp = n.output_shapes.get(tid)
                rows.append(f'<TR><TD ALIGN="left">tid:{short_tid(tid)}  shape:{esc(shp)}</TD></TR>')
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
        tname = tname if tname else f"tensor_{short_tid(tid)}"

        # routing logic summary (same for all fan-outs in Option A)
        if e.routing_logic is None:
            r_summary = "routing: (none)"
        else:
            bt = getattr(e.routing_logic, "buffer_type", None)
            bs = getattr(e.routing_logic, "buffer_shape", None)
            r_summary = f"routing: {bt}  buf:{bs}"

        sem_summary = f"semantic: {e.semantic_shape}" if e.semantic_shape is not None else "semantic: (none)"

        for to in sorted(e.to_nodes):
            to_shape = e.to_compute_shapes.get(to)

            lbl = (
                f"name: {esc(tname)}  tid:{short_tid(tid)}\\n"
                f"compute: {esc(e.from_compute_shape)} → {esc(to_shape)}\\n"
                f"{esc(sem_summary)}\\n"
                f"{esc(r_summary)}"
            )

            col = edge_color(e, to)
            lines.append(
                f'node_{e.from_node} -> node_{to} '
                f'[label="{lbl}", color="{col}", fontcolor="{col}"];'
            )

    lines.append("}")
    return "\n".join(lines)


### depr
# class LRGraphAPI:
#     def __init__(self, lr_graph: LRGraph):
#         self.lr_graph = lr_graph

#     def print(self) -> None:
#         print("\nLogic Nodes:")
#         for node_id, n in self.lr_graph.logic_nodes.items():
#             op_name = None
#             if n.operation is not None:
#                 op_name = f"{n.operation.__class__.__name__}({getattr(n.operation, 'name', '')})"
#             else:
#                 op_name = "PureOutput"
#             print(
#                 f"Node {node_id}: {op_name} | "
#                 f"in={len(n.input_tids)} out={len(n.output_tids)}"
#             )

#         print("\nRouting Edges:")
#         for tid, e in self.lr_graph.routing_edges.items():
#             tname = getattr(e.tensor, "name", str(tid)) if e.tensor is not None else str(tid)
#             print(
#                 f"Edge {tname}: "
#                 f"from={e.from_node} -> to={sorted(e.to_nodes)} "
#                 f"| from_shape={e.from_compute_shape} "
#                 f"| to_shapes={{{', '.join(f'{k}:{v}' for k,v in e.to_compute_shapes.items())}}}"
#             )
#         print("")