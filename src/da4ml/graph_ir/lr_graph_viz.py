# lr_graph_viz.py
from __future__ import annotations

from typing import Optional, Any, Tuple
from graphviz import Digraph

# Adjust this import path to wherever your LRGraph types live
from .lr_graph import LRGraph, LogicNode, PureLogic, RoutingEdge


def _shape_str(s: Optional[Tuple[Any, ...]]) -> str:
    if s is None:
        return "?"
    # allow ints / None (and keep other objects readable)
    def one(x: Any) -> str:
        if x is None:
            return "?"
        try:
            return str(int(x))
        except Exception:
            return str(x)
    return "(" + ", ".join(one(x) for x in s) + ")"


def _type_name(x: Any) -> str:
    try:
        return type(x).__name__
    except Exception:
        return str(x)


def _op_name(ln: LogicNode) -> str:
    """
    Try to show the underlying Keras op/layer name if available via op_ref.
    Falls back to Logic impl type.
    """
    op_ref = getattr(ln, "op_ref", None)
    if op_ref is not None:
        op_obj = getattr(op_ref, "op", None)
        operation = getattr(op_obj, "operation", None)
        if operation is not None:
            return _type_name(operation)
    return ("OutputLayer" if isinstance(ln.logic_impl, PureLogic) else _type_name(ln.logic_impl))


def _logic_label(node_id: int, ln: LogicNode, compact: bool) -> str:
    logic_name = _type_name(getattr(ln, "logic_impl", None))
    op_display = _op_name(ln)

    title = f"op {node_id}: {op_display}"

    if compact:
        return title

    in_lines = []
    for tid in ln.input_routing:
        in_lines.append(f"in  t{tid}: {_shape_str(ln.input_shapes.get(tid))}")

    out_lines = []
    for tid in ln.output_routing:
        out_lines.append(f"out t{tid}: {_shape_str(ln.output_shapes.get(tid))}")

    parts = [
        title,
        f"logic: {logic_name}",
        "---- inputs ----",
        *(in_lines if in_lines else ["(none)"]),
        "---- outputs ----",
        *(out_lines if out_lines else ["(none)"]),
    ]

    return "\\l".join(parts) + "\\l"



def _routing_edge_label(tid: int, e: RoutingEdge, compact: bool) -> str:
    if compact:
        return f"t{tid}"

    from_s = _shape_str(getattr(e, "from_node_shape", None))
    to_s = _shape_str(getattr(e, "to_node_shape", None))

    rl = getattr(e, "routing_logic", None)
    rl_name = "None" if rl is None else _type_name(rl)

    lines = [
        f"t{tid}",
        f"{from_s} -> {to_s}",
        f"routing={rl_name}",
    ]

    if rl is not None:
        buf_type = getattr(rl, "buffer_type", None)
        buf_shape = getattr(rl, "buffer_shape", None)
        if buf_type is not None or buf_shape is not None:
            lines.append(f"buffer={buf_type}:{buf_shape}")

        sched = getattr(rl, "schedule", None)
        if sched is not None:
            fs = _shape_str(getattr(sched, "from_shape", None))
            ts = _shape_str(getattr(sched, "to_shape", None))
            lines.append(f"sched={fs}->{ts}")

    return "\\n".join(lines)


def visualize_lr_graph(
    lr_g: LRGraph,
    *,
    filename: Optional[str] = None,
    format: str = "svg",
    view: bool = False,
    engine: str = "dot",
    graph_name: str = "LRGraph",
    rankdir: str = "LR",
    compact: bool = False,
    show_external_ios: bool = True,
) -> str:
    """
    Visualize LRGraph with Graphviz.

    - Logic nodes (ops) are boxes.
    - Routing edges are arrows labeled with tensor id and from/to shapes.
    - If show_external_ios=True, any edge with from_node=None or to_node=None
      is connected to a dashed oval INPUT/OUTPUT placeholder.

    If filename is provided, renders to filename.<format> (directories must exist).
    Returns DOT source string.
    """
    dot = Digraph(name=graph_name, engine=engine)
    dot.attr(rankdir=rankdir, fontsize="10")

    dot.attr("node", shape="box", fontname="Menlo", fontsize="10")
    dot.attr("edge", fontname="Menlo", fontsize="9")

    ext_in_nodes: dict[int, str] = {}
    ext_out_nodes: dict[int, str] = {}

    def ext_in_id(tid: int) -> str:
        nid = ext_in_nodes.get(tid)
        if nid is None:
            nid = f"ext_in_t{tid}"
            ext_in_nodes[tid] = nid
            dot.node(nid, label=f"INPUT\\nt{tid}", shape="oval", style="dashed")
        return nid

    def ext_out_id(tid: int) -> str:
        nid = ext_out_nodes.get(tid)
        if nid is None:
            nid = f"ext_out_t{tid}"
            ext_out_nodes[tid] = nid
            dot.node(nid, label=f"OUTPUT\\nt{tid}", shape="oval", style="dashed")
        return nid

    # 1) Add logic nodes
    for node_id, ln in lr_g.logic_nodes.items():
        label = _logic_label(node_id, ln, compact=compact)
        dot.node(f"op_{node_id}", label=label)

    # 2) Add routing edges
    for tid, e in lr_g.routing_edges.items():
        fn = getattr(e, "from_node", None)
        tn = getattr(e, "to_node", None)

        if show_external_ios:
            src = f"op_{fn}" if fn is not None else ext_in_id(tid)
            dst = f"op_{tn}" if tn is not None else ext_out_id(tid)
        else:
            # Skip truly external edges (no src or no dst) if IOS disabled
            if fn is None or tn is None:
                continue
            src = f"op_{fn}"
            dst = f"op_{tn}"

        elabel = _routing_edge_label(tid, e, compact=compact)
        dot.edge(src, dst, label=elabel)

    if filename is not None:
        dot.render(filename=filename, format=format, cleanup=True, view=view)

    return dot.source
