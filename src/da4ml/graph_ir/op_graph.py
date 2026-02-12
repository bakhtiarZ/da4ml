from dataclasses import dataclass, field
from typing import Any, Callable, Type
import keras

from .scheduling import DataSchedule, DataScheduler, _SCHEDULE_REGISTRY

@dataclass
class OpRepr:
    operation: keras.Operation
    args: list
    kwargs: dict
    produces: tuple[keras.KerasTensor, ...]
    requires: tuple[keras.KerasTensor, ...]


@dataclass
class OpNode:
    op: OpRepr 
    data_schedule: DataSchedule | Any
    in_tids: list[int] # the input tensors
    out_tids: list[int] # the output tensors

@dataclass
class TensorEdge:
    producer: int | None = None
    consumers: set[int] = field(default_factory=set)
    tensor: Any | None = None 

@dataclass
class OpGraph:
    ops: dict[int, OpNode]
    tensors: dict[int, TensorEdge]
    model_input_tids: list[int] = field(default_factory=list)
    model_output_tids: list[int] = field(default_factory=list)

def parse_model(model: keras.Model):
    if isinstance(model, keras.Sequential):
        model = model._functional
    operators: dict[int, list[OpRepr]] = {}
    for depth, nodes in model._nodes_by_depth.items():
        _oprs = []
        for node in nodes:
            assert isinstance(node.operation, keras.Operation)
            opr = OpRepr(
                operation=node.operation,
                args=node.arguments.args,
                kwargs=node.arguments.kwargs,
                produces=node.outputs,
                requires=node.arguments.keras_tensors,
            )
            _oprs.append(opr)
        operators[depth] = _oprs
    return [operators[i] for i in range(max(operators.keys()), -1, -1)]

def build_graph_from_model(model : keras.Model) -> OpGraph:
    if isinstance(model, keras.Sequential):
        model = model._functional

    g = build_graph(parsed_model=parse_model(model))

    # True graph ports from the model object (not inferred)
    g.model_input_tids = [id(t) for t in model.inputs]
    g.model_output_tids = [id(t) for t in model.outputs]

    return g

def build_graph(parsed_model: list[list[Any]]) -> OpGraph:
    """
    Build a bipartite-like graph:
      - ops are nodes (node_id is an int we assign)
      - tensors are edges keyed by id(tensor)

    parsed_model is your parse_model(model) output: list[list[OpObj]].
    """
    g = OpGraph(ops={}, tensors={})

    node_id = 0
    for group in parsed_model:
        for opr in group:
            in_tids = [id(t) for t in (opr.requires or ())]
            out_tids = [id(t) for t in (opr.produces or ())]

            data_schedule: DataSchedule | None = _SCHEDULE_REGISTRY.get(type(opr.operation))
            g.ops[node_id] = OpNode(op=opr, data_schedule=data_schedule, in_tids=in_tids, out_tids=out_tids)

            # Register produced tensors: set producer
            for t in (opr.produces or ()):
                tid = id(t)
                edge = g.tensors.get(tid)
                if edge is None:
                    edge = TensorEdge(tensor=t)
                    g.tensors[tid] = edge
                # If you ever hit this, your trace has two producers for same tensor id (shouldn't happen)
                if edge.producer is not None and edge.producer != node_id:
                    raise ValueError(f"Tensor {getattr(t, 'name', tid)} has multiple producers: "
                                     f"{edge.producer} and {node_id}")
                edge.producer = node_id

            # Register required tensors: add consumer
            for t in (opr.requires or ()):
                tid = id(t)
                edge = g.tensors.get(tid)
                if edge is None:
                    edge = TensorEdge(tensor=t)
                    g.tensors[tid] = edge
                edge.consumers.add(node_id)

            node_id += 1

    return g

def input_tensors(g: OpGraph) -> list[int]:
    if g.model_input_tids:
        return list(g.model_input_tids)
    return [tid for tid, e in g.tensors.items() if e.producer is None and e.consumers]

def output_tensors(g: OpGraph) -> list[int]:
    if g.model_output_tids:
        return list(g.model_output_tids)
    return [tid for tid, e in g.tensors.items() if e.producer is not None and not e.consumers]


class OpGraphAPI:
    def __init__(self, g: OpGraph):
        self.g = g

    # --- queries ---
    def successors(self, op_id: int) -> set[int]:
        out = set()
        for tid in self.g.ops[op_id].out_tids:
            out |= self.g.tensors[tid].consumers
        return out

    def predecessors(self, op_id: int) -> set[int]:
        ins = set()
        for tid in self.g.ops[op_id].in_tids:
            p = self.g.tensors[tid].producer
            if p is not None:
                ins.add(p)
        return ins

    # --- viz ---
    def summary(self) -> None:
        for op_id, node in self.g.ops.items():
            op = node.op.operation
            print(f"[{op_id}] {op.__class__.__name__}({getattr(op,'name',None)}) "
                  f"in={len(node.in_tids)} out={len(node.out_tids)}")

    def to_dot(self) -> str:
        # ops as boxes, tensors as arrows
        lines = ["digraph G {", "rankdir=LR;"]
        for op_id, node in self.g.ops.items():
            op = node.op.operation
            label = f"{op.__class__.__name__}\\n{getattr(op,'name','')}"
            lines.append(f'op_{op_id} [shape=box,label="{label}"];')
        for tid, e in self.g.tensors.items():
            if e.producer is None:
                continue
            for c in e.consumers:
                tname = getattr(e.tensor, "name", str(tid))
                lines.append(f'op_{e.producer} -> op_{c} [label="{tname}"];')
        lines.append("}")
        return "\n".join(lines)
    


# def replace_dense_with_minimal(dense_node):
#     def getLeadingDims(node):
#         shape_of_input = node.requires.shape
#         leading_shape = shape_of_input[:-1] # all dims except last
#         return leading_shape
    
#     def create_minNode_from_node(node):
#         leading_dims = getLeading(node)
#         min_dim = node.requires.shape[-1] # n, feature vector length

#         def construct_min_node_from_trace_and_fx_var(min_dim):
#             fxinp = FixedVariableArrayInput(1, min_dim) # replace 1 with desired parallelism
#             inp, out = trace_model(dense_node.model, verbose=True, inputs=fxinp) # trace model with forced shape as min_dim, hopefully node can act as a model.
#             comb_logic = comb_trace(inp, out)
#             return CombLogicNode(comb_logic)

#         return construct_min_node_from_trace_and_fx_var(min_dim)
    
#     minNode = create_minNode_from_node(dense_node)
#     routingNodeInp, routingNodeOut = create_routing_for_min_node(minNode, dense_node) # creates and connects routing

#     dense_node._previous._next = routingNodeInp
#     dense_node._next._previous = routingNodeOut
    
#     #software verify the routing and dense.

