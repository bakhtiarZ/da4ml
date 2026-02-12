from dataclasses import dataclass, field
from typing import Any, Callable, Type
import keras

from da4ml.cmvm.types import CombLogic
from da4ml.trace import FixedVariableArrayInput, comb_trace
from da4ml.converter import trace_model

from .op_graph import OpGraph, OpNode, TensorEdge, output_tensors

@dataclass
class PureLogic:
    empty_logic: bool = True # shouldn't be assigned, just a type.

@dataclass
class RoutingSched:
    empty_for_now=True
    from_shape: tuple[int] | None = None
    to_shape: tuple[int] | None = None

@dataclass
class RoutingLogic:
    to_hardware: Callable[[Any], Any]
    schedule: RoutingSched | None = None
    buffer_type: str = "default" 
    buffer_shape: int = 0    

@dataclass
class RoutingEdge:
    routing_logic: RoutingLogic | None = None
    from_node: int | None = None
    from_node_shape: tuple[int] | None = None
    to_node: int | None= None
    to_node_shape: tuple[int] | None = None


@dataclass
class LogicNode:
    logic_impl: CombLogic | PureLogic
    op_ref: OpNode | None
    input_routing: list[int]
    output_routing: list[int]
    input_shapes: dict[int, tuple[int]] = field(default_factory=dict)
    output_shapes: dict[int, tuple[int]] = field(default_factory=dict)

@dataclass
class LRGraph:
    logic_nodes: dict[int, LogicNode]
    routing_edges: dict[int, RoutingEdge]

def routing_logic_from_schedule(opNode: OpNode):
    # placeholder for now, but this is where we would determine the routing logic based on the schedule of the opNode
    input_routing = {t_in: RoutingLogic(lambda x: x) for t_in in opNode.in_tids} # identity routing for now
    output_routing = {t_out: RoutingLogic(lambda x: x) for t_out in opNode.out_tids} # identity routing for now
    return input_routing, output_routing

def create_comb_logic_from_op(opNode: OpNode):
    if opNode.op.operation.__class__ is keras.layers.InputLayer:
        return PureLogic(), opNode.op.produces[0].shape[1:], opNode.op.produces[0].shape[1:] # for input layer, the logic is pure and the output shape is the same as the input shape, remove batch dimension which is None

    current_input_shape: tuple[int] = opNode.op.requires[0].shape 
    min_inp_shape = (1,) + opNode.data_schedule.minimum_input_shape(current_input_shape) if opNode.data_schedule else current_input_shape
    min_out_shape : tuple[int] = (1,) + opNode.data_schedule.minimum_output_shape(opNode.op.produces[0].shape) if opNode.data_schedule else opNode.op.produces[0].shape
    min_inp = keras.Input(shape=min_inp_shape)
    out = opNode.op.operation(min_inp)
    min_model = keras.Model(min_inp, out)
    fx_inp = FixedVariableArrayInput(shape=(min_inp_shape))
    tr_in, tr_out = trace_model(min_model, inputs=fx_inp)
    comb_logic = comb_trace(tr_in, tr_out)
    assert out.shape[1:] == min_out_shape, f"Output shape {out.shape[1:]} does not match minimum output shape {min_out_shape}"
    return comb_logic, min_inp_shape, min_out_shape


def append_pure_output_node(lr_g: LRGraph, g: OpGraph, output_tids: list[int]) -> int:
    if not output_tids:
        raise ValueError("append_pure_output_node: output_tids was empty")

    out_node_id = (max(lr_g.logic_nodes.keys()) + 1) if lr_g.logic_nodes else 0
    input_shapes: dict[int, tuple[int]] = {}
    for tid in output_tids:
        r_edge = lr_g.routing_edges.get(tid)
        if r_edge is None or r_edge.from_node is None or r_edge.from_node_shape is None:
            raise ValueError(
                f"Output tensor edge {tid} has no producer/shape in LRGraph. "
                f"Have you built LRGraph before appending output?"
            )
        # Connect tensor edge -> output node
        t_shape = g.tensors.get(tid).tensor.shape[1:] if g.tensors.get(tid) and g.tensors.get(tid).tensor is not None else r_edge.from_node_shape
        r_edge.to_node = out_node_id
        r_edge.to_node_shape = t_shape
        input_shapes[tid] = t_shape

    lr_g.logic_nodes[out_node_id] = LogicNode(
        logic_impl=PureLogic(),
        op_ref=None,
        input_routing=list(output_tids),
        output_routing=[],
        input_shapes=input_shapes,
        output_shapes={},
    )
    return out_node_id

def build_lr_graph_from_opgraph(g: OpGraph):
    lr_g: LRGraph = LRGraph({}, {})

    for op_id, opNode in g.ops.items():
        # assume this is in order
        # input_routing, output_routing = routing_logic_from_schedule(opNode)

        comb_logic, min_input_shape, min_output_shape = create_comb_logic_from_op(opNode)
        
        assert lr_g.logic_nodes.get(op_id) is None, f"Op id already exists... {op_id}, op: {opNode}"
        lr_g.logic_nodes[op_id] = LogicNode(
                                    comb_logic, 
                                    opNode, 
                                    opNode.in_tids, 
                                    opNode.out_tids, 
                                    input_shapes={t_in: min_input_shape for t_in in opNode.in_tids}, 
                                    output_shapes={t_out: min_output_shape for t_out in opNode.out_tids}
                                )

        for t_out in opNode.out_tids:
            r_edge = lr_g.routing_edges.get(t_out)
            if r_edge is None:
                # routing_logic = output_routing[t_out] # when looking at output nodes, we can only construct a base routing edge since we do not know what the to_node accepts as shape
                # routing_logic = RoutingLogic()
                r_edge = RoutingEdge()
                lr_g.routing_edges[t_out] = r_edge
            r_edge.from_node_shape = lr_g.logic_nodes[op_id].output_shapes[t_out] # remove batch dimension for routing edge shape
            r_edge.from_node = op_id

        # check requires and produces for input and output node
        for t_in in opNode.in_tids:
            #should only be one now, but need a routing for each
            r_edge = lr_g.routing_edges.get(t_in)
            if r_edge is None:
                # routing_logic = input_routing[t_in] # if routing is different for all in_t's then this is a dictionary lookup
                # routing_logic = RoutingLogic() 
                r_edge = RoutingEdge()
                lr_g.routing_edges[t_in] = r_edge
            r_edge.to_node_shape = lr_g.logic_nodes[op_id].input_shapes[t_in]
            r_edge.to_node = op_id
    
    out_tids = g.model_output_tids if getattr(g, "model_output_tids", None) else output_tensors(g)
    append_pure_output_node(lr_g, g, output_tids=out_tids)
    return lr_g

class LRGraphAPI:
    def __init__(self, lr_graph: LRGraph):
        self.lr_graph = lr_graph
    
    def print(self):
        print("\nLogic Nodes:")
        for node_id, logic_node in self.lr_graph.logic_nodes.items():
            print(f"Node ID: {node_id}, Logic Impl: {logic_node.logic_impl}, Input Routing: {logic_node.input_routing}, Output Routing: {logic_node.output_routing}")
        print("\nRouting Edges:")
        for edge_id, routing_edge in self.lr_graph.routing_edges.items():
            print(f"Edge ID: {edge_id}, Routing Impl: {routing_edge.routing_logic}, From Node: {routing_edge.from_node}, To Node: {routing_edge.to_node}")
        print("\n")





