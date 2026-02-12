from dataclasses import dataclass, field
from typing import Any, Callable, Type
import keras

from da4ml.cmvm.types import CombLogic
from da4ml.trace import FixedVariableArrayInput, comb_trace
from da4ml.converter import trace_model

from .op_graph import OpGraph, OpNode, TensorEdge


@dataclass
class PureLogic:
    empty_logic: bool = True # shouldn't be assigned, just a type.

@dataclass
class RoutingLogic:
    to_hardware: Callable[[Any], Any]

@dataclass
class RoutingEdge:
    routing_impl: RoutingLogic | Any
    from_node: int | None = None
    to_node: int | None= None


@dataclass
class LogicNode:
    logic_impl: CombLogic | Any
    input_routing: list[int]
    output_routing: list[int]

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
        return PureLogic() 

    current_input_shape: tuple = opNode.op.requires[0].shape # placeholder, assuming single input for now
    # output_shape = output tensor from opnode shape
    min_inp_shape = opNode.data_schedule.minimum_input_shape(current_input_shape) if opNode.data_schedule else current_input_shape
    min_inp_shape = (1,) + min_inp_shape 
    min_inp = keras.Input(shape=min_inp_shape)
    out = opNode.op.operation(min_inp)
    min_model = keras.Model(min_inp, out)
    fx_inp = FixedVariableArrayInput(shape=(min_inp_shape))
    tr_in, tr_out = trace_model(min_model, inputs=fx_inp)
    comb_logic = comb_trace(tr_in, tr_out)
    return comb_logic


def build_lr_graph_from_opgraph(g: OpGraph):
    lr_g: LRGraph = LRGraph({}, {})

    for op_id, opNode in g.ops.items():
        # assume this is in order
        input_routing, output_routing = routing_logic_from_schedule(opNode)

        op_logic: CombLogic | Any = create_comb_logic_from_op(opNode)

        assert lr_g.logic_nodes.get(op_id) is None, f"Op id already exists... {op_id}, op: {opNode}"
        lr_g.logic_nodes[op_id] = LogicNode(op_logic, opNode.in_tids, opNode.out_tids)

        for t_out in opNode.out_tids:
            r_edge = lr_g.routing_edges.get(t_out)
            if r_edge is None:
                routing_impl = output_routing[t_out]
                r_edge = RoutingEdge(routing_impl)
                lr_g.routing_edges[t_out] = r_edge
            r_edge.from_node = op_id

        # check requires and produces for input and output node
        for t_in in opNode.in_tids:
            #should only be one now, but need a routing for each
            r_edge = lr_g.routing_edges.get(t_in)
            if r_edge is None:
                routing_impl = input_routing[t_in] # if routing is different for all in_t's then this is a dictionary lookup
                r_edge = RoutingEdge(routing_impl)
                lr_g.routing_edges[t_in] = r_edge
            r_edge.to_node = op_id
    
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
            print(f"Edge ID: {edge_id}, Routing Impl: {routing_edge.routing_impl}, From Node: {routing_edge.from_node}, To Node: {routing_edge.to_node}")
        print("\n")





