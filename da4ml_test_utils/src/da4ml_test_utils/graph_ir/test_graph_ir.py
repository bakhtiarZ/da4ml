import keras
import tensorflow as tf

from da4ml.graph_ir.op_graph import *
from da4ml.graph_ir.schedules.scheduling import *
from da4ml.graph_ir.schedules.scheduling import _SCHEDULE_REGISTRY


def test_build_graph_from_model():
    i = keras.Input((3,2))
    d0 = keras.layers.Dense(1)
    out = d0(i)
    m = keras.Model(i, out)
    g = build_graph_from_model(m)
    exp_ops = {
        0: OpNode(
            op=OpRepr(
                operation=m._nodes_by_depth[1][0].operation,
                args=m._nodes_by_depth[1][0].arguments.args,                
                kwargs=m._nodes_by_depth[1][0].arguments.kwargs,                
                produces=m._nodes_by_depth[1][0].outputs,
                requires=m._nodes_by_depth[1][0].arguments.keras_tensors,
            ),
            data_schedule=None,
            in_tids=[id(t) for t in m._nodes_by_depth[1][0].arguments.keras_tensors],
            out_tids=[id(t) for t in m._nodes_by_depth[1][0].outputs],
        ),
        1: OpNode(
            op=OpRepr(
                operation=m._nodes_by_depth[0][0].operation,
                args=m._nodes_by_depth[0][0].arguments.args,                
                kwargs=m._nodes_by_depth[0][0].arguments.kwargs,                
                produces=m._nodes_by_depth[0][0].outputs,
                requires=m._nodes_by_depth[0][0].arguments.keras_tensors,
            ),
            data_schedule=_SCHEDULE_REGISTRY[keras.layers.Dense],
            in_tids=[id(t) for t in m._nodes_by_depth[0][0].arguments.keras_tensors],
            out_tids=[id(t) for t in m._nodes_by_depth[0][0].outputs],
        )
    }
    assert g.ops == exp_ops
    exp_tensors = {
        id(m._nodes_by_depth[0][0].arguments.keras_tensors[0]): TensorEdge(producer=0, consumers={1}, tensor=m._nodes_by_depth[0][0].arguments.keras_tensors[0]),
        id(m._nodes_by_depth[0][0].outputs[0]): TensorEdge(producer=1, consumers=set(), tensor=m._nodes_by_depth[0][0].outputs[0]),
    }
    assert g.tensors == exp_tensors

def test_graphir_equivalence_to_model():
    pass