import pytest
import keras
import tensorflow as tf

from hgq.layers import QDense
from hgq.config import QuantizerConfig

from da4ml.graph_ir.op_graph import *
from da4ml.graph_ir.scheduling import *
from da4ml.graph_ir.lr_graph import *



# @pytest.fixture
def simple_opgraph():
    i = keras.Input((3,2))
    # d0 = keras.layers.Dense(1) # da4ml doesn't support regular dense, only einsum and the hgq versions of dense
    # d0 = keras.layers.EinsumDense('...f,fc->...c', output_shape=(1,), kernel_initializer='ones', bias_initializer='zeros')
    d0 = QDense(1, iq_conf=QuantizerConfig(heterogeneous_axis=()), kernel_initializer='ones', bias_initializer='zeros')
    # d0 = QDense(1, kernel_initializer='ones', bias_initializer='zeros')
    out = d0(i)
    m = keras.Model(i, out)
    g = build_graph_from_model(m)
    return g


def test_build_lrgraph_from_opgraph(simple_opgraph):
    lr_g = build_lr_graph_from_opgraph(simple_opgraph)

    # finish these tests, build a working graph, see traversal, visualise it
    lr_g_api = LRGraphAPI(lr_g)
    lr_g_api.print()
    print(lr_g)
    assert len(lr_g.logic_nodes.items()) == 2
    assert len(lr_g.routing_edges.items()) == 2

test_build_lrgraph_from_opgraph(simple_opgraph())  