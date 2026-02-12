import pytest
import keras
import tensorflow as tf
from graphviz import Source

from hgq.layers import QDense
from hgq.config import QuantizerConfig

from da4ml.graph_ir.op_graph import *
from da4ml.graph_ir.scheduling import *
# from da4ml.graph_ir.lr_graph_orig import *
# from da4ml.graph_ir.lr_graph_viz_orig import visualize_lr_graph

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

# @pytest.fixture
def two_layer_opgraph():
    i = keras.Input((3,2))
    d0 = QDense(3, iq_conf=QuantizerConfig(heterogeneous_axis=()), kernel_initializer='ones', bias_initializer='zeros')
    d1 = QDense(1, iq_conf=QuantizerConfig(heterogeneous_axis=()), kernel_initializer='ones', bias_initializer='zeros')
    out = d1(d0(i))
    m = keras.Model(i, out)
    g = build_graph_from_model(m)
    return g

def two_layer_model():
    i = keras.Input((3,2))
    d0 = QDense(3, iq_conf=QuantizerConfig(heterogeneous_axis=()), kernel_initializer='ones', bias_initializer='zeros')
    d1 = QDense(1, iq_conf=QuantizerConfig(heterogeneous_axis=()), kernel_initializer='ones', bias_initializer='zeros')
    out = d1(d0(i))
    m = keras.Model(i, out)
    return m

def test_build_lrgraph_from_model():
    lr_g = build_lr_graph_from_model(two_layer_model())
    dot_str = lr_to_dot(lr_g)
    src = Source(dot_str)
    src.render("/homes/bm920/workspace/da4ml/.tmp/figures/lr_graphv2", format="svg", view=True)

# test_build_lrgraph_from_opgraph(simple_opgraph())  
# test_build_lrgraph_from_opgraph(two_layer_opgraph())
test_build_lrgraph_from_model()