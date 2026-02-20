import pytest
import keras
import tensorflow as tf
from graphviz import Source

from hgq.layers import QDense
from hgq.config import QuantizerConfig

from da4ml.codegen.rtl.rtl_model import get_io_kifs

from da4ml.graph_ir.op_graph import *
from da4ml.graph_ir.scheduling import *
# from da4ml.graph_ir.lr_graph_orig import *
# from da4ml.graph_ir.lr_graph_viz_orig import visualize_lr_graph

from da4ml.graph_ir.lr_graph import *

from pathlib import Path

def make_next_numbered_dir(base_dir: str | Path, prefix: str = "", suffix: str = "") -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    i = 0
    while True:
        candidate = base / f"{prefix}{i}{suffix}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            i += 1


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


def test_write_rtl_from_lrgraph():
    lr_g = build_lr_graph_from_model(two_layer_model())
    # project_dir = make_next_numbered_dir('/homes/bm920/workspace/da4ml/.tmp/lr_graph_rtl_projects/', prefix='project_')
    project_dir = f"/homes/bm920/workspace/da4ml/.tmp/lr_graph_rtl_projects/project_test"
    rtl_code = lr_graph_to_hardware(lr_g, project_dir, debug=True)
    print(rtl_code)
    print(f"RTL code written to project directory: {project_dir}/top_module.sv")

# test_build_lrgraph_from_opgraph(simple_opgraph())  
# test_build_lrgraph_from_opgraph(two_layer_opgraph())
# test_build_lrgraph_from_model()
# test_write_rtl_from_lrgraph()

test_write_rtl_from_lrgraph()
