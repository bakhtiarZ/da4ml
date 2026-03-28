import pytest
import keras
import tensorflow as tf
from graphviz import Source

from hgq.layers import QDense, QAdd, QSum
from hgq.config import QuantizerConfig

from da4ml.codegen.rtl.rtl_model import get_io_kifs

from da4ml.graph_ir.op_graph import *
from da4ml.graph_ir.schedules.scheduling import *
from da4ml.graph_ir.util import *
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

def interpret_as(
    x: np.ndarray,
    k: np.ndarray|int,
    i: np.ndarray|int,
    f: np.ndarray|int,
) -> np.ndarray:
    b = k + i + f
    bias = 2.0 ** (b - 1) * k
    return (np.floor(x + bias) % 2.0**b - bias) * 2.0**-f

def encode_to(
    x: np.ndarray,
    k: np.ndarray|int,
    i: np.ndarray|int,
    f: np.ndarray|int,
) -> np.ndarray:
    return np.floor(x * 2.0**f).astype(int) % 2**(k + i + f)

def encode_to_d(
    x: np.ndarray,
    k: np.ndarray|int,
    i: np.ndarray|int,
    f: np.ndarray|int,
    as_hex:bool=False,
    packed:bool=True,
):
    """Encode the input array `x` to hex string dump either in packed or unpacked format.
    packed: the ones at unwrapped interfaces
    unpacked: the ones at internal interfaces (for direct input from binder, some bits may appear at places where the bits are unused)

    x, k, i, f: input array, keep_negative, integer (e. sign), fractional.
    """
    x_int = encode_to(x, k, i, f)
    if not packed:
        _df = np.max(f) - f
        x_int <<= _df
        k,i,f = np.max(k), np.max(i), np.max(f)
    bb = k + i + f
    if not isinstance(bb, np.ndarray):
        bb = np.full_like(x_int, bb)
    accum = 0
    for v, b in zip(x_int[::-1], bb[::-1]):
        accum<<=int(b)
        accum|=int(v)
    if as_hex:
        return hex(accum)[2:]
    else:
        return accum


def m_with_qsum():
    i = keras.Input((3,2))
    s = QSum(iq_conf=QuantizerConfig(heterogeneous_axis=()), axes=0, scale=1, keepdims=False)(i) # 1, 2
    d0 = QDense(1, iq_conf=QuantizerConfig(heterogeneous_axis=()), kernel_initializer='ones', bias_initializer='zeros')(s) # 1, 1
    m = keras.Model(i, d0)
    return m

def simple_qsum():
    i = keras.Input((3,1))
    s = QSum(iq_conf=QuantizerConfig(heterogeneous_axis=()), axes=1, scale=1, keepdims=True)(i)
    print(f"DEBUG simple_qsum s.shape: {s.shape}")
    m = keras.Model(i, s)
    return m


def test_build_lrgraph_from_model(model, figurepath):
    lr_g = build_lr_graph_from_model(model)
    dot_str = lr_to_dot(lr_g)
    src = Source(dot_str)
    src.render(f"{figurepath}", format="svg", view=True)
    return lr_g

def test_write_rtl_from_lrgraph(model):
    lr_g = build_lr_graph_from_model(model)
    # project_dir = make_next_numbered_dir('/homes/bm920/workspace/da4ml/.tmp/lr_graph_rtl_projects/', prefix='project_')
    name = model.name if model.name else "model"
    project_dir = f"/homes/bm920/workspace/da4ml/.tmp/lr_graph_rtl_projects/project_test/{name}"
    rtl_code = lr_graph_to_hardware(lr_g, project_dir, debug=True)
    print(rtl_code)
    print(f"RTL code written to project directory: {project_dir}/top_module.sv")


def test_qsum_gen(model = simple_qsum()):
    lrg = test_build_lrgraph_from_model(model, f"/homes/bm920/workspace/da4ml/.tmp/figures/qsum_lrg")
    logic_impl = lrg.logic_nodes[1].logic_impl
    print("\n\n\n")  
    print(logic_impl) 
    project_dir = f"/homes/bm920/workspace/da4ml/.tmp/qsum_testing_dir"
    module_file, instance_decl, input_port_conns, output_port_conns = logic_impl.generate_hw(project_dir=project_dir, node=lrg.logic_nodes[1])
    print(f"\n\n {module_file} {instance_decl} {input_port_conns} {output_port_conns}")

# test_qsum_gen()
a = QSum(iq_conf=QuantizerConfig(heterogeneous_axis=()), axes=1, scale=1, keepdims=True)
print(a)
print(a.iq.config.config)
k = a.iq.config.config['k0']
i = a.iq.config.config['i0']
f = a.iq.config.config['f0']
print(f"k: {k}, i: {i}, f: {f}")

print(a.oq.config.config)
