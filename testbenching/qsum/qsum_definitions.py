from hgq.layers import QDense, QAdd, QSum
from hgq.config import QuantizerConfig, QuantizerConfigScope

from da4ml.graph_ir.lr_graph import build_lr_graph_from_model

import keras


def config():
    return {"PARALLELISM": 2,
            "Q_CONF": QuantizerConfig(heterogeneous_axis=(), k0=1, i0=4, f0=3),
            }

def simple_qsum():
    i = keras.Input((3, 1))
    s = QSum(iq_conf=QuantizerConfig(heterogeneous_axis=()), axes=1, scale=1, keepdims=True)(i)
    m = keras.Model(i, s)
    return m

def m_with_qsum():
    i = keras.Input((4,2))
    d0 = QDense(1, iq_conf=QuantizerConfig(heterogeneous_axis=()), kernel_initializer='ones', bias_initializer='zeros', enable_iq=True, enable_oq=True)(i) # 4, 1
    s = QSum(iq_conf=QuantizerConfig(heterogeneous_axis=()), axes=1, scale=1, keepdims=True, enable_iq=True)(d0) # 1, 1
    m = keras.Model(i, s)
    return m

def m_testing_parallelism():
    i = keras.Input((4,2))
    d0 = QDense(2, iq_conf=QuantizerConfig(heterogeneous_axis=()), kernel_initializer='ones', bias_initializer='zeros', enable_iq=True, enable_oq=True)(i) # 4, 2
    d1 = QDense(1, iq_conf=QuantizerConfig(heterogeneous_axis=()), kernel_initializer='ones', bias_initializer='zeros', enable_iq=True)(d0) # 4, 2
    m = keras.Model(i, d1)
    return m

def m_with_qsum_fixed_q_conf():
    cf = config()
    q = cf["Q_CONF"]
    i = keras.Input((4,2))
    d0 = QDense(1, iq_conf=q, oq_conf=q, kernel_initializer='ones', bias_initializer='zeros', enable_iq=True, enable_oq=True)(i) # 4, 1
    s = QSum(iq_conf=q, axes=1, scale=1, keepdims=True, enable_iq=True)(d0) # 1, 1
    m = keras.Model(i, s)
    return m


def qsum_to_dense():
    cf = config()
    with QuantizerConfigScope(place='datalane', heterogeneous_axis=(-1,)):
        i = keras.Input((4,2))
        s = QSum(axes=1, scale=1, keepdims=True, enable_iq=True)(i) # 1, 1
        d0 = QDense(1, kernel_initializer='ones', bias_initializer='zeros')(s) #
        m = keras.Model(i, d0)
        return m

def generate_qsum_hw(lrg, dir):
    logic_impl = lrg.logic_nodes[1].logic_impl
    print("\n\n\n")  
    print(logic_impl) 
    project_dir = dir
    module_file, instance_decl, input_port_conns, output_port_conns = logic_impl.generate_hw(project_dir=project_dir, node=lrg.logic_nodes[1])
    # print(f"\n\n {module_file} {instance_decl} {input_port_conns} {output_port_conns}")
