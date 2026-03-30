from hgq.layers import QDense, QAdd, QSum
from hgq.config import QuantizerConfig

from da4ml.graph_ir.lr_graph import build_lr_graph_from_model

import keras


def config():
    return {}

def simple_qsum():
    i = keras.Input((3, 1))
    s = QSum(iq_conf=QuantizerConfig(heterogeneous_axis=()), axes=1, scale=1, keepdims=True)(i)
    m = keras.Model(i, s)
    return m

def m_with_qsum():
    i = keras.Input((3,1))
    d0 = QDense(1, iq_conf=QuantizerConfig(heterogeneous_axis=()), kernel_initializer='ones', bias_initializer='zeros')(i) # 3, 1
    s = QSum(iq_conf=QuantizerConfig(heterogeneous_axis=()), axes=1, scale=1, keepdims=True)(d0) # 1, 1
    m = keras.Model(i, s)
    return m

def qsum_lrg(model = simple_qsum()):
    lrg = build_lr_graph_from_model(model)
    return lrg

def generate_qsum_hw(lrg, dir):
    logic_impl = lrg.logic_nodes[1].logic_impl
    print("\n\n\n")  
    print(logic_impl) 
    project_dir = dir
    module_file, instance_decl, input_port_conns, output_port_conns = logic_impl.generate_hw(project_dir=project_dir, node=lrg.logic_nodes[1])
    # print(f"\n\n {module_file} {instance_decl} {input_port_conns} {output_port_conns}")
    
# def test_qsum_gen(model = simple_qsum()):
#     lrg = test_build_lrgraph_from_model(model, f"/homes/bm920/workspace/da4ml/.tmp/figures/qsum_lrg")
#     logic_impl = lrg.logic_nodes[1].logic_impl
#     print("\n\n\n")  
#     print(logic_impl) 
#     project_dir = f"{RTL_DIR}"
#     module_file, instance_decl, input_port_conns, output_port_conns = logic_impl.generate_hw(project_dir=project_dir, node=lrg.logic_nodes[1])
#     print(f"\n\n {module_file} {instance_decl} {input_port_conns} {output_port_conns}")

