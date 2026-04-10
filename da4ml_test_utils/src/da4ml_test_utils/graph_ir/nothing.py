import keras
import numpy as np

from hgq.layers import QDense, QAdd, QSum
from da4ml.trace import FixedVariableArrayInput, comb_trace, FixedVariableArray
from da4ml.trace.ops import quantize
from da4ml.converter import trace_model
from da4ml.codegen.rtl.rtl_model import RTLModel, get_io_kifs
from hgq.config import QuantizerConfig

from da4ml.graph_ir.lr_graph import build_lr_graph_from_model, lr_graph_to_hardware

def two_layer_model():
    i = keras.Input((3,2))
    d0 = QDense(3, iq_conf=QuantizerConfig(heterogeneous_axis=()), kernel_initializer='ones', bias_initializer='zeros')
    d1 = QDense(1, iq_conf=QuantizerConfig(heterogeneous_axis=()), kernel_initializer='ones', bias_initializer='zeros')
    out = d1(d0(i))
    m = keras.Model(i, out)
    return m

def quints():
    m = two_layer_model()
    i, o = trace_model(m)
    cs = comb_trace(i,o)
    # rtl = RTLModel(cs, )
    print(get_io_kifs(cs))
    import numpy as np
    a = map(np.max, get_io_kifs(cs))
    for i in a :
        print(i)
    import pdb; pdb.set_trace()
    print("asd")

def create_qadd():
    i0 = keras.Input(shape=(1,))
    i1 = keras.Input(shape=(1,))
    o = QAdd(iq_confs=[QuantizerConfig(heterogeneous_axis=()), QuantizerConfig(heterogeneous_axis=())])([i0, i1])
    model = keras.Model(inputs=[i0,i1], outputs=o)
    i,o = trace_model(model)
    cl = comb_trace(i,o)
    instance_name = f"adder_for_qsum_node"
    project_dir = f"/homes/bm920/workspace/da4ml/.tmp/adder_for_qsum_node_project"
    rtl_model = RTLModel(
        solution=cl,
        prj_name=f"mod_{instance_name}",
        path=project_dir,
        flavor="verilog",
    )
    rtl_model.write()
    
# create_qadd()

# def create_internal_adder():
#     inp = FixedVariableArrayInput(inp_to_adder_shape) # i.e. 3,
#     inp = quantize(inp, k, i, f)
#     out = np.sum(inp, axis=0, keepdims=True)
#     cl = comb_trace(inp, out)
    
def test_da4ml():
    inp = FixedVariableArrayInput((4,))
    k=1
    i=4
    f=3
    overflow_mode = "WRAP"
    round_mode = "RND"
    iq = quantize(inp, k, i, f)
    s = np.sum(iq, axis=0, keepdims=True)
    cl = comb_trace(inp, s)
    print(cl)
    print(cl.inp_qint)

# def sum_and_accumulate():
#     k=1
#     i=4
#     f=3
#     overflow_mode = "WRAP"
#     round_mode = "RND"
#     #####
#     data_in = FixedVariableArrayInput((2,3))
#     sum_reg = FixedVariableArrayInput((1,3))
#     inps = np.array([data_in, sum_reg]);
#     data_in_q = quantize(data_in, k, i, f)
#     sum_reg_q = quantize(data_in, k, i, f)
    
#     # first sum data_in_q
#     cur_sum = np.sum(data_in_q, axis=0, keepdims=True) # results in a 1,3
    
#     # then accumulate
#     sum_req_q = sum_reg_q + cur_sum
#     cl = comb_trace(inps, sum_reg_q)
#     print(cl)

def sum_and_accumulate():
    k=1
    i=4
    f=3
    overflow_mode = "WRAP"
    round_mode = "RND"
    #####
    data_in_and_cursum = FixedVariableArrayInput((3,3))
    q_concat = quantize(data_in_and_cursum, k, i, f)
    cursum = data_in_and_cursum[-1,:]
    data_in = data_in_and_cursum[:-1,:]
    summed_data_in = np.sum(data_in, axis=0, keepdims=True)

    res = summed_data_in + cursum # shouldn't this just be another sum?
    cl = comb_trace(data_in_and_cursum, res)
    print(cl)
    
def sum_and_accumulate_2():
    k=1
    i=4
    f=3
    overflow_mode = "WRAP"
    round_mode = "RND"
    #####
    data_in_and_cursum = FixedVariableArrayInput((3,3))
    q = quantize(data_in_and_cursum, k, i, f)

# def sum_and_accumulate_3():
#     k=1
#     i=4
#     f=3
#     overflow_mode = "WRAP"
#     round_mode = "RND"
#     #####
#     data_in = quantize(FixedVariableArrayInput((2,3)), k, i, f)
#     cursum  = quantize(FixedVariableArrayInput((1,3)), k, i+1, f)
#     data_in_and_cursum = np.concatenate([data_in, cursum], axis=0)
#     cl = comb_trace(data_in_and_cursum, np.sum(data_in_and_cursum, axis=0, keepdims=True))
#     print(cl)
#     k, i, f =  get_io_kifs(cl)[1]
    
#     def get_k_i_f(cl):
        
#     print(f"DEBUG: k={k}, i={i}, f={f}")
    

# sum_and_accumulate_3()
# self.k = self.node.operation.iq.config.config['k0']
# self.i = self.node.operation.iq.config.config['i0']
# self.f = self.node.operation.iq.config.config['f0']

def print_quantisation_info(m):
    print(f"Model: {m.name}\n")
    for layer in m.layers:
        if type(layer) == keras.layers.InputLayer:
            continue
        # print(f"Layer: {layer.name}, Dir: {dir(layer)} \n\n\n")
        iq = getattr(layer, "iq", None)
        oq = getattr(layer, "oq", None)
        print(f"Layer: {layer.name}:\n\tiq_config = {iq.config.config if iq else None} \n\toq_config = {oq.config.config if oq else None} \n\n")
        
        
def m_with_qsum():
    i = keras.Input((4,2))
    d0 = QDense(1, iq_conf=QuantizerConfig(heterogeneous_axis=()), kernel_initializer='ones', bias_initializer='zeros')(i) # 4, 1
    s = QSum(iq_conf=QuantizerConfig(heterogeneous_axis=()), axes=1, scale=1, keepdims=True)(d0) # 1, 1
    m = keras.Model(i, s)
    m.name = "model_with_qsum"
    return m

def m_with_qsum_and_iq_oq():
    i = keras.Input((4,2))
    d0 = QDense(1, iq_conf=QuantizerConfig(heterogeneous_axis=()), kernel_initializer='ones', bias_initializer='zeros', enable_iq = True, enable_oq = True)(i) # 4, 1
    s = QSum(iq_conf=QuantizerConfig(heterogeneous_axis=()), axes=1, scale=1, keepdims=True, enable_iq = True)(d0) # 1, 1
    m = keras.Model(i, s)
    m.name = "model_with_qsum_and_iq_oq"
    return m

def m_with_qsum_and_fixed_conf():
    qconf = QuantizerConfig(heterogeneous_axis=(), k0=1, i0=4, f0=3)
    i = keras.Input((4,2))
    d0 = QDense(1, iq_conf=qconf, oq_conf = qconf, kernel_initializer='ones', bias_initializer='zeros', enable_iq = True, enable_oq = True)(i) # 4, 1
    s = QSum(iq_conf=qconf, axes=1, scale=1, keepdims=True, enable_iq = True)(d0) # 1, 1
    m = keras.Model(i, s)
    m.name = "model_with_qsum_and_fixed_conf"
    return m

def big_m_with_qsum_and_fixed_conf():
    qconf = QuantizerConfig(heterogeneous_axis=(), k0=1, i0=4, f0=3)
    i = keras.Input((64,2))
    d0 = QDense(1, iq_conf=qconf, oq_conf = qconf, kernel_initializer='ones', bias_initializer='zeros', enable_iq = True, enable_oq = True)(i) # 4, 1
    s = QSum(iq_conf=qconf, axes=1, scale=1, keepdims=True, enable_iq = True)(d0) # 1, 1
    m = keras.Model(i, s)
    m.name = "model_with_qsum_and_fixed_conf"
    return m

# m = m_with_qsum()
# print_quantisation_info(m)
# m0 = m_with_qsum_and_iq_oq()
# m1 = m_with_qsum_and_fixed_conf()

# print_quantisation_info(m0)
# print_quantisation_info(m1)

# m = big_m_with_qsum_and_fixed_conf()
# lrg = build_lr_graph_from_model(m, parallelism=64)
# hw = lr_graph_to_hardware(lrg, "delete_me", debug=True)

