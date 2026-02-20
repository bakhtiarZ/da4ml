import keras

from hgq.layers import QDense
from hgq.config import QuantizerConfig

from da4ml.codegen.rtl.rtl_model import RTLModel, get_io_kifs
from da4ml.converter.hgq2.parser import trace_model
from da4ml.converter.hgq2.parser import comb_trace

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

quints()