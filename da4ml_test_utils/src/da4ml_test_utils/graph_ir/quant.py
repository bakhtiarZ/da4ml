import keras
import hgq

from da4ml.trace import FixedVariableArrayInput, FixedVariableArray
from hgq._dais_tracer.layers._base import mirror_quantizer
from da4ml.converter import trace_model
from da4ml.trace import comb_trace
import numpy as np

from da4ml.graph_ir.hardware_types import QSumLogic, QSumGen 

def printdump(dump):
    for k, v in dump.items():
        print(f"k: {k}, \nv: {v[0]}\nkif={v[0].kif}\n\n\n")

def tmp():
    with hgq.config.QuantizerConfigScope(place='datalane', heterogeneous_axis=(-1,)):
        s = hgq.layers.QSum(axes=1)
        l0 = hgq.layers.QDense(2, name='l0')
        l1 = hgq.layers.QDense(1, name='l1')

    m0, m1 = keras.Sequential([l0], name='m0'), keras.Sequential([l1], name='m1')

    _inp = keras.Input((4, 3))
    m = keras.Model(_inp, l1(l0(s(_inp))))
    model = keras.Model(inputs=_inp, outputs=m1(m0(_inp)))
    # dump = trace_model(model, dump=True, inputs=FixedVariableArrayInput((1, 16)))
    dump = trace_model(m, dump=True)
    for k, v in dump.items():
        print(f"k: {k}, \nv: {v[0]}\nkif={v[0].kif}\n\n\n")
        
    kif_0_1 = dump['/m1/l1/post_iq'][0].kif
    comb0 = comb_trace(dump['inputs'], dump['/m1/l1/post_iq'])
    # construct from trace
    # trace starting from non-new variables not always work due to in-place simplifications.

    # construct manually
    inp, out = trace_model(m0, inputs=FixedVariableArrayInput((1, 16)))
    out = mirror_quantizer(m1.layers[0].iq, out[None])
    _comb0 = comb_trace(inp,out)

    inp = FixedVariableArray.from_kif(*kif_0_1)
    comb1 = comb_trace(*trace_model(m1, inputs=inp))
    print(comb0 == _comb0)
    print(comb0.out_kifs == comb1.inp_kifs)

"""

take dump object, and all layers input and output quantizers, build a dict of layer_name to post_iq, post_call, final, fx arrays, 
and also that layers iq, and oq if it exists.

then in building the comb logics, need to use post_iq kifs of that layer, and fixedvararrayfromkif. then apply mirror quantizer to output.


CHECK IF THE QSUM QUANT VARIES IF THE STREAMED QSUM SHAPE CHANGES

"""


_inp = keras.Input((4, 3))
l = hgq.layers.QSum(axes=1)
m = keras.Model(_inp, l(_inp))
dump = trace_model(m, dump=True)
printdump(dump)

