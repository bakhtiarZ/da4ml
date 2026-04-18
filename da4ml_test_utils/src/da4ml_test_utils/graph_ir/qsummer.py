import keras
import hgq

from da4ml.trace import FixedVariableArrayInput, FixedVariableArray
from hgq._dais_tracer.layers._base import mirror_quantizer
from da4ml.converter import trace_model
from da4ml.trace import comb_trace
import numpy as np
from da4ml.graph_ir.lr_graph import *

from da4ml.graph_ir.hardware_types import QSumLogic, QSumGen 

def printdump(dump):
    for k, v in dump.items():
        print(f"k: {k}, \nv: {v[0]}\nkif={v[0].kif}\n\n\n")
        
        
with hgq.config.QuantizerConfigScope(place='datalane', heterogeneous_axis=(-1,)):
    _inp = keras.Input(shape=(4, 3))
    s = hgq.layers.QSum(axes=1, keepdims=True)
    d0 = hgq.layers.QDense(2)
    m = keras.Model(_inp, d0(s(_inp)))
    m.name = "model_with_qsum_inject_kifs"
    dump = trace_model(m, dump=True)
    printdump(dump)
    
    qsinput_kifs = dump["/q_sum/post_iq"][0].kif
    qsoutput_kifs = dump["/q_dense/post_iq"][0].kif
    dense_iq = d0.iq

    lrg = build_lr_graph_from_model(m)
    qsnode = lrg.logic_nodes[1]
    lr_graph_to_hardware(lrg, f".tmp/{m.name}", debug=True)
    print(f"input_kif:\n{lrg.logic_nodes[1].logic_impl.input_kifs}\n\n\noutput_kif:\n{lrg.logic_nodes[1].logic_impl.output_kifs}\n")