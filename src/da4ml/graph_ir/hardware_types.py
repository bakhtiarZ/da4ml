from abc import ABC, abstractmethod

from dataclasses import dataclass
import math
import math
from platform import node
import keras
import numpy as np

from hgq._dais_tracer.layers._base import mirror_quantizer
from hgq.layers import QAdd
from da4ml.trace import FixedVariableArrayInput, FixedVariableArray, comb_trace
from da4ml.converter import trace_model
from da4ml.codegen.rtl.rtl_model import RTLModel, get_io_kifs
from hgq.config import QuantizerConfig
from da4ml.trace.ops import quantize
from .util import OpRepr, _strip_batch_and_ensure_ints, convert_kif_to_streaming_shape

class CustomLogic(ABC):
    
    @abstractmethod
    def __init__(self, opr):
        self.opr = opr
    
    @abstractmethod
    def configure(self, node):
        pass
    
    @abstractmethod
    def generate_hw(self, project_dir):
        pass
    

class PureLogic:
    def __init__(self, opr: OpRepr):
        self.opr = opr
    def __repr__(self) -> str:
        return f"PureLogic(opr={self.opr})"

@dataclass
class RoutingLogic:
    buffer_type: str = "fifo"
    buffer_shape: tuple[int, int] = (-1, -1)


class HWInterface:
    def __init__(self, node):
        self.node = node
        if issubclass(type(node.logic_impl), CustomLogic):
            self.comb_logic = node.logic_impl.internal_comb_logic
            self.input_kif = node.logic_impl.input_kifs
            self.output_kif = node.logic_impl.output_kifs
        else:
            self.comb_logic = node.logic_impl
            # self.input_kif, self.output_kif = get_io_kifs(self.comb_logic)
            self.input_kif = self.comb_logic.inp_kifs[:,np.newaxis,:]
            self.output_kif = self.comb_logic.out_kifs[:,np.newaxis,:]
            
        self.input_bitwidth = int(sum(np.max(arr) for arr in self.input_kif))
        self.output_bitwidth = int(sum(np.max(arr) for arr in self.output_kif))
        self.input_item_size = math.prod(node.input_shapes[node.input_tids[0]])
        self.output_item_size = math.prod(node.output_shapes[node.output_tids[0]])
    
    def get_input_bw_is(self):
        return self.input_bitwidth, self.input_item_size

    def get_output_bw_is(self):
        return self.output_bitwidth, self.output_item_size


class PortConnection:
    def __init__(self, data: tuple[str, int], valid: str, ready: str):
        self.data = data # (name, bitwidth)
        self.valid = valid
        self.ready = ready
    
    def get_intermediate_decls(self):
        data_decl = f"logic [{self.data[1]-1}:0] {self.data[0]};"
        valid_decl = f"logic {self.valid};"
        ready_decl = f"logic {self.ready};"
        return data_decl, valid_decl, ready_decl
    
    def __str__(self):
        return f"data={self.data}, valid={self.valid}, ready={self.ready}"

