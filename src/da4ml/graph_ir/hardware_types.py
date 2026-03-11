from dataclasses import dataclass
import keras

from hgq.layers import QAdd
from da4ml.trace import FixedVariableArrayInput, comb_trace
from da4ml.converter import trace_model
from da4ml.codegen.rtl.rtl_model import RTLModel, get_io_kifs
from hgq.config import QuantizerConfig

from .lr_graph import LogicNode



@dataclass
class PureLogic:
    empty_logic: bool = True  # marker type


class HWInterface:
    def __init__(self, node: LogicNode):
        self.node = node
        self.comb_logic = node.logic_impl
        self.input_kif, self.output_kif = get_io_kifs(self.comb_logic)
        self.input_bitwidth = sum(np.max(arr) for arr in self.input_kif)
        self.output_bitwidth = sum(np.max(arr) for arr in self.output_kif)
        self.input_item_size = node.input_shapes[node.input_tids[0]][-1] 
        self.output_item_size = node.output_shapes[node.output_tids[0]][-1]
    
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

class QSumLogic: # will be used to generate a 'Sum' hardware, this means it needs information about the data in, data out, and when to reset its internal sum
    def __init__(self, data_in: HWInterface, data_out: HWInterface, input_sematic_shape, input_streaming_shape, axis):
        self.input_bitwidth, self.input_item_size = data_in.get_input_bw_is()
        self.output_bitwidth, self.output_item_size = data_out.get_output_bw_is()
        self.data_in_interface = PortConnection(data = (f"data_in_{data_in.node.op_id}", self.input_bitwidth), valid = f"data_in_valid_{data_in.node.op_id}", ready = f"data_in_ready_{data_in.node.op_id}")
        self.data_out_interface = PortConnection(data = (f"data_out_{data_out.node.op_id}", self.output_bitwidth), valid = f"data_out_valid_{data_out.node.op_id}", ready = f"data_out_ready_{data_out.node.op_id}")
        
        ## assume the axis is the one being streamed over otherwise it gets very complicated
        self.input_sematic_shape = input_sematic_shape
        self.input_streaming_shape = input_streaming_shape
        self.axis = axis
        
        self.accum_count = int(self.input_sematic_shape[axis]) / int(self.input_streaming_shape[axis])
        # calculate when to reset the register, and logic for asserting data in and data out valid ready etc
        
    def _create_internal_adder(self):
        # this will be responsible for adding 2 values that are the same quantisation
        i0 = keras.Input(shape=self.input_streaming_shape)
        i1 = keras.Input(shape=self.input_streaming_shape)
        o = QAdd(iq_conf=QuantizerConfig(heterogeneous_axis=()))([i0, i1])
        model = keras.Model(inputs=[i0,i1], outputs=o)
        i,o = trace_model(model)
        self.internal_adder = comb_trace(i,o)
    
    def _create_preamble(self):
        module_definition = "" \ 
        "module QSum #(" \
        # f"  ACCUM_COUNT = {self.accum_count},"
        # f"  IN_WIDTH = {self.input_bitwidth * self.input_item_size}"
        # f"  OUT_WIDTH = {self.output_bitwidth * self.output_item_size}"
        ") ("
        f"  input logic clk,"
        f"  input logic rst,"
        f"  input logic [{self.input_bitwidth * self.input_item_size - 1}:0] {self.data_in_interface.data[0]},"
        f"  input logic {self.data_in_interface.valid},"
        f"  output logic {self.data_in_interface.ready},"
        f"  output logic [{self.output_bitwidth * self.output_item_size - 1}:0] {self.data_out_interface.data[0]},"
        f"  output logic {self.data_out_interface.valid},"
        f"  input logic {self.data_out_interface.ready}"
        ");"
        return module_definition
    
    def _write_top_level(self):
        preamble = self._create_preamble()
        # define and connect control logic, maybe use a state machine
        lines = ""
        "\n\n" \
        "localparam int ACCUM_COUNT = {self.accum_count};\n" \
        f"logic [{self.output_bitwidth * self.output_item_size - 1}:0] sum_reg;\n" \
        f"logic [$clog2(ACCUM_COUNT):0] count_reg;\n" \
        f"logic [{self.output_bitwidth * self.output_item_size - 1}:0] adder_result;\n" \
        f"// internal adder instance\n" \
        f"QAdd internal_adder (\n" \
        f"  .i0({self.data_in_interface.data[0]}),\n" \
        f"  .i1(sum_reg),\n" \
        f"  .o(adder_result)\n" \
        f");\n" \
        f"// control logic\n" \
        f"logic accept_input; assign accept_input = {self.data_in_interface.valid} && {self.data_in_interface.ready};\n" \
        f"logic output_accepted; assign output_accepted = {self.data_out_interface.valid} && {self.data_out_interface.ready};\n" \
        f"assign {self.data_in_interface.ready} = !{self.data_out_interface.valid};\n" \
        f"assign last_input_of_group = accept_input && (count_reg == ACCUM_COUNT - 1);\n" \
        f"always_ff @(posedge clk) begin\n" \
        f"  if (rst) begin\n" \
        f"    sum_reg <= 0;\n" \
        f"    count_reg <= 0;\n" \
        f"    {self.data_out_interface.valid} <= 0;\n" \
        f"    {self.data_out_interface.data[0]} <= 0;\n" \
        f"  end else begin\n" \
        f"  if (accept_output) begin\n" \
        f"    {self.data_out_interface.valid} <= 0;\n" \
        f"    count_reg <= 0;\n" \
        f"    accum_reg <= 0;\n" \
        f"  end\n" \
        f"  if (accept_input) begin\n" \
        f"      if (last_input_of_group) begin\n" \
        f"          sum_reg <= adder_result;\n" \
        f"          {self.data_out_interface.valid} <= 1;\n" \
        f"          count_reg <= count_reg;\n" \
        f"      end else begin\n" \
        f"          sum_reg <= adder_result;\n" \
        f"          count_reg <= count_reg + 1;\n" \
        f"      end\n" \
        f"  end\n" \
        f"  end\n" \
        "end\n\n" \
        "endmodule"

        
    def write_hw(self, path):
        # should create the necessary internal files, then create a top level module, and write them all to the specified path
        top_level_module = self._write_top_level()
