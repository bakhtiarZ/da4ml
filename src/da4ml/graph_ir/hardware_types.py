from abc import ABC, abstractmethod

from dataclasses import dataclass
import math
import math
from platform import node
import keras
import numpy as np

from hgq.layers import QAdd
from da4ml.trace import FixedVariableArrayInput, comb_trace
from da4ml.converter import trace_model
from da4ml.codegen.rtl.rtl_model import RTLModel, get_io_kifs
from hgq.config import QuantizerConfig
from da4ml.trace.ops import quantize
from .util import OpRepr, _strip_batch_and_ensure_ints

class CustomLogic(ABC):
    
    @abstractmethod
    def __init__(self, node):
        self.node = node
    
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
            self.input_kif = node.logic_impl.input_kif
            self.output_kif = node.logic_impl.output_kif
        else:
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


class QSumLogic(CustomLogic):
    def __init__(self, opr):
        self.opr = opr
        self.axis_with_batch = opr.operation.axes[0] 
        self.axis_without_batch = self.axis_with_batch - 1
        
        
    def __repr__(self) -> str:
        return f"QSumLogic(opr={self.opr.operation.name}, axis_with_batch={self.axis_with_batch}, axis_without_batch={self.axis_without_batch})"
    
    def configure(self, node):
        self.node = node
        self.node_id = node.op_id
        self.qsumgen = QSumGen(self.node, None, self.node_id, self.axis_without_batch)
        self.qsumgen.configure()
        
        self.internal_comb_logic = self.qsumgen.internal_adder
        self.input_kif = self.qsumgen.input_kif
        self.output_kif = self.qsumgen.output_kif
        self.input_bitwidth = sum(np.max(arr) for arr in self.input_kif)
        self.output_bitwidth = sum(np.max(arr) for arr in self.output_kif)
        self.input_item_size = node.input_shapes[node.input_tids[0]][-1] 
        self.output_item_size = node.output_shapes[node.output_tids[0]][-1]
        print(f"\n\n DEBUG QSumLogic configured with input_kif: {self.input_kif}, output_kif: {self.output_kif}, input_bitwidth: {self.input_bitwidth}, output_bitwidth: {self.output_bitwidth}, input_item_size: {self.input_item_size}, output_item_size: {self.output_item_size}\n\n")
    
    def generate_hw(self, project_dir) -> str:
        assert self.qsumgen is not None, "QSumGen is not configured. Please call configure(node) before generating hardware."
        return self.qsumgen.write_hw(project_dir)
    
class QSumGen():
    def __init__(self, node, project_dir: str, node_id: int, axis: int):
        self.node = node
        self.project_dir = project_dir
        self.node_id = node_id
        self.axis = axis
        
    def configure(self):
        self.k = self.node.operation.iq.config.config['k0']
        self.i = self.node.operation.iq.config.config['i0']
        self.f = self.node.operation.iq.config.config['f0']
        self.input_bitwidth = int(self.k) + int(self.i) + int(self.f)
        print(f"\n\n DEBUG k: {self.k}, i: {self.i}, f: {self.f}, input_bitwidth: {self.input_bitwidth}\n\n")
        self.semantic_input_shape = _strip_batch_and_ensure_ints(self.node.op_repr.requires[0].shape)
        self.streaming_input_shape = self.node.input_shapes[self.node.input_tids[0]]
        print(f"\n\n DEBUG semantic_input_shape: {self.semantic_input_shape}, streaming_input_shape: {self.streaming_input_shape}\n\n")
        self.accum_count = int(self.semantic_input_shape[self.axis] / self.streaming_input_shape[self.axis])
        self.semantic_output_shape = _strip_batch_and_ensure_ints(self.node.op_repr.produces[0].shape)
        self.streaming_output_shape = self.node.output_shapes[self.node.output_tids[0]]
        self.input_item_size = math.prod(self.streaming_input_shape)
        self.output_item_size = math.prod(self.streaming_output_shape)
        print(f"\n\n DEBUG semantic_input_shape: {self.semantic_input_shape}, semantic_output_shape: {self.semantic_output_shape}\n\n {self.streaming_input_shape} {self.streaming_output_shape}\n\n")
        assert self.semantic_output_shape[self.axis] == 1, f"Currently only support summing to a single value along the reduction axis, semantic_output_shape: {self.semantic_output_shape}, axis = {self.axis}, semantic input shape: {self.semantic_input_shape}"
        assert self.streaming_output_shape[self.axis] == 1, f"Currently only support summing to a single value along the reduction axis, but got streaming_output_shape: {self.streaming_output_shape}"
        self.accum_count = int(self.semantic_input_shape[self.axis] / self.streaming_input_shape[self.axis])    
        print(f"\n\n DEBUG accum_count: {self.accum_count}\n\n")
        self.module_port_names = {
            "clk": "clk",
            "rst": "rst",
            "data_in": f"data_in",
            "in_valid": f"in_valid",
            "in_ready": f"in_ready",
            "data_out": f"data_out",
            "out_valid": f"out_valid",
            "out_ready": f"out_ready"
        }
        self._create_internal_adder()

    def _create_internal_adder(self):
        # performs an adder tree reduction on the streamed inputs along the axis specified
        axis_without_batch = self.axis - 1
        # input_to_adder_shape = (
        #     self.streaming_input_shape[:axis_without_batch]
        #     + (self.streaming_input_shape[axis_without_batch] + 1,)
        #     + self.streaming_input_shape[axis_without_batch + 1:]
        # )
        cursum_input_shape = self.semantic_output_shape        
        
        def __get_output_quantisation_of_adder():
            inp = FixedVariableArrayInput(self.semantic_input_shape)
            inp = quantize(inp, self.k, self.i, self.f)
            out = np.sum(inp, axis=axis_without_batch, keepdims=True)
            cl = comb_trace(inp, out)
            output_kif = get_io_kifs(cl)[1]
            k, i, f =  output_kif
            print(f"\n\n DEBUG output quantization k: {k}, i: {i}, f: {f}\n\n")
            return max(k), max(i), max(f)
        
        output_k, output_i, output_f = __get_output_quantisation_of_adder()
        self.sum_reg_width = output_k + output_i + output_f 
        data_in = quantize(FixedVariableArrayInput(self.streaming_input_shape), self.k, self.i, self.f)
        cur_sum = quantize(FixedVariableArrayInput(cursum_input_shape), output_k, output_i, output_f)
        data_in_and_cursum = np.concatenate([data_in, cur_sum], axis=axis_without_batch)
        out = quantize(np.sum(data_in_and_cursum, axis=axis_without_batch, keepdims=True), output_k, output_i, output_f)
        self.internal_adder = comb_trace(data_in_and_cursum, out)
        total_inp_kif, self.output_kif = get_io_kifs(self.internal_adder)
        self.input_kif = total_inp_kif[:,:-1]
        self.output_bitwidth = sum(np.max(arr) for arr in self.output_kif)
        # print(f"\n\n DEBUG internal adder output_kif: {self.output_kif}, output_bitwidth: {self.output_bitwidth}\n\n")
    
    def _create_preamble(self):
        module_definition = (" module QSum #() (\n" 
        f"    input logic clk,\n"
        f"    input logic rst,\n"
        f"    input logic [{self.input_bitwidth * self.input_item_size - 1}:0] {self.module_port_names['data_in']},\n"
        f"    input logic {self.module_port_names['in_valid']},\n"
        f"    output logic {self.module_port_names['in_ready']},\n"
        f"    output logic [{self.output_bitwidth * self.output_item_size - 1}:0] {self.module_port_names['data_out']},\n"
        f"    output logic {self.module_port_names['out_valid']},\n"
        f"    input logic {self.module_port_names['out_ready']}\n"
        ");\n\n")
        return module_definition
    
    def _write_body_of_module(self):
        adder_name = self.adder_instance_name
        # print(f"\n\n DEBUG accum_count: {self.accum_count}, input_bitwidth: {self.input_bitwidth}, input_item_size: {self.input_item_size}, output_bitwidth: {self.output_bitwidth}, output_item_size: {self.output_item_size}\n\n`")
        count_w = 1 if self.accum_count <= 1 else math.ceil(math.log2(self.accum_count + 1))
        input_width = self.input_bitwidth * self.input_item_size
        output_width = self.output_bitwidth * self.output_item_size
        lines = (
            f"    localparam logic [{count_w-1}:0] COUNT_MINUS_ONE = {self.accum_count - 1};\n"
            f"    localparam logic [{count_w-1}:0] COUNT_VALUE     = {self.accum_count};\n"
            f"\n" \
            f"    logic [{self.sum_reg_width - 1}:0] sum_reg;\n"
            f"    logic [{count_w-1}:0] count_reg;\n"
            f"    logic full_reg;\n"
            f"\n"
            f"    logic [{output_width - 1}:0] adder_result;\n"
            f"    logic accept_input;\n"
            f"    logic accept_output;\n"
            f"    logic last_input;\n"
            f"\n"
            f"    logic [{input_width + output_width - 1}:0] packed_operands; assign packed_operands = {{{self.module_port_names['data_in']}, sum_reg}};\n"
            f"    mod_{adder_name} #(\n"
            f"    ) {adder_name}_inst (\n"
            f"        .model_inp(packed_operands),\n"
            f"        .model_out(adder_result)\n"
            f"    );\n"
            f"\n"
            f"    assign {self.module_port_names['in_ready']}      = !full_reg;\n"
            f"    assign {self.module_port_names['out_valid']}     = full_reg;\n"
            f"    assign {self.module_port_names['data_out']}      = sum_reg;\n"
            f"\n"
            f"    assign accept_input  = {self.module_port_names['in_valid']} && {self.module_port_names['in_ready']};\n"
            f"    assign accept_output = {self.module_port_names['out_valid']} && {self.module_port_names['out_ready']};\n"
            f"    assign last_input    = (count_reg == COUNT_MINUS_ONE);\n"
            f"\n"
            f"    always_ff @(posedge clk) begin\n"
            f"        if (rst) begin\n"
            f"            sum_reg   <= '0;\n"
            f"            count_reg <= '0;\n"
            f"            full_reg  <= 1'b0;\n"
            f"        end else begin\n"
            f"            if (accept_output) begin\n"
            f"                sum_reg   <= '0;\n"
            f"                count_reg <= '0;\n"
            f"                full_reg  <= 1'b0;\n"
            f"            end else if (accept_input) begin\n"
            f"                sum_reg <= adder_result;\n"
            f"\n"
            f"                if (last_input) begin\n"
            f"                    count_reg <= COUNT_VALUE;\n"
            f"                    full_reg  <= 1'b1;\n"
            f"                end else begin\n"
            f"                    count_reg <= count_reg + 1'b1;\n"
            f"                end\n"
            f"            end\n"
            f"        end\n"
            f"    end\n"
            f"\nendmodule\n"
        )

        return lines

    def _write_module_file(self, project_dir):
        # 1) write the internal adder
        self.adder_instance_name = f"adder_for_qsum_node_{self.node_id}"
        rtl_model = RTLModel(
            solution=self.internal_adder,
            prj_name=f"mod_{self.adder_instance_name}",
            path=project_dir,
            flavor="verilog",
        )
        rtl_model.write()
        # 2) write the module itself
        preamble = self._create_preamble()
        body = self._write_body_of_module()
        module_file_content = preamble + body
        module_file_path = f"{project_dir}/qsum_module.sv"
        with open(module_file_path, "w") as f:
            f.write(module_file_content)
        return module_file_path
    
    def _write_instance_decl(self):
        # input port conns, output port conns, wiring and instance decl
        lines = []
        lines.append(f"// **** Instance of QSum for node {self.node_id} ****\n")
        input_port_conns = PortConnection(
            data=(f"data_in_{self.node_id}", self.input_bitwidth * self.input_item_size),
            valid=f"in_valid_{self.node_id}",
            ready=f"in_ready_{self.node_id}"
        )
        input_port_decls = input_port_conns.get_intermediate_decls()
        lines.append("\n    // Intermediate signals for input port\n")
        lines.extend(input_port_decls)
        output_port_conns = PortConnection(
            data=(f"data_out_{self.node_id}", self.output_bitwidth * self.output_item_size),
            valid=f"out_valid_{self.node_id}",
            ready=f"out_ready_{self.node_id}"
        )
        output_port_decls = output_port_conns.get_intermediate_decls()
        lines.append("\n    // Intermediate signals for output port\n")
        lines.extend(output_port_decls)
        lines.append("\n    // Instance declaration\n")
        port_conns = (
            f".clk(clk),\n"
            f".rst(rst),\n"
            f".{self.module_port_names['data_in']}(data_in_{self.node_id}),\n"
            f".{self.module_port_names['in_valid']}(in_valid_{self.node_id}),\n"
            f".{self.module_port_names['in_ready']}(in_ready_{self.node_id}),\n"
            f".{self.module_port_names['data_out']}(data_out_{self.node_id}),\n"
            f".{self.module_port_names['out_valid']}(out_valid_{self.node_id}),\n"
            f".{self.module_port_names['out_ready']}(out_ready_{self.node_id})\n"
        )
        lines.append(f"QSum #( ) qsum_inst_{self.node_id} (\n{port_conns});\n")
        instance_decl = "\n".join(lines)
        return instance_decl, input_port_conns, output_port_conns
    
    def write_hw(self, project_dir):
        # should create the necessary internal files, then create a top level module, and write them all to the specified path
        # top_level_module = self._write_top_level()
        module_file = self._write_module_file(project_dir)
        instance_decl, input_port_conns, output_port_conns = self._write_instance_decl()
        return module_file, instance_decl, input_port_conns, output_port_conns
