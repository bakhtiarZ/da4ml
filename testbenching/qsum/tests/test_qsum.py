import math
from typing import Optional

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

import numpy as np
import tensorflow as tf
from da4ml_test_utils.graph_ir.test_lrgraph import Source, two_layer_model, encode_to_d, encode_to
from da4ml.graph_ir.hardware_types import HWInterface
from da4ml.graph_ir.lr_graph import LRGraph, build_lr_graph_from_model, lr_to_dot, configure_custom_logic_nodes, get_top_level_interface
from da4ml.codegen.rtl.rtl_model import get_io_kifs
from hgq.layers import QDense
from hgq.config import QuantizerConfig

from qsum_definitions import m_with_qsum, simple_qsum, qsum_lrg, generate_qsum_hw


# ============================================================
# Edit these directly
# ============================================================
PARAMS = {
    # clock/reset
    "CLK_PERIOD_NS": 10,
    "RESET_CYCLES": 5,

    # Input packing: one "row" per cycle, row has IN_ELEMS elements each IN_ELEM_W bits
    "IN_ELEMS": 3,
    "IN_ELEM_W": 8,

    # Output unpacking: each cycle produces OUT_ELEMS elements each OUT_ELEM_W bits
    "OUT_ELEMS": 1,
    "OUT_ELEM_W": 8,

    # Your input tensor (T rows). Example: (T=2, IN_ELEMS=2) => [[..,..],[..,..]]
    "VECTORS": [
        [1, 1],
        [1, 1],
        [1, 1],
    ],

    # How to align streamed RTL output to Keras output
    # Option A (recommended): discard first N output cycles (pipeline latency), then take next T rows
    "ALIGN_MODE": "discard",     # "discard" or "take_last"
    "DISCARD_OUT_CYCLES": 3,     # set this to your pipeline latency in cycles

    # Flush cycles after last input is sent (give pipeline time to finish)
    "FLUSH_CYCLES": 10,
}
def build_golden_model(model = simple_qsum()):
    return model

def hex_to_bin(hex_val: str | int, bitwidth: int) -> str:
    if isinstance(hex_val, str):
        hex_val = hex_val.lower().replace("0x", "")
        value = int(hex_val, 16)
    else:
        value = int(hex_val)

    return format(value, f"0{bitwidth}b")

def get_top_level_hw_interfaces(lrg: LRGraph):
    first_node = lrg.logic_nodes[1] # skip input
    fnhwi = HWInterface(first_node)
    last_node = lrg.logic_nodes[max(lrg.logic_nodes.keys()) - 1] # skip output
    lnhwi = HWInterface(last_node)
    return fnhwi, lnhwi

class TestTopSimpleTB:
    
    def __init__(self, model = build_golden_model()):
        self.model = model
        self.lrg = build_lr_graph_from_model(model)
        dot_str = lr_to_dot(self.lrg)
        configure_custom_logic_nodes(self.lrg)
        src = Source(dot_str)
        src.render(f"./lr_graph_{self.model.name}", format="svg", view=True)
        
        self.inp_hwi, self.out_hwi = get_top_level_hw_interfaces(self.lrg)
        self.input_bitwidth_unpacked = self.inp_hwi.input_bitwidth
        self.input_item_size = self.inp_hwi.input_item_size
        self.input_kif = self.inp_hwi.input_kif
        self.input_bitwidth_packed = self.input_bitwidth_unpacked * self.input_item_size
        
        self.output_bitwidth_unpacked = self.out_hwi.output_bitwidth
        self.output_item_size = self.out_hwi.output_item_size
        self.output_kif = self.out_hwi.output_kif        
        self.output_bitwidth_packed = self.output_bitwidth_unpacked * self.output_item_size


    def get_input_shapes(self):
        edge = self.lrg.routing_edges[self.lrg.logic_nodes[0].output_tids[0]]
        streamed_shape = edge.to_compute_shapes[1]
        real_shape = (1,) + self.model.input.shape[1:]
        return real_shape, streamed_shape

    def tensor_to_sample_vectors(self, x: tf.Tensor, dim: int, kifs, debug=False) -> list[list[float]]:
        if debug:
            print(f"\ntensor_to_sample_vectors\n x: {x}, dim: {dim}, kifs {kifs} \n")
               
        flat = tf.reshape(x, [-1])
        # reshape into rows of dim
        reshaped = tf.reshape(flat, [-1, dim])
        # convert to python list of lists
        reshaped_list = reshaped.numpy().tolist()
        streamed_inputs = []
        total_bits = sum(np.max(arr) for arr in kifs)
        for v in reshaped_list:
            hex_vector = []
            for idx, e in enumerate(v):
                k = kifs[0][idx]
                i = kifs[1][idx]
                f = kifs[2][idx]
                streamed_scalar = encode_to_d(
                    np.array([e]),
                    k,
                    i,
                    f,
                    as_hex=False,
                    packed=False,
                )
                if debug:
                    print(f"Value {e} encoded to hex: {streamed_scalar} with type {type(streamed_scalar)}, bin = {hex_to_bin(streamed_scalar, total_bits)}")
                hex_vector.append(streamed_scalar)
            if debug:
                print(f"Vector {v} encoded to hex: {hex_vector}")
            streamed_inputs.append(hex_vector)
        if debug:
            print(f"Input tensor: {x} encoded to hex: {streamed_inputs}, with kif {kifs}")
        return streamed_inputs

    def generate_stimulus(self) -> tuple[tf.Tensor, list[list[int]]]:
        real_shape, streamed_shape = self.get_input_shapes()
        real_input = tf.ones(shape=real_shape, dtype=tf.float32) 
        packets = self.tensor_to_sample_vectors(real_input, math.prod(streamed_shape), self.input_kif, debug=True) # shape of 0 is just 1 as our convention
        return real_input, packets # maybe can use schedule from the router or something instead of hardcoding this reshape
    
    def get_intermediate_values(self):
        x, packets = self.generate_stimulus()
        intermediate_model = tf.keras.Model(inputs=self.model.input, outputs=[layer.output for layer in self.model.layers])
        intermediate_outputs = intermediate_model(x)
        out = []
        for i, output in enumerate(intermediate_outputs): 
            if i == 0 :
                continue
            as_rtl = self.tensor_to_sample_vectors(output, output.shape[-1], self.output_kif)
            # print(f"Intermediate output of layer {i} ({self.model.layers[i].name}): {output.numpy()}")
            print(f"Expected intermediate output of layer {i} ({self.model.layers[i].name}), real {output.numpy()}, as hex: {as_rtl}")
            out.append((output, as_rtl))
        return out
    
    def postprocess_golden(self, y: tf.Tensor) -> np.ndarray:
        print(f"Postprocessing golden output tensor {y} with kif {self.output_kif}")
        y_vecs = self.tensor_to_sample_vectors(y, self.output_item_size, self.output_kif, debug=True)
        y_packed = [self.pack_lsb_first(vec, self.output_bitwidth_unpacked) for vec in y_vecs]
        print(f"Postprocessed golden output vectors: real {y}, dut_exp {y_vecs}, packed: {y_packed}")
        return np.array(y_vecs, dtype=np.int64)


    # ============================================================
    # Packing / Unpacking helpers
    # ============================================================
    def pack_lsb_first(self, elems, elem_width: Optional[int] = None) -> int:
        """elem0 in LSB, elem1 next, ..."""
        if elem_width is None:
            elem_width = int(self.input_bitwidth_unpacked)
        mask = (1 << elem_width) - 1
        out = 0
        for i, x in enumerate(elems):
            out |= (int(x) & mask) << (i * elem_width)
        return out

    def unpack_lsb_first(self, word: int, n_elems: Optional[int] = None, elem_width: Optional[int] = None):
        if elem_width is None:
            elem_width = self.output_bitwidth_unpacked
        if n_elems is None:
            n_elems = self.output_item_size
        mask = (1 << elem_width) - 1
        return [(word >> (i * elem_width)) & mask for i in range(n_elems)]

def get_lrg():
    model = build_golden_model()
    return build_lr_graph_from_model(model)


# ============================================================
# DUT helpers
# ============================================================
async def reset(dut, cycles: int):
    dut.rst.value = 1
    dut.data_in.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

async def print_qsum(dut):
    # dut._log.info(f"accum_count = {}")
    dut._log.info(f"data_in: {dut.data_in.value}, in_valid: {dut.in_valid.value}, out_valid: {dut.out_valid.value}, data_out: {dut.data_out.value}")
    


async def stream_vectors_and_capture(tb, dut, vectors):
    """
    Stream T rows into data_in (1 row per cycle), then flush.
    Capture *every* data_out cycle as unpacked rows.
    """
    in_words = [tb.pack_lsb_first(row) for row in vectors]
    print(f"Streaming in words {vectors}, packed words: {in_words}, len(vectors)={len(vectors)} shapes: {[len(row) for row in vectors]}")
    print(f"Packed bitvectors in binary: {[hex_to_bin(w, tb.input_bitwidth_packed) for w in in_words]}")
    captured_rows = []
    captured_packed = []

    total_cycles = len(in_words) + PARAMS["FLUSH_CYCLES"]

    for i in range(total_cycles):
        dut.data_in.value = int(in_words[i]) if i < len(in_words) else 0
        dut.in_valid.value = 1 if i < len(in_words) else 0
        dut.out_ready.value = 1  # always ready to receive output
        print(f"Cycle {i}: data_in={dut.data_in.value} valid={dut.in_valid.value} ready={dut.out_ready.value}")
        await RisingEdge(dut.clk)

        w = int(dut.data_out.value)
        if dut.out_valid.value == 1:
            print(f"Captured output word: {w}, binary: {hex_to_bin(w, tb.output_bitwidth_packed)}")
            print(f"Unpacked output elems: {tb.unpack_lsb_first(w)}")
            print(f"DEBUG: dut.data_out.value: {dut.data_out.value}")
            captured_packed.append(w)
            captured_rows.append(tb.unpack_lsb_first(w))

    return np.array(captured_rows, dtype=np.int64), captured_packed


def extract_T_rows_from_stream(captured_rows: np.ndarray, T: int) -> np.ndarray:
    """
    Convert the long captured stream into exactly T output rows for comparison.
    """
    mode = PARAMS["ALIGN_MODE"]
    if mode == "discard":
        d = int(PARAMS["DISCARD_OUT_CYCLES"])
        return captured_rows[d : d + T]
    if mode == "take_last":
        return captured_rows[-T:]
    raise ValueError(f"Unknown ALIGN_MODE={mode!r}, expected 'discard' or 'take_last'")


# ============================================================
# The test
# ============================================================
# tb = TestTopSimpleTB(model=build_golden_model())
# @cocotb.test()
async def print_rtl_vs_keras_final_outputs_simple_qsum(dut):
    # Clock
    cocotb.start_soon(Clock(dut.clk, PARAMS["CLK_PERIOD_NS"], unit="ns").start())

    # Reset
    await reset(dut, PARAMS["RESET_CYCLES"])

    real_inp, vectors = tb.generate_stimulus()
    T = len(vectors)

    # --- Run golden Keras once on full tensor ---
    # vectors has shape (T, IN_ELEMS), model expects (batch, IN_ELEMS)
    model = tb.model
    y = model(real_inp, training=False)
    expected_output = tb.postprocess_golden(y.numpy())
    dut._log.info(f"Generated expected outputs {expected_output}")
    # Convert to numpy
    try:
        y_np = y.numpy()
    except Exception:
        y_np = np.asarray(y)

    # --- Run RTL streaming + capture ---
    captured_rows, _captured_packed = await stream_vectors_and_capture(tb, dut, vectors)

    dut._log.info(f"Captured rows {captured_rows}\npacked: {_captured_packed}")
    
    rtl_T_rows = extract_T_rows_from_stream(captured_rows, T)

    # --- Print final outputs only ---
    dut._log.info("=== INPUT (T x IN_ELEMS) ===")
    dut._log.info("\n" + str(vectors))

    dut._log.info("=== KERAS OUTPUT at each layer ===")
    int_values = tb.get_intermediate_values()
    for i, v in enumerate(int_values):
        dut._log.info(f"Layer {i}: shape={v[0].shape}, values_real={v[0]}, rtl vals = {v[1]}")
    
    dut._log.info("=== KERAS OUTPUT (T x OUT_ELEMS) ===")
    dut._log.info("\n" + str(y_np))

    dut._log.info("=== KERAS OUTPUT as bitvectors ===")
    dut._log.info("\n" + "\n".join(str(x) for x in expected_output.flatten()))

    dut._log.info("=== RTL OUTPUT (T x OUT_ELEMS) extracted from stream ===")
    dut._log.info("\n" + str(rtl_T_rows))


tb1 = TestTopSimpleTB(model=build_golden_model(model=m_with_qsum()))
@cocotb.test()
async def print_rtl_vs_keras_final_outputs_m_with_qsum(dut):
    # Clock
    cocotb.start_soon(Clock(dut.clk, PARAMS["CLK_PERIOD_NS"], unit="ns").start())
    # Reset
    await reset(dut, PARAMS["RESET_CYCLES"])

    real_inp, vectors = tb1.generate_stimulus()
    dut._log.info(f"Generated stimulus vectors: {vectors}, with real input tensor {real_inp}")
    T = len(vectors)

    # --- Run golden Keras once on full tensor ---
    model = tb1.model
    y = model(real_inp, training=False)
    expected_output = tb1.postprocess_golden(y.numpy())
    dut._log.info(f"Generated expected outputs {expected_output}")
    # Convert to numpy
    try:
        y_np = y.numpy()
    except Exception:
        y_np = np.asarray(y)

    # --- Run RTL streaming + capture ---
    captured_rows, _captured_packed = await stream_vectors_and_capture(tb1, dut, vectors)

    dut._log.info(f"Captured rows {captured_rows}\npacked: {_captured_packed}")
    
    rtl_T_rows = extract_T_rows_from_stream(captured_rows, T)

    # --- Print final outputs only ---
    dut._log.info("=== INPUT (T x IN_ELEMS) ===")
    dut._log.info("\n" + str(vectors))

    dut._log.info("=== KERAS OUTPUT at each layer ===")
    int_values = tb1.get_intermediate_values()
    for i, v in enumerate(int_values):
        dut._log.info(f"Layer {i}: shape={v[0].shape}, values_real={v[0]}, expected rtl vals = {v[1]}")
    
    dut._log.info("=== KERAS OUTPUT (T x OUT_ELEMS) ===")
    dut._log.info("\n" + str(y_np))

    dut._log.info("=== KERAS OUTPUT as bitvectors ===")
    dut._log.info("\n" + "\n".join(str(x) for x in expected_output.flatten()))

    dut._log.info("=== RTL OUTPUT (T x OUT_ELEMS) extracted from stream ===")
    dut._log.info("\n" + str(rtl_T_rows))
