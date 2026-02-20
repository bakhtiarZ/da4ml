import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

import numpy as np
import tensorflow as tf

# ============================================================
# Edit these directly
# ============================================================
PARAMS = {
    # clock/reset
    "CLK_PERIOD_NS": 10,
    "RESET_CYCLES": 5,

    # Input packing: one "row" per cycle, row has IN_ELEMS elements each IN_ELEM_W bits
    "IN_ELEMS": 2,
    "IN_ELEM_W": 8,

    # Output unpacking: each cycle produces OUT_ELEMS elements each OUT_ELEM_W bits
    "OUT_ELEMS": 1,
    "OUT_ELEM_W": 9,

    # Your input tensor (T rows). Example: (T=2, IN_ELEMS=2) => [[..,..],[..,..]]
    "VECTORS": [
        [2, 2],
        [7, 1],
    ],

    # How to align streamed RTL output to Keras output
    # Option A (recommended): discard first N output cycles (pipeline latency), then take next T rows
    "ALIGN_MODE": "discard",     # "discard" or "take_last"
    "DISCARD_OUT_CYCLES": 3,     # set this to your pipeline latency in cycles

    # Flush cycles after last input is sent (give pipeline time to finish)
    "FLUSH_CYCLES": 10,
}


# ============================================================
# Define your golden model here (EDIT THIS)
# ============================================================
def build_golden_model():
    """
    Return a Keras model that implements the SAME network functionally.
    You will edit this.

    Tips:
      - Keep input shape = (IN_ELEMS,) so feeding (T, IN_ELEMS) works.
      - Load weights however you like (set_weights, assign, etc.)
      - Apply quantization in the model or in postprocess_golden().
    """
    from tests.graph_ir.test_lrgraph import two_layer_model

    return two_layer_model()


def get_lrg():
    from da4ml.graph_ir.lr_graph import build_lr_graph_from_model
    model = build_golden_model()
    return build_lr_graph_from_model(model)

def postprocess_golden(y_np: np.ndarray) -> np.ndarray:
    """
    Optional: convert Keras output to what you want to *visually compare* against RTL.

    RTL output is typically quantized integers; Keras may give floats.
    You can edit this to match your quantization (rounding/saturation/sign).

    Must return shape (T, OUT_ELEMS).
    """
    # Default: just return floats
    return y_np


# ============================================================
# Packing / Unpacking helpers
# ============================================================
def pack_lsb_first(elems, elem_width: int) -> int:
    """elem0 in LSB, elem1 next, ..."""
    mask = (1 << elem_width) - 1
    out = 0
    for i, x in enumerate(elems):
        out |= (int(x) & mask) << (i * elem_width)
    return out


def unpack_lsb_first(word: int, n_elems: int, elem_width: int):
    mask = (1 << elem_width) - 1
    return [(word >> (i * elem_width)) & mask for i in range(n_elems)]


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


async def stream_vectors_and_capture(dut, vectors):
    """
    Stream T rows into data_in (1 row per cycle), then flush.
    Capture *every* data_out cycle as unpacked rows.
    """
    in_words = [pack_lsb_first(row, PARAMS["IN_ELEM_W"]) for row in vectors]

    captured_rows = []
    captured_packed = []

    total_cycles = len(in_words) + PARAMS["FLUSH_CYCLES"]

    for i in range(total_cycles):
        dut.data_in.value = in_words[i] if i < len(in_words) else 0
        await RisingEdge(dut.clk)

        w = int(dut.data_out.value)
        captured_packed.append(w)
        captured_rows.append(unpack_lsb_first(w, PARAMS["OUT_ELEMS"], PARAMS["OUT_ELEM_W"]))

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


def gen_input():
    real_input = tf.tensor(
        [[2, 2], [7, 1]],
        dtype=tf.float32,
    )
    lrg = get_lrg()
    edge = lrg.routing_edges[lrg.logic_nodes[0].output_tids[0]]
    streamed_shape = edge.to_compute_shapes[1]
    streamed_input = tf.reshape(real_input, streamed_shape)
    print(f"Real input: {real_input}")
    print(f"Streamed input shape: {streamed_shape}")
    print(f"Streamed input: {streamed_input}")
    return streamed_input # maybe can use schedule from the router or something instead of hardcoding this reshape

# ============================================================
# The test
# ============================================================
@cocotb.test()
async def print_rtl_vs_keras_final_outputs(dut):
    # Clock
    cocotb.start_soon(Clock(dut.clk, PARAMS["CLK_PERIOD_NS"], units="ns").start())

    # Reset
    await reset(dut, PARAMS["RESET_CYCLES"])

    vectors = np.asarray(PARAMS["VECTORS"], dtype=np.int64)
    T = vectors.shape[0]

    # --- Run golden Keras once on full tensor ---
    # vectors has shape (T, IN_ELEMS), model expects (batch, IN_ELEMS)
    model = build_golden_model()
    y = model(vectors.astype(np.float32), training=False)

    # Convert to numpy
    try:
        y_np = y.numpy()
    except Exception:
        y_np = np.asarray(y)

    # Normalize shape to (T, OUT_ELEMS)
    y_np = np.asarray(y_np)
    if y_np.ndim == 1:
        y_np = y_np.reshape(T, 1)
    y_np = postprocess_golden(y_np)

    # --- Run RTL streaming + capture ---
    captured_rows, _captured_packed = await stream_vectors_and_capture(dut, PARAMS["VECTORS"])

    dut._log.info(f"Captured rows {captured_rows}\npacked: {_captured_packed}")
    
    rtl_T_rows = extract_T_rows_from_stream(captured_rows, T)

    # --- Print final outputs only ---
    dut._log.info("=== INPUT (T x IN_ELEMS) ===")
    dut._log.info("\n" + str(vectors))

    dut._log.info("=== KERAS OUTPUT (T x OUT_ELEMS) ===")
    dut._log.info("\n" + str(y_np))

    dut._log.info("=== RTL OUTPUT (T x OUT_ELEMS) extracted from stream ===")
    dut._log.info("\n" + str(rtl_T_rows))
