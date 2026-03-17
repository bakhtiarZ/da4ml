from os import getenv
from pathlib import Path
from cocotb_tools.runner import get_runner
import keras

from hgq.layers import QDense, QAdd, QSum
from hgq.config import QuantizerConfig

from da4ml_test_utils.src.da4ml_test_utils.graph_ir.test_lrgraph import test_build_lrgraph_from_model


HERE = Path(__file__).resolve().parent
RTL_DIR = HERE / "rtl"
TESTS_DIR = HERE / "tests"


def main():
    sim = getenv("SIM", "verilator")
    test_qsum_gen()
    verilog_sources = sorted(map(str, RTL_DIR.rglob("*.v")))
    verilog_sources += sorted(map(str, RTL_DIR.rglob("*.sv")))
    print("Verilog sources:", verilog_sources, "\n")  # debug

    runner = get_runner(sim)

    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="QSum",
        build_args=[
            "--trace",
            "--trace-structs",
            "-O0",
            "-build-jobs", "8",
        ],
        waves=True,
        always=True,
    )

    runner.test(
        hdl_toplevel="QSum",
        test_module="test_qsum",
        test_dir=str(TESTS_DIR),
        waves=True,
    )


if __name__ == "__main__":
    main()