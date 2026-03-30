import os
from os import getenv
from pathlib import Path
from cocotb_tools.runner import get_runner

from da4ml.graph_ir.lr_graph import lr_graph_to_hardware
from qsum_definitions import m_with_qsum, qsum_lrg, generate_qsum_hw, simple_qsum
import shutil


PROJ_ROOT = Path(__file__).resolve().parent

os.environ["PYTHONPATH"] = str(PROJ_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

HERE = Path(__file__).resolve().parent
RTL_DIR = HERE / "rtl"
TESTS_DIR = HERE / "tests"


def main():
    sim = getenv("SIM", "verilator")
    model = m_with_qsum()
    # model = simple_qsum()
    # if RTL_DIR.exists():
    #     shutil.rmtree(RTL_DIR)
    # RTL_DIR.mkdir(parents=True, exist_ok=True)
    
    lines_written = lr_graph_to_hardware(qsum_lrg(model), RTL_DIR, debug=False)
    verilog_sources = sorted(map(str, RTL_DIR.rglob("*.v")))
    verilog_sources += sorted(map(str, RTL_DIR.rglob("*.sv")))

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
        extra_env= {
            "PYTHONPATH": str(PROJ_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
    )


if __name__ == "__main__":
    main()