import os
from os import getenv
from pathlib import Path
from cocotb_tools.runner import get_runner

from da4ml.graph_ir.lr_graph import build_lr_graph_from_model, lr_graph_to_hardware
from qsum_definitions import config, m_with_qsum, m_with_qsum_fixed_q_conf, simple_qsum, m_testing_parallelism
import shutil


PROJ_ROOT = Path(__file__).resolve().parent

os.environ["PYTHONPATH"] = str(PROJ_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

HERE = Path(__file__).resolve().parent
RTL_DIR = HERE / "rtl"
TESTS_DIR = HERE / "tests"


def main():
    sim = getenv("SIM", "verilator")
    # model = m_with_qsum()
    model = m_with_qsum_fixed_q_conf()
    # model = simple_qsum()
    # if RTL_DIR.exists():
    #     shutil.rmtree(RTL_DIR)
    # RTL_DIR.mkdir(parents=True, exist_ok=True)
    lrg = build_lr_graph_from_model(model, parallelism=config().get("PARALLELISM", 1))
    lines_written = lr_graph_to_hardware(lrg, RTL_DIR, debug=False)
    verilog_sources = sorted(map(str, RTL_DIR.rglob("*.v")))
    verilog_sources += sorted(map(str, RTL_DIR.rglob("*.sv")))

    runner = get_runner(sim)

    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="top_module",
        build_args=[
            "--trace",
            "--trace-structs",
            "-O0",
        ],
        waves=True,
        always=True,
    )

    runner.test(
        hdl_toplevel="top_module",
        test_module="test_qsum",
        test_dir=str(TESTS_DIR),
        waves=True,
        extra_env= {
            "PYTHONPATH": str(PROJ_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
    )


if __name__ == "__main__":
    main()