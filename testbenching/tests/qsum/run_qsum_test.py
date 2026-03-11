from os import getenv
from pathlib import Path
from cocotb_tools.runner import get_runner


HERE = Path(__file__).resolve().parent
RTL_DIR = HERE / "rtl"
TESTS_DIR = HERE / "tests"


def main():
    sim = getenv("SIM", "verilator")

    verilog_sources = [
        str(RTL_DIR / "qsum.sv"),
    ]

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