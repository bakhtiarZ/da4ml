from pathlib import Path
import shutil
from cocotb_test.simulator import run
from da4ml_test_utils.graph_ir.test_lrgraph import two_layer_model, make_next_numbered_dir

RTL_DIR = Path(__file__).parent / "rtl"
TESTS_DIR = Path(__file__).parent / "tests"

def create_verilog_sources(dir):
    from da4ml.graph_ir.lr_graph import build_lr_graph_from_model, lr_graph_to_hardware
    from da4ml.graph_ir.scheduling import DataSchedule, DataScheduler, dense_schedule, input_dense_schedule_requirement, minimum_output_shape_for_dense, minimum_input_shape_for_dense, dense_rebuilder
    from hgq.layers import QDense
    from hgq.config import QuantizerConfig
    import keras

    # Build the LR graph from the Keras model
    model = two_layer_model()
    lr_g = build_lr_graph_from_model(model)
    project_dir = make_next_numbered_dir('/homes/bm920/workspace/da4ml/.tmp/lr_graph_rtl_projects/', prefix='project_')
    # Generate RTL code for the LR graph and write to file
    lines_written = lr_graph_to_hardware(lr_g, project_dir, debug=True)
    shutil.copytree(project_dir/f"src", dir, dirs_exist_ok=True)
    shutil.copy(project_dir/f"top_module.sv", dir)

def main():

    create_verilog_sources(RTL_DIR)
    verilog_sources = sorted(map(str, RTL_DIR.rglob("*.v")))
    verilog_sources += sorted(map(str, RTL_DIR.rglob("*.sv")))
    print("Verilog sources:", verilog_sources, "\n")  # debug
    run(
        simulator="verilator",          
        verilog_sources=verilog_sources,
        toplevel="top_module",
        module="test_top_simple", # name of python test module (without .py)
        # extra_env={
        #     "TEST_VECTOR_LEN": "128",
        #     "PIPELINE_LATENCY": "3",
        #     "MODEL_PATH": "models/golden.keras",
        # }
        # extra compile flags if needed:
        compile_args=["--sv", "--timing"],
        # waves=True,  # supported by some sims; otherwise pass sim_args
    )

if __name__ == "__main__":
    main()