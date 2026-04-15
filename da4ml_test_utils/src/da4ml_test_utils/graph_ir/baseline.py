from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import keras
from hgq.config import QuantizerConfig
from hgq.layers import QDense, QSum

from da4ml.codegen.rtl.rtl_model import RTLModel
from da4ml.converter import trace_model
from da4ml.converter import trace_model
from da4ml.graph_ir.lr_graph import build_lr_graph_from_model, lr_graph_to_hardware
from da4ml.trace.tracer import comb_trace


BASE_DIR = Path("/homes/bm920/workspace/da4ml/synthesis_results")
SCRIPT_DIR = BASE_DIR
DEFAULT_TCL_SCRIPT = SCRIPT_DIR / "run_synth.tcl"


def m_with_qsum_fixed_q_conf_8_bit(q: QuantizerConfig) -> keras.Model:
    i = keras.Input((64, 2), name="inp")
    d0 = QDense(
        1,
        iq_conf=q,
        oq_conf=q,
        kernel_initializer="ones",
        bias_initializer="zeros",
        enable_iq=True,
        enable_oq=True,
    )(i)
    s = QSum(
        iq_conf=q,
        axes=1,
        scale=1,
        keepdims=True,
        enable_iq=True,
    )(d0)
    return keras.Model(i, s, name="qsum_fixed_q")


def setup_test_model_and_config() -> tuple[keras.Model, list[int], int]:
    q = QuantizerConfig(heterogeneous_axis=(), k0=1, i0=4, f0=3)
    m = m_with_qsum_fixed_q_conf_8_bit(q)

    parallelisms = [1, 2, 4, 8, 16, 32, 64]
    total_bitwidth = q.config["k0"] + q.config["i0"] + q.config["f0"]

    return m, parallelisms, total_bitwidth


def load_summary_json(report_path: str | Path) -> dict[str, Any]:
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {report_path}")
    return json.loads(report_path.read_text())


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    # Top-level simple fields
    for key in ("part", "top"):
        if key in summary:
            flat[key] = summary[key]

    # Clock block
    clock = summary.get("clock", {})
    flat["clock_port"] = clock.get("port")
    flat["target_period_ns"] = clock.get("target_period_ns")

    # Utilization block
    util = summary.get("utilization", {})
    flat["lut"] = util.get("lut")
    flat["ff"] = util.get("ff")
    flat["bram36"] = util.get("bram36")
    flat["bram18"] = util.get("bram18")
    flat["uram"] = util.get("uram")
    flat["dsp"] = util.get("dsp")

    # Timing block
    timing = summary.get("timing", {})
    flat["wns_ns"] = timing.get("wns_ns")
    flat["tns_ns"] = timing.get("tns_ns")
    flat["whs_ns"] = timing.get("whs_ns")
    flat["ths_ns"] = timing.get("ths_ns")
    flat["achieved_period_ns"] = timing.get("achieved_period_ns")
    flat["estimated_fmax_mhz"] = timing.get("estimated_fmax_mhz")

    return flat


def run_vivado_synth(
    top: str,
    part: str,
    rtl_dir: str | Path,
    out_dir: str | Path,
    vivado_bin: str = "vivado",
    tcl_script: str | Path = DEFAULT_TCL_SCRIPT,
    clk_port: str | None = None,
    clk_period_ns: float | None = None,
) -> subprocess.CompletedProcess[str]:
    rtl_dir = Path(rtl_dir).resolve()
    out_dir = Path(out_dir).resolve()
    tcl_script = Path(tcl_script).resolve()

    if not rtl_dir.exists():
        raise FileNotFoundError(f"RTL directory does not exist: {rtl_dir}")

    if not tcl_script.exists():
        raise FileNotFoundError(f"Tcl script does not exist: {tcl_script}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        vivado_bin,
        "-mode",
        "batch",
        "-nolog",
        "-nojournal",
        "-source",
        str(tcl_script),
        "-tclargs",
        top,
        part,
        str(rtl_dir),
        str(out_dir),
        clk_port or "",
        str(clk_period_ns) if clk_period_ns is not None else "",
    ]

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )

    (out_dir / "vivado_stdout.log").write_text(result.stdout)
    (out_dir / "vivado_stderr.log").write_text(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Vivado failed with exit code {result.returncode}\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"stdout:\n{result.stdout}\n\n"
            f"stderr:\n{result.stderr}"
        )

    return result


def synth_and_collect(
    top: str,
    part: str,
    rtl_dir: str | Path,
    out_dir: str | Path,
    tcl_script: str | Path = DEFAULT_TCL_SCRIPT,
    clk_port: str | None = None,
    clk_period_ns: float | None = None,
) -> dict[str, Any]:
    run_vivado_synth(
        top=top,
        part=part,
        rtl_dir=rtl_dir,
        out_dir=out_dir,
        tcl_script=tcl_script,
        clk_port=clk_port,
        clk_period_ns=clk_period_ns,
    )

    summary_json = Path(out_dir) / "summary.json"
    summary = load_summary_json(summary_json)
    return flatten_summary(summary)


def generate_designs_and_collect_results(
    model,
    config: dict[str, Any],
    rtl_dir: str | Path,
    part: str,
    top: str = "top_module",
    tcl_script: str | Path = DEFAULT_TCL_SCRIPT,
    clk_port: str | None = "clk",
    clk_period_ns: float | None = 5.0,
) -> list[dict[str, Any]]:
    rtl_dir = Path(rtl_dir).resolve()
    rtl_dir.mkdir(parents=True, exist_ok=True)
    
    results: list[dict[str, Any]] = []
    design_name = "baseline"
    conf_dir = rtl_dir / design_name
    project_rtl_dir = conf_dir / "rtl"
    out_dir = conf_dir / "results"
    
    inp, out = trace_model(model, verbose=True) 
    comb_logic = comb_trace(inp, out)
        
    rtl_model = RTLModel(comb_logic, 'vmodel', project_rtl_dir, flavor='verilog', latency_cutoff=5)
    rtl_model.write()
        
    
    summary = synth_and_collect(
        top=top,
        part=part,
        rtl_dir=project_rtl_dir,
        out_dir=out_dir,
        tcl_script=tcl_script,
        clk_port=clk_port,
        clk_period_ns=clk_period_ns,
    )

    row = {
        "model": design_name,
        "parallelism": -1,
        "total_bitwidth": config.get("TOTAL_BITWIDTH"),
        "design_name": design_name,
        "part_requested": part,
        "top_requested": top,
        "rtl_dir": str(project_rtl_dir),
        "out_dir": str(out_dir),
        "lines_written": -1,
        "status": "ok",
        **summary,
    }
    results.append(row)

                

    return results


def write_results_csv(results: list[dict[str, Any]], csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({k for row in results for k in row.keys()})

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    model, parallelisms, total_bitwidth = setup_test_model_and_config()

    config = {
        "PARALLELISM": parallelisms,
        "TOTAL_BITWIDTH": total_bitwidth,
    }

    results = generate_designs_and_collect_results(
        model = model,
        config=config,
        rtl_dir=BASE_DIR,
        part="xcvu9p-flgb2104-2-i",
        top="vmodel",
        tcl_script=DEFAULT_TCL_SCRIPT,
        clk_port="clk",
        clk_period_ns=5.0,
    )

    csv_path = BASE_DIR / "summary.csv"
    write_results_csv(results, csv_path)

    print("\nFinished.")
    print(f"Summary written to: {csv_path}")
    for row in results:
        print(row)