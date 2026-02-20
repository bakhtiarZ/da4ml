TOPLEVEL_LANG = verilog
SIM ?= verilator

TOPLEVEL = top_module
MODULE = test_top_module

VERILOG_SOURCES += $(wildcard $(PWD)/rtl/*.sv)
VERILOG_SOURCES += $(wildcard $(PWD)/rtl/static/*.sv)

# Verilator options for SystemVerilog
VERILATOR_ARGS += --sv --timing

include $(shell cocotb-config --makefiles)/Makefile.sim
