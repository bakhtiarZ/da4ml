`timescale 1 ns / 1 ps

module mod_op_2__QDense_wrapper (
    // verilator lint_off UNUSEDSIGNAL
    input [23:0] model_inp,
    // verilator lint_on UNUSEDSIGNAL
    output [8:0] model_out
);
    wire [23:0] packed_inp;
    wire [8:0] packed_out;

    assign packed_inp[23:0] = model_inp[23:0];

    mod_op_2__QDense op (
        .model_inp(packed_inp),
        .model_out(packed_out)
    );

    assign model_out[8:0] = packed_out[8:0];

endmodule
