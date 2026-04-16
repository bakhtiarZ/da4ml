`timescale 1 ns / 1 ps

module mod_op_1__QDense_wrapper (
    // verilator lint_off UNUSEDSIGNAL
    input [1151:0] model_inp,
    // verilator lint_on UNUSEDSIGNAL
    output [511:0] model_out
);
    wire [1151:0] packed_inp;
    wire [511:0] packed_out;

    assign packed_inp[1151:0] = model_inp[1151:0];

    mod_op_1__QDense op (
        .model_inp(packed_inp),
        .model_out(packed_out)
    );

    assign model_out[511:0] = packed_out[511:0];

endmodule
