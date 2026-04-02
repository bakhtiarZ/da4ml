`timescale 1ns/1ps

module mod_op_1__QDense (
    input [15:0] model_inp,
    output [15:0] model_out
);

    // verilator lint_off UNUSEDSIGNAL
    // Explicit quantization operation will drop bits if exists

    wire [7:0] v0; assign v0[7:0] = model_inp[7:0]; // 0.0
    wire [7:0] v1; assign v1[7:0] = model_inp[15:8]; // 0.0
    wire [7:0] v2; assign v2[7:0] = v0[7:0]; // 0.0
    wire [8:0] v3; shift_adder #(8, 1, 1, 0, 9, 0, 0) op_3 (v2[7:0], 1'b1, v3[8:0]); // 0.0
    wire [6:0] v4; assign v4[6:0] = v3[7:1]; // 0.0
    wire [7:0] v5; assign v5[7:0] = v1[7:0]; // 0.0
    wire [8:0] v6; shift_adder #(8, 1, 1, 0, 9, 0, 0) op_6 (v5[7:0], 1'b1, v6[8:0]); // 0.0
    wire [6:0] v7; assign v7[6:0] = v6[7:1]; // 0.0
    wire [7:0] v8; shift_adder #(7, 7, 1, 1, 8, 0, 0) op_8 (v4[6:0], v7[6:0], v8[7:0]); // 1.0

    // verilator lint_on UNUSEDSIGNAL

    assign model_out[7:0] = v8[7:0];
    assign model_out[15:8] = v8[7:0];

    endmodule
