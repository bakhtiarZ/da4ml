`timescale 1ns/1ps

module mod_op_2__QDense (
    input [23:0] model_inp,
    output [8:0] model_out
);

    // verilator lint_off UNUSEDSIGNAL
    // Explicit quantization operation will drop bits if exists

    wire [7:0] v0; assign v0[7:0] = model_inp[7:0]; // 0.0
    wire [7:0] v1; assign v1[7:0] = model_inp[15:8]; // 0.0
    wire [7:0] v2; assign v2[7:0] = model_inp[23:16]; // 0.0
    wire [7:0] v3; assign v3[7:0] = v2[7:0]; // 0.0
    wire [8:0] v4; shift_adder #(8, 1, 1, 0, 9, 0, 0) op_4 (v3[7:0], 1'b1, v4[8:0]); // 0.0
    wire [6:0] v5; assign v5[6:0] = v4[7:1]; // 0.0
    wire [7:0] v6; assign v6[7:0] = v0[7:0]; // 0.0
    wire [8:0] v7; shift_adder #(8, 1, 1, 0, 9, 0, 0) op_7 (v6[7:0], 1'b1, v7[8:0]); // 0.0
    wire [6:0] v8; assign v8[6:0] = v7[7:1]; // 0.0
    wire [7:0] v9; assign v9[7:0] = v1[7:0]; // 0.0
    wire [8:0] v10; shift_adder #(8, 1, 1, 0, 9, 0, 0) op_10 (v9[7:0], 1'b1, v10[8:0]); // 0.0
    wire [6:0] v11; assign v11[6:0] = v10[7:1]; // 0.0
    wire [7:0] v12; shift_adder #(7, 7, 1, 1, 8, 0, 0) op_12 (v8[6:0], v11[6:0], v12[7:0]); // 1.0
    wire [8:0] v13; shift_adder #(7, 8, 1, 1, 9, 0, 0) op_13 (v5[6:0], v12[7:0], v13[8:0]); // 2.0

    // verilator lint_on UNUSEDSIGNAL

    assign model_out[8:0] = v13[8:0];

    endmodule
