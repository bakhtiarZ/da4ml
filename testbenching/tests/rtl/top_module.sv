module top_module (
        input logic clk, 
        input logic rst, 
        input logic [15:0] data_in, 
        output logic [8:0] data_out 
    );
    
logic [15:0] inp_to_op_1;
logic [23:0] out_from_op_1;
mod_op_1__QDense op_1__QDense (.model_inp(inp_to_op_1), .model_out(out_from_op_1));
assign inp_to_op_1 = data_in;
logic [23:0] edge_to_op_2_from_output_buffer_020544;
fifo_packed #(DEPTH(1), DATA_WIDTH(8), DATA_SIZE(3)) buffer_020544 (.clk(clk), .rst(rst), .data_in(out_from_op_1), .data_out(edge_to_op_2_from_output_buffer_020544));
logic [23:0] inp_to_op_2;
logic [8:0] out_from_op_2;
mod_op_2__QDense op_2__QDense (.model_inp(inp_to_op_2), .model_out(out_from_op_2));
assign inp_to_op_2 = edge_to_op_2_from_output_buffer_020544;
assign data_out = out_from_op_2;

endmodule