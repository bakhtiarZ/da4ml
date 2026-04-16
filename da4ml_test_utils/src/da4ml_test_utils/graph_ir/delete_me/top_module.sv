module top_module (
        input logic clk, 
        input logic rst, 
        input logic [1151:0] data_in, 
        input logic data_in_valid,
        input logic data_out_ready,
        output logic [13:0] data_out,
        output logic data_out_valid,
        output logic data_in_ready 
    );
    
// Logic node 1 for operation <QDense name=q_dense, built=True>
logic [1151:0] inp_to_op_1;
logic in_valid_to_1;
logic out_ready_to_1;
logic [511:0] out_from_op_1;
logic out_valid_from_1;
logic in_ready_from_1;
assign out_valid_from_1 = in_valid_to_1; // passthrough valid
assign in_ready_from_1 = out_ready_to_1; // passthrough ready
mod_op_1__QDense_wrapper op_1__QDense (.model_inp(inp_to_op_1), .model_out(out_from_op_1));
// End of logic node 1

// Connecting intermediate signals of node 1 to previous intermediate signals
assign data_in_ready = in_ready_from_1;
assign inp_to_op_1 = data_in;
assign in_valid_to_1 = data_in_valid;
// End of connections for node 1

// Buffer for edge tid=139619976780192 with routing logic RoutingLogic(buffer_type='fifo', buffer_shape=(1, 64))
logic [511:0] inp_to_buffer_edge__from_op1_to_op2;
logic in_valid_to_buffer_edge__from_op1_to_op2;
logic out_ready_to_buffer_edge__from_op1_to_op2;
logic [511:0] out_from_buffer_edge__from_op1_to_op2;
logic out_valid_from_buffer_edge__from_op1_to_op2;
logic in_ready_from_buffer_edge__from_op1_to_op2;
fifo_rv #(.DEPTH(1), .DATA_WIDTH(512)) buffer_edge__from_op1_to_op2 (.clk(clk), .rst(rst), .in_data(inp_to_buffer_edge__from_op1_to_op2), .in_valid(in_valid_to_buffer_edge__from_op1_to_op2), .out_ready(out_ready_to_buffer_edge__from_op1_to_op2), .out_data(out_from_buffer_edge__from_op1_to_op2), .out_valid(out_valid_from_buffer_edge__from_op1_to_op2), .in_ready(in_ready_from_buffer_edge__from_op1_to_op2), /* verilator lint_off PINCONNECTEMPTY */ .full(), .empty(), .count() /* verilator lint_on PINCONNECTEMPTY */);
// End of buffer for edge tid=139619976780192

// Connecting buffer for edge tid=139619976780192 to logic node 1
assign out_ready_to_1 = in_ready_from_buffer_edge__from_op1_to_op2;
assign inp_to_buffer_edge__from_op1_to_op2 = out_from_op_1;
assign in_valid_to_buffer_edge__from_op1_to_op2 = out_valid_from_1;
// End of connections for buffer for edge tid=139619976780192

// **** Instance of QSum for node 2 ****


    // Intermediate signals for input port

logic [511:0] data_in_2;
logic in_valid_2;
logic in_ready_2;

    // Intermediate signals for output port

logic [13:0] data_out_2;
logic out_valid_2;
logic out_ready_2;

    // Instance declaration

QSum #( ) qsum_inst_2 (
.clk(clk),
.rst(rst),
.data_in(data_in_2),
.data_in_valid(in_valid_2),
.data_in_ready(in_ready_2),
.data_out(data_out_2),
.data_out_valid(out_valid_2),
.data_out_ready(out_ready_2)
);


// Connecting intermediate signals of node 2 to previous intermediate signals
assign out_ready_to_buffer_edge__from_op1_to_op2 = out_ready_2;
assign data_in_2 = out_from_buffer_edge__from_op1_to_op2;
assign in_valid_2 = out_valid_from_buffer_edge__from_op1_to_op2;
// End of connections for node 2

assign in_ready_2 = data_out_ready;
assign data_out = data_out_2;
assign data_out_valid = out_valid_2;

endmodule