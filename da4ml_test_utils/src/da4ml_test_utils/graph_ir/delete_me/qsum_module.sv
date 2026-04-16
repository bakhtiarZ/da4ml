 module QSum #() (
    input logic clk,
    input logic rst,
    input logic [511:0] data_in,
    input logic data_in_valid,
    output logic data_in_ready,
    output logic [13:0] data_out,
    output logic data_out_valid,
    input logic data_out_ready
);

    localparam logic [0:0] COUNT_MINUS_ONE = 0;
    localparam logic [0:0] COUNT_VALUE     = 1;

    logic [13:0] sum_reg;
    logic [0:0] count_reg;
    logic full_reg;

    logic [13:0] adder_result;
    logic accept_input;
    logic accept_output;
    logic last_input;

    logic [525:0] packed_operands; assign packed_operands = {data_in, sum_reg};
    mod_adder_for_qsum_node_2 #(
    ) adder_for_qsum_node_2_inst (
        .model_inp(packed_operands),
        .model_out(adder_result)
    );

    assign data_in_ready      = !full_reg;
    assign data_out_valid     = full_reg;
    assign data_out      = sum_reg;

    assign accept_input  = data_in_valid && data_in_ready;
    assign accept_output = data_out_valid && data_out_ready;
    assign last_input    = (count_reg == COUNT_MINUS_ONE);

    always_ff @(posedge clk) begin
        if (rst) begin
            sum_reg   <= '0;
            count_reg <= '0;
            full_reg  <= 1'b0;
        end else begin
            if (accept_output) begin
                sum_reg   <= '0;
                count_reg <= '0;
                full_reg  <= 1'b0;
            end else if (accept_input) begin
                sum_reg <= adder_result;

                if (last_input) begin
                    count_reg <= COUNT_VALUE;
                    full_reg  <= 1'b1;
                end else begin
                    count_reg <= count_reg + 1'b1;
                end
            end
        end
    end

endmodule
