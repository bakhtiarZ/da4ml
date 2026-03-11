module QAdd #(
    parameter A_WIDTH = 16,
    parameter B_WIDTH = 8,
    parameter OUTPUT_WIDTH = 16
) (
    input  logic [A_WIDTH-1:0] i0,
    input  logic [B_WIDTH-1:0] i1,
    output logic [OUTPUT_WIDTH-1:0] o
);
    logic [OUTPUT_WIDTH-1:0] i0_ext;
    logic [OUTPUT_WIDTH-1:0] i1_ext;

    assign i0_ext = OUTPUT_WIDTH'(i0);
    assign i1_ext = OUTPUT_WIDTH'(i1);
    assign o = i0_ext + i1_ext;
endmodule

module QSum #(
    parameter INPUT_WIDTH  = 8,
    parameter OUTPUT_WIDTH = 16,
    parameter COUNT        = 4
)(
    input  logic [INPUT_WIDTH-1:0]  data_in,
    input  logic                    in_valid,
    output logic                    in_ready,

    output logic [OUTPUT_WIDTH-1:0] data_out,
    output logic                    out_valid,
    input  logic                    out_ready,

    input  logic                    clk,
    input  logic                    rst
);

    localparam int COUNT_W = (COUNT <= 1) ? 1 : $clog2(COUNT + 1);
    localparam logic [COUNT_W-1:0] COUNT_MINUS_ONE = COUNT - 1;
    localparam logic [COUNT_W-1:0] COUNT_VALUE     = COUNT;

    logic [OUTPUT_WIDTH-1:0] sum_reg;
    logic [COUNT_W-1:0]      count_reg;
    logic                    full_reg;

    logic [OUTPUT_WIDTH-1:0] adder_result;
    logic                    accept_input;
    logic                    accept_output;
    logic                    last_input;

    QAdd #(
        .A_WIDTH(OUTPUT_WIDTH),
        .B_WIDTH(INPUT_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH)
    ) adder (
        .i0(sum_reg),
        .i1(data_in),
        .o(adder_result)
    );

    assign in_ready      = !full_reg;
    assign out_valid     = full_reg;
    assign data_out      = sum_reg;

    assign accept_input  = in_valid && in_ready;
    assign accept_output = out_valid && out_ready;
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