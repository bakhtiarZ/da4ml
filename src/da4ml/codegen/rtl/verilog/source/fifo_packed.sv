`timescale 1ns/1ps

module fifo_packed #(
    parameter int DEPTH      = 8,
    parameter int DATA_WIDTH = 8,
    parameter int DATA_SIZE  = 3
) (
    input  logic clk,
    input  logic rst,

    // Input stream: a packed "packet" of DATA_SIZE words
    input  logic [DATA_SIZE*DATA_WIDTH-1:0] in_data,
    input  logic                            in_valid,
    output logic                            in_ready,

    // Output stream: a packed "packet" of DATA_SIZE words
    output logic [DATA_SIZE*DATA_WIDTH-1:0] out_data,
    output logic                            out_valid,
    input  logic                            out_ready,

    // Flags / count
    output logic                            full,
    output logic                            empty,
    output logic [$clog2(DEPTH):0]          count
);

    localparam int PACK_W = DATA_SIZE * DATA_WIDTH;
    localparam int PTR_W  = (DEPTH <= 1) ? 1 : $clog2(DEPTH);

    // Storage: DEPTH slots, each slot is one packed packet
    logic [PACK_W-1:0] ram [0:DEPTH-1];

    logic [PTR_W-1:0] w_ptr, r_ptr;

    // Handshake events
    logic push, pop;
    assign push = in_valid  && in_ready;
    assign pop  = out_valid && out_ready;

    // Status
    assign empty     = (count == 0);
    assign full      = (count == DEPTH);
    assign in_ready  = !full;
    assign out_valid = !empty;

    // Count update
    always_ff @(posedge clk) begin
        if (rst) begin
            count <= '0;
        end else begin
            unique case ({push, pop})
                2'b10: if (!full)  count <= count + 1'b1; // push only
                2'b01: if (!empty) count <= count - 1'b1; // pop only
                default: count <= count;                  // no change or push+pop
            endcase
        end
    end

    // Write logic (push)
    always_ff @(posedge clk) begin
        if (rst) begin
            w_ptr <= '0;
        end else if (push) begin
            ram[w_ptr] <= in_data;

            if (w_ptr == DEPTH-1) w_ptr <= '0;
            else                  w_ptr <= w_ptr + 1'b1;
        end
    end

    // Read data (combinational view of current read pointer)
    // out_data is always driven from ram[r_ptr]; valid tells you if it's meaningful.
    always_comb begin
        out_data = ram[r_ptr];
    end

    // Read pointer advance (pop)
    always_ff @(posedge clk) begin
        if (rst) begin
            r_ptr <= '0;
        end else if (pop) begin
            if (r_ptr == DEPTH-1) r_ptr <= '0;
            else                  r_ptr <= r_ptr + 1'b1;
        end
    end

    // Optional helpers if you ever want word-level access to out_data/in_data:
    // function automatic logic [DATA_WIDTH-1:0] get_word(input logic [PACK_W-1:0] pkt, input int idx);
    //     return pkt[idx*DATA_WIDTH +: DATA_WIDTH];
    // endfunction

endmodule
