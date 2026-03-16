`timescale 1ns/1ps

module fifo_rv #(
    parameter int unsigned DEPTH      = 8,
    parameter int unsigned DATA_WIDTH = 32
) (
    input  logic                  clk,
    input  logic                  rst,   // synchronous, active-high

    // input (producer) side
    input  logic [DATA_WIDTH-1:0] in_data,
    input  logic                  in_valid,
    output logic                  in_ready,

    // output (consumer) side
    output logic [DATA_WIDTH-1:0] out_data,
    output logic                  out_valid,
    input  logic                  out_ready,

    // status
    output logic                  full,
    output logic                  empty,
    output logic [$clog2(DEPTH+1)-1:0] count
);

    initial begin
        if (DEPTH < 1)      $fatal(1, "fifo_rv: DEPTH must be >= 1");
        if (DATA_WIDTH < 1) $fatal(1, "fifo_rv: DATA_WIDTH must be >= 1");
    end

    localparam int unsigned CNT_W = $clog2(DEPTH+1);

    generate
        // ------------------------------------------------------------
        // DEPTH == 1 : single-element ready/valid buffer
        // ------------------------------------------------------------
        if (DEPTH == 1) begin : g_depth1
            logic [DATA_WIDTH-1:0] data_r;
            logic                  full_r;

            logic do_pop, do_push;

            always_comb begin
                out_valid = full_r;
                out_data  = data_r;

                // if full but consumer ready, we can accept a new element this cycle
                in_ready  = !full_r || out_ready;

                do_pop  = out_valid && out_ready;      // full_r && out_ready
                do_push = in_valid  && in_ready;

                empty = !full_r;
                full  = full_r;
                count = CNT_W'(full_r ? 1 : 0);
            end

            always_ff @(posedge clk) begin
                if (rst) begin
                    full_r <= 1'b0;
                    data_r <= '0;
                end else begin
                    unique case ({do_push, do_pop})
                        2'b10: begin
                            data_r <= in_data;
                            full_r <= 1'b1;
                        end
                        2'b01: begin
                            full_r <= 1'b0;
                        end
                        2'b11: begin
                            // consumer takes old, producer provides new (no bubble)
                            data_r <= in_data;
                            full_r <= 1'b1;
                        end
                        default: begin end
                    endcase
                end
            end
        end

        // ------------------------------------------------------------
        // DEPTH > 1 : circular buffer FIFO
        // ------------------------------------------------------------
        else begin : g_depthN
            localparam int unsigned PTR_W = $clog2(DEPTH);

            logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];
            logic [PTR_W-1:0]      rd_ptr, wr_ptr;

            logic do_pop, do_push;

            always_comb begin
                empty     = (count == '0);
                full      = (count == CNT_W'(DEPTH));

                out_valid = !empty;
                in_ready  = !full;

                do_pop  = out_valid && out_ready;
                do_push = in_valid  && in_ready;

                out_data = out_valid ? mem[rd_ptr] : '0;
            end

            always_ff @(posedge clk) begin
                if (rst) begin
                    count  <= '0;
                    rd_ptr <= '0;
                    wr_ptr <= '0;
                end else begin
                    // write
                    if (do_push) begin
                        mem[wr_ptr] <= in_data;
                        if (wr_ptr == PTR_W'(DEPTH-1)) wr_ptr <= '0;
                        else                           wr_ptr <= wr_ptr + PTR_W'(1);
                    end

                    // read pointer advance
                    if (do_pop) begin
                        if (rd_ptr == PTR_W'(DEPTH-1)) rd_ptr <= '0;
                        else                           rd_ptr <= rd_ptr + PTR_W'(1);
                    end

                    // count update
                    unique case ({do_push, do_pop})
                        2'b10: count <= count + CNT_W'(1);
                        2'b01: count <= count - CNT_W'(1);
                        default: /* 00 or 11 */ count <= count;
                    endcase
                end
            end
        end
    endgenerate

endmodule
