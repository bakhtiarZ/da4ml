`timescale 1 ns / 1 ps

module mod_adder_for_qsum_node_2_wrapper (
    // verilator lint_off UNUSEDSIGNAL
    input [909:0] model_inp,
    // verilator lint_on UNUSEDSIGNAL
    output [13:0] model_out
);
    wire [525:0] packed_inp;
    wire [13:0] packed_out;

    assign packed_inp[7:0] = model_inp[7:0];
    assign packed_inp[15:8] = model_inp[21:14];
    assign packed_inp[23:16] = model_inp[35:28];
    assign packed_inp[31:24] = model_inp[49:42];
    assign packed_inp[39:32] = model_inp[63:56];
    assign packed_inp[47:40] = model_inp[77:70];
    assign packed_inp[55:48] = model_inp[91:84];
    assign packed_inp[63:56] = model_inp[105:98];
    assign packed_inp[71:64] = model_inp[119:112];
    assign packed_inp[79:72] = model_inp[133:126];
    assign packed_inp[87:80] = model_inp[147:140];
    assign packed_inp[95:88] = model_inp[161:154];
    assign packed_inp[103:96] = model_inp[175:168];
    assign packed_inp[111:104] = model_inp[189:182];
    assign packed_inp[119:112] = model_inp[203:196];
    assign packed_inp[127:120] = model_inp[217:210];
    assign packed_inp[135:128] = model_inp[231:224];
    assign packed_inp[143:136] = model_inp[245:238];
    assign packed_inp[151:144] = model_inp[259:252];
    assign packed_inp[159:152] = model_inp[273:266];
    assign packed_inp[167:160] = model_inp[287:280];
    assign packed_inp[175:168] = model_inp[301:294];
    assign packed_inp[183:176] = model_inp[315:308];
    assign packed_inp[191:184] = model_inp[329:322];
    assign packed_inp[199:192] = model_inp[343:336];
    assign packed_inp[207:200] = model_inp[357:350];
    assign packed_inp[215:208] = model_inp[371:364];
    assign packed_inp[223:216] = model_inp[385:378];
    assign packed_inp[231:224] = model_inp[399:392];
    assign packed_inp[239:232] = model_inp[413:406];
    assign packed_inp[247:240] = model_inp[427:420];
    assign packed_inp[255:248] = model_inp[441:434];
    assign packed_inp[263:256] = model_inp[455:448];
    assign packed_inp[271:264] = model_inp[469:462];
    assign packed_inp[279:272] = model_inp[483:476];
    assign packed_inp[287:280] = model_inp[497:490];
    assign packed_inp[295:288] = model_inp[511:504];
    assign packed_inp[303:296] = model_inp[525:518];
    assign packed_inp[311:304] = model_inp[539:532];
    assign packed_inp[319:312] = model_inp[553:546];
    assign packed_inp[327:320] = model_inp[567:560];
    assign packed_inp[335:328] = model_inp[581:574];
    assign packed_inp[343:336] = model_inp[595:588];
    assign packed_inp[351:344] = model_inp[609:602];
    assign packed_inp[359:352] = model_inp[623:616];
    assign packed_inp[367:360] = model_inp[637:630];
    assign packed_inp[375:368] = model_inp[651:644];
    assign packed_inp[383:376] = model_inp[665:658];
    assign packed_inp[391:384] = model_inp[679:672];
    assign packed_inp[399:392] = model_inp[693:686];
    assign packed_inp[407:400] = model_inp[707:700];
    assign packed_inp[415:408] = model_inp[721:714];
    assign packed_inp[423:416] = model_inp[735:728];
    assign packed_inp[431:424] = model_inp[749:742];
    assign packed_inp[439:432] = model_inp[763:756];
    assign packed_inp[447:440] = model_inp[777:770];
    assign packed_inp[455:448] = model_inp[791:784];
    assign packed_inp[463:456] = model_inp[805:798];
    assign packed_inp[471:464] = model_inp[819:812];
    assign packed_inp[479:472] = model_inp[833:826];
    assign packed_inp[487:480] = model_inp[847:840];
    assign packed_inp[495:488] = model_inp[861:854];
    assign packed_inp[503:496] = model_inp[875:868];
    assign packed_inp[511:504] = model_inp[889:882];
    assign packed_inp[525:512] = model_inp[909:896];

    mod_adder_for_qsum_node_2 op (
        .model_inp(packed_inp),
        .model_out(packed_out)
    );

    assign model_out[13:0] = packed_out[13:0];

endmodule
