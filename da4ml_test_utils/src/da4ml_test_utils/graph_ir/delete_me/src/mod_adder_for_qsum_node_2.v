`timescale 1ns/1ps

module mod_adder_for_qsum_node_2 (
    input [525:0] model_inp,
    output [13:0] model_out
);

    // verilator lint_off UNUSEDSIGNAL
    // Explicit quantization operation will drop bits if exists

    wire [7:0] v0; assign v0[7:0] = model_inp[7:0]; // 0.0
    wire [7:0] v1; assign v1[7:0] = model_inp[15:8]; // 0.0
    wire [7:0] v2; assign v2[7:0] = model_inp[23:16]; // 0.0
    wire [7:0] v3; assign v3[7:0] = model_inp[31:24]; // 0.0
    wire [7:0] v4; assign v4[7:0] = model_inp[39:32]; // 0.0
    wire [7:0] v5; assign v5[7:0] = model_inp[47:40]; // 0.0
    wire [7:0] v6; assign v6[7:0] = model_inp[55:48]; // 0.0
    wire [7:0] v7; assign v7[7:0] = model_inp[63:56]; // 0.0
    wire [7:0] v8; assign v8[7:0] = model_inp[71:64]; // 0.0
    wire [7:0] v9; assign v9[7:0] = model_inp[79:72]; // 0.0
    wire [7:0] v10; assign v10[7:0] = model_inp[87:80]; // 0.0
    wire [7:0] v11; assign v11[7:0] = model_inp[95:88]; // 0.0
    wire [7:0] v12; assign v12[7:0] = model_inp[103:96]; // 0.0
    wire [7:0] v13; assign v13[7:0] = model_inp[111:104]; // 0.0
    wire [7:0] v14; assign v14[7:0] = model_inp[119:112]; // 0.0
    wire [7:0] v15; assign v15[7:0] = model_inp[127:120]; // 0.0
    wire [7:0] v16; assign v16[7:0] = model_inp[135:128]; // 0.0
    wire [7:0] v17; assign v17[7:0] = model_inp[143:136]; // 0.0
    wire [7:0] v18; assign v18[7:0] = model_inp[151:144]; // 0.0
    wire [7:0] v19; assign v19[7:0] = model_inp[159:152]; // 0.0
    wire [7:0] v20; assign v20[7:0] = model_inp[167:160]; // 0.0
    wire [7:0] v21; assign v21[7:0] = model_inp[175:168]; // 0.0
    wire [7:0] v22; assign v22[7:0] = model_inp[183:176]; // 0.0
    wire [7:0] v23; assign v23[7:0] = model_inp[191:184]; // 0.0
    wire [7:0] v24; assign v24[7:0] = model_inp[199:192]; // 0.0
    wire [7:0] v25; assign v25[7:0] = model_inp[207:200]; // 0.0
    wire [7:0] v26; assign v26[7:0] = model_inp[215:208]; // 0.0
    wire [7:0] v27; assign v27[7:0] = model_inp[223:216]; // 0.0
    wire [7:0] v28; assign v28[7:0] = model_inp[231:224]; // 0.0
    wire [7:0] v29; assign v29[7:0] = model_inp[239:232]; // 0.0
    wire [7:0] v30; assign v30[7:0] = model_inp[247:240]; // 0.0
    wire [7:0] v31; assign v31[7:0] = model_inp[255:248]; // 0.0
    wire [7:0] v32; assign v32[7:0] = model_inp[263:256]; // 0.0
    wire [7:0] v33; assign v33[7:0] = model_inp[271:264]; // 0.0
    wire [7:0] v34; assign v34[7:0] = model_inp[279:272]; // 0.0
    wire [7:0] v35; assign v35[7:0] = model_inp[287:280]; // 0.0
    wire [7:0] v36; assign v36[7:0] = model_inp[295:288]; // 0.0
    wire [7:0] v37; assign v37[7:0] = model_inp[303:296]; // 0.0
    wire [7:0] v38; assign v38[7:0] = model_inp[311:304]; // 0.0
    wire [7:0] v39; assign v39[7:0] = model_inp[319:312]; // 0.0
    wire [7:0] v40; assign v40[7:0] = model_inp[327:320]; // 0.0
    wire [7:0] v41; assign v41[7:0] = model_inp[335:328]; // 0.0
    wire [7:0] v42; assign v42[7:0] = model_inp[343:336]; // 0.0
    wire [7:0] v43; assign v43[7:0] = model_inp[351:344]; // 0.0
    wire [7:0] v44; assign v44[7:0] = model_inp[359:352]; // 0.0
    wire [7:0] v45; assign v45[7:0] = model_inp[367:360]; // 0.0
    wire [7:0] v46; assign v46[7:0] = model_inp[375:368]; // 0.0
    wire [7:0] v47; assign v47[7:0] = model_inp[383:376]; // 0.0
    wire [7:0] v48; assign v48[7:0] = model_inp[391:384]; // 0.0
    wire [7:0] v49; assign v49[7:0] = model_inp[399:392]; // 0.0
    wire [7:0] v50; assign v50[7:0] = model_inp[407:400]; // 0.0
    wire [7:0] v51; assign v51[7:0] = model_inp[415:408]; // 0.0
    wire [7:0] v52; assign v52[7:0] = model_inp[423:416]; // 0.0
    wire [7:0] v53; assign v53[7:0] = model_inp[431:424]; // 0.0
    wire [7:0] v54; assign v54[7:0] = model_inp[439:432]; // 0.0
    wire [7:0] v55; assign v55[7:0] = model_inp[447:440]; // 0.0
    wire [7:0] v56; assign v56[7:0] = model_inp[455:448]; // 0.0
    wire [7:0] v57; assign v57[7:0] = model_inp[463:456]; // 0.0
    wire [7:0] v58; assign v58[7:0] = model_inp[471:464]; // 0.0
    wire [7:0] v59; assign v59[7:0] = model_inp[479:472]; // 0.0
    wire [7:0] v60; assign v60[7:0] = model_inp[487:480]; // 0.0
    wire [7:0] v61; assign v61[7:0] = model_inp[495:488]; // 0.0
    wire [7:0] v62; assign v62[7:0] = model_inp[503:496]; // 0.0
    wire [7:0] v63; assign v63[7:0] = model_inp[511:504]; // 0.0
    wire [13:0] v64; assign v64[13:0] = model_inp[525:512]; // 0.0
    wire [8:0] v65; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_65 (v54[7:0], v17[7:0], v65[8:0]); // 1.0
    wire [8:0] v66; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_66 (v23[7:0], v24[7:0], v66[8:0]); // 1.0
    wire [8:0] v67; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_67 (v46[7:0], v19[7:0], v67[8:0]); // 1.0
    wire [8:0] v68; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_68 (v44[7:0], v40[7:0], v68[8:0]); // 1.0
    wire [8:0] v69; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_69 (v5[7:0], v11[7:0], v69[8:0]); // 1.0
    wire [8:0] v70; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_70 (v28[7:0], v14[7:0], v70[8:0]); // 1.0
    wire [8:0] v71; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_71 (v51[7:0], v18[7:0], v71[8:0]); // 1.0
    wire [8:0] v72; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_72 (v52[7:0], v36[7:0], v72[8:0]); // 1.0
    wire [8:0] v73; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_73 (v60[7:0], v63[7:0], v73[8:0]); // 1.0
    wire [8:0] v74; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_74 (v61[7:0], v31[7:0], v74[8:0]); // 1.0
    wire [8:0] v75; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_75 (v0[7:0], v1[7:0], v75[8:0]); // 1.0
    wire [8:0] v76; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_76 (v59[7:0], v32[7:0], v76[8:0]); // 1.0
    wire [8:0] v77; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_77 (v57[7:0], v33[7:0], v77[8:0]); // 1.0
    wire [8:0] v78; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_78 (v26[7:0], v6[7:0], v78[8:0]); // 1.0
    wire [8:0] v79; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_79 (v43[7:0], v20[7:0], v79[8:0]); // 1.0
    wire [8:0] v80; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_80 (v47[7:0], v9[7:0], v80[8:0]); // 1.0
    wire [8:0] v81; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_81 (v50[7:0], v37[7:0], v81[8:0]); // 1.0
    wire [8:0] v82; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_82 (v49[7:0], v38[7:0], v82[8:0]); // 1.0
    wire [8:0] v83; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_83 (v48[7:0], v4[7:0], v83[8:0]); // 1.0
    wire [8:0] v84; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_84 (v22[7:0], v2[7:0], v84[8:0]); // 1.0
    wire [8:0] v85; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_85 (v45[7:0], v39[7:0], v85[8:0]); // 1.0
    wire [8:0] v86; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_86 (v10[7:0], v21[7:0], v86[8:0]); // 1.0
    wire [8:0] v87; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_87 (v3[7:0], v7[7:0], v87[8:0]); // 1.0
    wire [8:0] v88; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_88 (v29[7:0], v30[7:0], v88[8:0]); // 1.0
    wire [8:0] v89; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_89 (v62[7:0], v15[7:0], v89[8:0]); // 1.0
    wire [8:0] v90; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_90 (v58[7:0], v16[7:0], v90[8:0]); // 1.0
    wire [8:0] v91; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_91 (v41[7:0], v42[7:0], v91[8:0]); // 1.0
    wire [8:0] v92; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_92 (v55[7:0], v8[7:0], v92[8:0]); // 1.0
    wire [8:0] v93; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_93 (v56[7:0], v34[7:0], v93[8:0]); // 1.0
    wire [8:0] v94; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_94 (v13[7:0], v27[7:0], v94[8:0]); // 1.0
    wire [8:0] v95; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_95 (v53[7:0], v35[7:0], v95[8:0]); // 1.0
    wire [8:0] v96; shift_adder #(8, 8, 1, 1, 9, 0, 0) op_96 (v12[7:0], v25[7:0], v96[8:0]); // 1.0
    wire [9:0] v97; shift_adder #(9, 9, 1, 1, 10, 0, 0) op_97 (v65[8:0], v66[8:0], v97[9:0]); // 2.0
    wire [9:0] v98; shift_adder #(9, 9, 1, 1, 10, 0, 0) op_98 (v67[8:0], v68[8:0], v98[9:0]); // 2.0
    wire [9:0] v99; shift_adder #(9, 9, 1, 1, 10, 0, 0) op_99 (v69[8:0], v70[8:0], v99[9:0]); // 2.0
    wire [9:0] v100; shift_adder #(9, 9, 1, 1, 10, 0, 0) op_100 (v71[8:0], v72[8:0], v100[9:0]); // 2.0
    wire [9:0] v101; shift_adder #(9, 9, 1, 1, 10, 0, 0) op_101 (v74[8:0], v75[8:0], v101[9:0]); // 2.0
    wire [14:0] v102; shift_adder #(14, 9, 1, 1, 15, 0, 0) op_102 (v64[13:0], v76[8:0], v102[14:0]); // 2.0
    wire [9:0] v103; shift_adder #(9, 9, 1, 1, 10, 0, 0) op_103 (v77[8:0], v78[8:0], v103[9:0]); // 2.0
    wire [9:0] v104; shift_adder #(9, 9, 1, 1, 10, 0, 0) op_104 (v79[8:0], v80[8:0], v104[9:0]); // 2.0
    wire [9:0] v105; shift_adder #(9, 9, 1, 1, 10, 0, 0) op_105 (v81[8:0], v82[8:0], v105[9:0]); // 2.0
    wire [9:0] v106; shift_adder #(9, 9, 1, 1, 10, 0, 0) op_106 (v83[8:0], v84[8:0], v106[9:0]); // 2.0
    wire [9:0] v107; shift_adder #(9, 9, 1, 1, 10, 0, 0) op_107 (v85[8:0], v86[8:0], v107[9:0]); // 2.0
    wire [9:0] v108; shift_adder #(9, 9, 1, 1, 10, 0, 0) op_108 (v87[8:0], v88[8:0], v108[9:0]); // 2.0
    wire [9:0] v109; shift_adder #(9, 9, 1, 1, 10, 0, 0) op_109 (v89[8:0], v90[8:0], v109[9:0]); // 2.0
    wire [9:0] v110; shift_adder #(9, 9, 1, 1, 10, 0, 0) op_110 (v91[8:0], v92[8:0], v110[9:0]); // 2.0
    wire [9:0] v111; shift_adder #(9, 9, 1, 1, 10, 0, 0) op_111 (v93[8:0], v94[8:0], v111[9:0]); // 2.0
    wire [9:0] v112; shift_adder #(9, 9, 1, 1, 10, 0, 0) op_112 (v95[8:0], v96[8:0], v112[9:0]); // 2.0
    wire [10:0] v113; shift_adder #(10, 10, 1, 1, 11, 0, 0) op_113 (v97[9:0], v98[9:0], v113[10:0]); // 3.0
    wire [10:0] v114; shift_adder #(10, 10, 1, 1, 11, 0, 0) op_114 (v99[9:0], v100[9:0], v114[10:0]); // 3.0
    wire [10:0] v115; shift_adder #(9, 10, 1, 1, 11, 0, 0) op_115 (v73[8:0], v101[9:0], v115[10:0]); // 3.0
    wire [10:0] v116; shift_adder #(10, 10, 1, 1, 11, 0, 0) op_116 (v103[9:0], v104[9:0], v116[10:0]); // 3.0
    wire [10:0] v117; shift_adder #(10, 10, 1, 1, 11, 0, 0) op_117 (v105[9:0], v106[9:0], v117[10:0]); // 3.0
    wire [10:0] v118; shift_adder #(10, 10, 1, 1, 11, 0, 0) op_118 (v107[9:0], v108[9:0], v118[10:0]); // 3.0
    wire [10:0] v119; shift_adder #(10, 10, 1, 1, 11, 0, 0) op_119 (v109[9:0], v110[9:0], v119[10:0]); // 3.0
    wire [10:0] v120; shift_adder #(10, 10, 1, 1, 11, 0, 0) op_120 (v111[9:0], v112[9:0], v120[10:0]); // 3.0
    wire [11:0] v121; shift_adder #(11, 11, 1, 1, 12, 0, 0) op_121 (v114[10:0], v115[10:0], v121[11:0]); // 4.0
    wire [14:0] v122; shift_adder #(15, 11, 1, 1, 15, 0, 0) op_122 (v102[14:0], v116[10:0], v122[14:0]); // 4.0
    wire [11:0] v123; shift_adder #(11, 11, 1, 1, 12, 0, 0) op_123 (v117[10:0], v118[10:0], v123[11:0]); // 4.0
    wire [11:0] v124; shift_adder #(11, 11, 1, 1, 12, 0, 0) op_124 (v119[10:0], v120[10:0], v124[11:0]); // 4.0
    wire [12:0] v125; shift_adder #(11, 12, 1, 1, 13, 0, 0) op_125 (v113[10:0], v121[11:0], v125[12:0]); // 5.0
    wire [12:0] v126; shift_adder #(12, 12, 1, 1, 13, 0, 0) op_126 (v123[11:0], v124[11:0], v126[12:0]); // 5.0
    wire [14:0] v127; shift_adder #(15, 13, 1, 1, 15, 0, 0) op_127 (v122[14:0], v126[12:0], v127[14:0]); // 6.0
    wire [14:0] v128; shift_adder #(13, 15, 1, 1, 15, 0, 0) op_128 (v125[12:0], v127[14:0], v128[14:0]); // 7.0
    wire [13:0] v129; assign v129[13:0] = v128[13:0]; // 7.0

    // verilator lint_on UNUSEDSIGNAL

    assign model_out[13:0] = v129[13:0];

    endmodule
