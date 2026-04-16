#include <cstddef>
#include "binder_util.hh"
#include "Vmod_adder_for_qsum_node_2_wrapper.h"

struct mod_adder_for_qsum_node_2_wrapper_config {
    static const size_t N_inp = 65;
    static const size_t N_out = 1;
    static const size_t max_inp_bw = 14;
    static const size_t max_out_bw = 14;
    static const size_t II = 0;
    static const size_t latency = 0;
    typedef Vmod_adder_for_qsum_node_2_wrapper dut_t;
};

extern "C" {
bool openmp_enabled() {
    return _openmp;
}

void inference(int32_t *c_inp, int32_t *c_out, size_t n_samples, size_t n_threads) {
    batch_inference<mod_adder_for_qsum_node_2_wrapper_config>(c_inp, c_out, n_samples, n_threads);
}
}
