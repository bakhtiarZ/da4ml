#include <cstddef>
#include "binder_util.hh"
#include "Vmod_op_1__QDense_wrapper.h"

struct mod_op_1__QDense_wrapper_config {
    static const size_t N_inp = 128;
    static const size_t N_out = 64;
    static const size_t max_inp_bw = 9;
    static const size_t max_out_bw = 8;
    static const size_t II = 0;
    static const size_t latency = 0;
    typedef Vmod_op_1__QDense_wrapper dut_t;
};

extern "C" {
bool openmp_enabled() {
    return _openmp;
}

void inference(int32_t *c_inp, int32_t *c_out, size_t n_samples, size_t n_threads) {
    batch_inference<mod_op_1__QDense_wrapper_config>(c_inp, c_out, n_samples, n_threads);
}
}
