from math import log2

import keras

from hgq.layers import QAdd, QEinsumDense, QSum, QEinsumDenseBatchnorm, QBatchNormDense
from hgq.config import QuantizerConfigScope


def get_gnn(conf, uq1: bool = False):
    N = conf.n_constituents
    n = 3 if conf.pt_eta_phi else 16
    heterogeneous_axis = None if not uq1 else (-1,)

    with (
        QuantizerConfigScope(place=('weight', 'bias'), overflow_mode='SAT_SYM'),
        QuantizerConfigScope(place='datalane', heterogeneous_axis=heterogeneous_axis),
    ):
        inp = keras.layers.Input((N, n))

        pool_scale = 2.0 ** -round(log2(N))
        x = QEinsumDenseBatchnorm(
            'bnc,cC->bnC', (N, 64), bias_axes='C', activation='relu'
        )(inp)
        s = QEinsumDenseBatchnorm(
            'bnc,cC->bnC',
            (N, 64),
            bias_axes='C',
            activation='relu',
        )(x)
        d = QEinsumDenseBatchnorm(
            'bnc,cC->bnC', (1, 64), bias_axes='C', activation='relu'
        )(QSum(axes=1, scale=pool_scale, keepdims=True)(x))
        x = QAdd()([s, d])

        x = QEinsumDenseBatchnorm(
            'bnc,cC->bnC',
            (N, 64),
            bias_axes='C',
            activation='relu',
        )(x)
        x = QSum(axes=1, scale=1 / 16, keepdims=False)(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 64, bias_axes='C', activation='relu')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 32, bias_axes='C', activation='relu')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 16, bias_axes='C', activation='relu')(x)
        out = QEinsumDenseBatchnorm('bc,cC->bC', 5, bias_axes='C')(x)

    model = keras.Model(inputs=inp, outputs=out)
    return model

def dense_gnn(conf, uq1: bool = False):
    N = conf.n_constituents
    n = 3 if conf.pt_eta_phi else 16
    heterogeneous_axis = None if not uq1 else (-1,)
    with (
        QuantizerConfigScope(place=('weight', 'bias'), overflow_mode='SAT_SYM'),
        QuantizerConfigScope(place='datalane', heterogeneous_axis=heterogeneous_axis),
    ):
        inp = keras.layers.Input((N, n))

        pool_scale = 2.0 ** -round(log2(N))
        x = QBatchNormDense(64, activation='relu')(inp)
        # x = QEinsumDenseBatchnorm(
        #     'bnc,cC->bnC', (N, 64), bias_axes='C', activation='relu'
        # )(inp)
        s = QBatchNormDense(64, activation='relu')(x) # N, 64
        # s = QEinsumDenseBatchnorm(
        #     'bnc,cC->bnC',
        #     (N, 64),
        #     bias_axes='C',
        #     activation='relu',
        # )(x)
        sum = QSum(axes=1, scale=pool_scale, keepdims=True)(x) # sum over N, shape = 1,C
        d  = QBatchNormDense(64, 'relu')(sum) # 1,64
        # d = QEinsumDenseBatchnorm(
        #     'bnc,cC->bnC', (1, 64), bias_axes='C', activation='relu'
        # )(QSum(axes=1, scale=pool_scale, keepdims=True)(x))
        x = QAdd()([s, d]) # N+1, 64 ?
        x = QBatchNormDense(64, 'relu')(x)
        # x = QEinsumDenseBatchnorm(
        #     'bnc,cC->bnC',
        #     (N, 64),
        #     bias_axes='C',
        #     activation='relu',
        # )(x)
        x = QSum(axes=1, scale=1 / 16, keepdims=False)(x) # 1,64
        x = QBatchNormDense(64, 'relu')(x)
        # x = QEinsumDenseBatchnorm('bc,cC->bC', 64, bias_axes='C', activation='relu')(x)
        x = QBatchNormDense(32,'relu')(x)
        # x = QEinsumDenseBatchnorm('bc,cC->bC', 32, bias_axes='C', activation='relu')(x)
        x = QBatchNormDense(16,'relu')(x)
        # x = QEinsumDenseBatchnorm('bc,cC->bC', 16, bias_axes='C', activation='relu')(x)
        out = QBatchNormDense(5,'relu')(x)
        # out = QEinsumDenseBatchnorm('bc,cC->bC', 5, bias_axes='C')(x)

    model = keras.Model(inputs=inp, outputs=out)
    return model