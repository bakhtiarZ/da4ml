from dataclasses import dataclass
from typing import Any, Callable, Type
import keras
from keras import ops
import tensorflow as tf

from hgq.layers import QDense, QSum
from da4ml.cmvm.types import CombLogic
from da4ml.graph_ir.hardware_types import QSumLogic


@dataclass
class DataSchedule():
    shape_req: Callable[[tuple], bool]
    minimum_output_shape: Callable[[tuple], tuple]
    minimum_input_shape: Callable[[tuple], tuple]
    schedule : Callable 
    rebuilder:  Callable
    buffer_type: str 
    hardware_type: Type
    
class DataScheduler():
    def __init__(self, data_schedule) -> None:
        self.data_schedule = data_schedule
            
    def call(self, x : keras.KerasTensor) -> list[keras.KerasTensor]:
        output = None
        assert self.data_schedule.shape_req(x.shape), f"Input shape {x.shape} does not meet schedule requirements"
        output = self.data_schedule.schedule(x)        
        return output

    def rebuild_tensor(self, chunks, original):
        return self.data_schedule.rebuilder(chunks, original)
    
    def __call__(self, x):
        return self.call(x)

def dense_schedule(x: keras.KerasTensor):
    f = x.shape[-1]
    if f is None:
        raise ValueError("Feature dim must be statically known")

    flattened = tf.reshape(x, (-1, f))      # (N, f)
    n = tf.shape(flattened)[0]              # scalar tensor ()

    # sizes is shape (N,), each split size = 1
    sizes = tf.ones([n], dtype=tf.int32)

    vectors = tf.split(flattened, sizes, axis=0)  # list of N tensors, each (1, f)
    return vectors


def input_dense_schedule_requirement(shape: tuple) -> bool:
    met = len(shape) >= 1 and shape[-1] > 0
    return met

def minimum_output_shape_for_dense(output_shape: tuple) -> tuple: # add parallelism here as input if you like
    return (1, output_shape[-1])

def minimum_input_shape_for_dense(input_shape: tuple) -> tuple: # add parallelism here as input if you like
    return (1, input_shape[-1])

def dense_rebuilder(x: list[keras.KerasTensor], original_shape: tuple) -> keras.KerasTensor:
    if original_shape[0] is None:
        original_shape = original_shape[1:]
    rebuilt = tf.concat(x, axis=0)
    rebuilt = tf.reshape(rebuilt, original_shape)
    return rebuilt


# def oned_conv_schedule(x: keras.KerasTensor, kernel_size: int, stride = 1, padding = 'same'):

#     # this function should take a keras tensor (imagining the batch dimension is active, so (None, 5, 3, 2) means ignore the none, use 5,3,2), and create a list of input kernels, where each kernel is a k sized vector of the input across
#     # all input channels, the idea is that I can pass each element into a minimal 1d conv kernel so it produces a similar shape of output kernels per clock. For now, assume padding is always 'same' and stride is always 1.
#     windows = []
#     if padding == 'same' and stride = 1:
#         paddingsize = (kernel_size - 1) / 2
#         first_window = [[0] * paddingsize , x[:, paddingsize + 2] 
#     for i in x.shape[-1]: #along the final axis
#         window = x[:, i : k + i]
#         windows.append(window)
    
    


#     output_shape = (ic, kernel_size)

#     # split x into kernel size chunks

def conv1d_extract_windows(
    x: keras.KerasTensor,
    kernel_size: int,
    stride: int = 1,
    padding: str = "same",
):
    B, L, C = x.shape

    # Pad along length axis (axis=1)
    if padding == "same":
        pad_total = kernel_size - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x_pad = ops.pad(x, paddings=[[0, 0], [pad_left, pad_right], [0, 0]])
        L_out = L  # for stride=1 "same"
    else:  # "valid"
        x_pad = x
        # L_out = floor((L - K)/stride) + 1 when L is known
        if L is None:
            L_out = None
        else:
            L_out = (L - kernel_size) // stride + 1

    # Build windows.
    # If L_out is dynamic (None), we can still build using a while_loop/map_fn,
    # but for most hardware-style schedules you likely have static L anyway.
    if L_out is None:
        # Dynamic-length fallback (works, but heavier)
        # We compute n = (len(x_pad)-K)//stride + 1 dynamically.
        Lp = ops.shape(x_pad)[1]
        n = (Lp - kernel_size) // stride + 1  # dynamic int tensor

        idx = ops.arange(n) * stride  # (n,)
        # For each start index s, slice x_pad[:, s:s+K, :]
        def take_window(s):
            return x_pad[:, s:s + kernel_size, :]  # (B, K, C)

        w = ops.map_fn(take_window, idx, fn_output_signature=x_pad.dtype)  # (n, B, K, C)
        windows = ops.transpose(w, (1, 0, 2, 3))  # (B, n, K, C)
        return windows

    # Static-length fast path
    windows = []
    for start in range(0, int(L_out) * stride, stride):
        windows.append(x_pad[:, start:start + kernel_size, :])  # (B, K, C)

    # Stack -> (L_out, B, K, C) then transpose -> (B, L_out, K, C)
    w = ops.stack(windows, axis=0)
    w = ops.transpose(w, (1, 0, 2, 3))
    return w

def input_qsum_requirement(shape):
    return len(shape) > 0 and not all([axis is None for axis in shape])

def minimum_output_shape_for_qsum(output_shape, axis):
    new_output_shape = output_shape[:axis] + (1,) + output_shape[axis+1:]
    return new_output_shape

def minimum_input_shape_for_qsum(input_shape, axis):
    return input_shape[axis] is not None and input_shape[axis] > 0

def qsum_schedule(x, axis):
    # tensor --> split on that axis, sum clock by clock...
    new_output_shape = x.shape[:axis] + (1,) + x.shape[axis+1:]
    flattened_shape = (-1) + new_output_shape
    flattened = tf.reshape(x, flattened_shape)
    vectors = tf.unstack(flattened, axis=0)
    return vectors
    
def qsum_rebuilder(x, axis):
    return x # no rebuilding    

_SCHEDULE_REGISTRY : dict[type, DataSchedule] = {
    # keras.layers.Dense : DataSchedule(shape_req=input_dense_schedule_requirement, minimum_output_shape=minimum_output_shape_for_dense, minimum_input_shape=minimum_input_shape_for_dense, schedule=dense_schedule, rebuilder=dense_rebuilder),
    QDense : DataSchedule(shape_req=input_dense_schedule_requirement, minimum_output_shape=minimum_output_shape_for_dense, minimum_input_shape=minimum_input_shape_for_dense, schedule=dense_schedule, rebuilder=dense_rebuilder, buffer_type="fifo", hardware_type=CombLogic),
    
    QSum : DataSchedule(shape_req=input_qsum_requirement, minimum_output_shape=minimum_output_shape_for_qsum, minimum_input_shape=minimum_input_shape_for_qsum, schedule=qsum_schedule, rebuilder=qsum_rebuilder, buffer_type="fifo", hardware_type=QSumLogic)
}