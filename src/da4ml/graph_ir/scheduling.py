from dataclasses import dataclass
from typing import Any, Callable, Type
import keras
import tensorflow as tf

from hgq.layers import QDense

@dataclass
class DataSchedule():
    shape_req: Callable[[tuple], bool]
    output_shape: Callable[[tuple], tuple]
    minimum_input_shape: Callable[[tuple], tuple]
    schedule : Callable 
    rebuilder:  Callable

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

def output_shape_for_dense(input_shape: tuple) -> tuple:
    return input_shape[:-1] + (1,) #? not me...

def minimum_input_shape_for_dense(input_shape: tuple) -> tuple:
    return (input_shape[-1],)

def dense_rebuilder(x: list[keras.KerasTensor], original_shape: tuple) -> keras.KerasTensor:
    if original_shape[0] is None:
        original_shape = original_shape[1:]
    rebuilt = tf.concat(x, axis=0)
    rebuilt = tf.reshape(rebuilt, original_shape)
    return rebuilt


_SCHEDULE_REGISTRY : dict[type, DataSchedule] = {
    keras.layers.Dense : DataSchedule(shape_req=input_dense_schedule_requirement, output_shape=output_shape_for_dense, minimum_input_shape=minimum_input_shape_for_dense, schedule=dense_schedule, rebuilder=dense_rebuilder),

    QDense : DataSchedule(shape_req=input_dense_schedule_requirement, output_shape=output_shape_for_dense, minimum_input_shape=minimum_input_shape_for_dense, schedule=dense_schedule, rebuilder=dense_rebuilder)

}