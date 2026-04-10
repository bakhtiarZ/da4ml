import keras
import tensorflow as tf

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

