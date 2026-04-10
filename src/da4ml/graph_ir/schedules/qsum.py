import tensorflow as tf


def input_qsum_requirement(shape):
    return len(shape) > 0 and not all([axis is None for axis in shape])

def _normalize_axis(axis, rank):
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(f"Axis {axis} out of bounds for rank-{rank} shape")
    return axis

def minimum_output_shape_for_qsum(output_shape, axes):
    assert len(axes) == 1, "Currently only support single axis qsum scheduling"
    axis = _normalize_axis(axes[0], len(output_shape))
    return output_shape[:axis] + (1,) + output_shape[axis+1:]


def minimum_input_shape_for_qsum(input_shape, axes):
    assert len(axes) == 1, "Currently only support single axis qsum scheduling"
    axis = _normalize_axis(axes[0], len(input_shape))

    if input_shape[axis] is not None and input_shape[axis] > 0:
        return input_shape[:axis] + (1,) + input_shape[axis+1:]

    raise ValueError(
        f"Input shape {input_shape} does not meet minimum shape requirements "
        f"for qsum scheduling on axis {axis}"
    )
    
def qsum_schedule(x, axis):
    # tensor --> split on that axis, sum clock by clock...
    new_output_shape = x.shape[:axis] + (1,) + x.shape[axis+1:]
    flattened_shape = (-1) + new_output_shape
    flattened = tf.reshape(x, flattened_shape)
    vectors = tf.unstack(flattened, axis=0)
    return vectors
    
def qsum_rebuilder(x, axis):
    return x # no rebuilding    
