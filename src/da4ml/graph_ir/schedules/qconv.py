import keras
from keras import ops

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
