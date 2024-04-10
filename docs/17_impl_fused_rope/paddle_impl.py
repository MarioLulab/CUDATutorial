import paddle
import numpy as np

def get_sin_cos_tensor(seq_len, head_dim, sign = 1):
    pos_seq = paddle.arange(0, seq_len, 1)
    indices = paddle.arange(0, head_dim, 2)
    indices = 1 / 10000 ** (indices / head_dim)
    sinusoid_inp = pos_seq.unsqueeze(1) * indices.unsqueeze(0)
    sin_sin = np.empty((seq_len * head_dim), dtype=np.float32)
    cos_cos = np.empty((seq_len * head_dim), dtype=np.float32)
    iter_array = np.nditer(sinusoid_inp.numpy())
    
    i = 0
    for value in iter_array:
        sin_sin[i * 2] = sign * np.sin(value)
        cos_cos[i * 2] = np.cos(value)
        sin_sin[i * 2 + 1] = np.sin(value)
        cos_cos[i * 2 + 1] = np.cos(value)
        
        i += 1
    
    tensor_sin = paddle.reshape(
        paddle.to_tensor(sin_sin),
        [1, seq_len, 1, head_dim]
    )
    tensor_cos = paddle.reshape(
        paddle.to_tensor(cos_cos),
        [1, seq_len, 1, head_dim]
    )
    
    return tensor_sin, tensor_cos


def mult_qkv_rotate_half(value, cos_tensor, sin_tensor):
    rotate_half_qkv = paddle.concat(
        (-value[..., value.shape[-1] // 2:],
         value[..., :value.shape[-1] // 2]),
        axis = -1
    ).reshape(value.shape)
    
    
    qkv = value * cos_tensor + rotate_half_qkv * sin_tensor
    return qkv

def rotate_half(value):
    b, s, n, head_dim = value.shape
    
    return paddle.concat(
        (
            - value[..., head_dim / 2:],
            value[..., :head_dim / 2]
        ),
        axis = -1
    )
    
def get_cos_sin(value):
    b, s, n, head_dim = value.shape
    
    inv_freq = 1.0 / (10000 ** (paddle.arange(0, head_dim, 2).astype(paddle.float32) / head_dim))
    t = paddle.arange(0, s).astype(paddle.float32)
    freq = paddle.einsum("i,j->ij", t, inv_freq)
    freq = paddle.concat((freq, freq), axis=-1)
    return freq.cos(), freq.sin()

def rope(value):
    cos, sin = get_cos_sin(value)
    return value * cos + rorate_half(value) * sin