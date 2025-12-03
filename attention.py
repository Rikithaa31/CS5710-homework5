import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Q: (batch, seq_len_q, d_k)
    K: (batch, seq_len_k, d_k)
    V: (batch, seq_len_k, d_v)
    """
    d_k = Q.shape[-1]

    # Attention scores: QK^T / sqrt(d_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    # Softmax to get attention weights
    attn = softmax(scores, axis=-1)

    # Context vector
    context = np.matmul(attn, V)

    return attn, context


# Quick test
if __name__ == "__main__":
    Q = np.random.rand(2, 5, 4)
    K = np.random.rand(2, 5, 4)
    V = np.random.rand(2, 5, 6)

    attn, ctx = scaled_dot_product_attention(Q, K, V)

    print("Attention shape:", attn.shape)
    print("Context shape:", ctx.shape)
