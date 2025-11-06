from typing import List
import numpy as np

def computeRTG(rewards: List[float]) -> List[float]:
    """
    rewards: list of rewards [r0, r1, ..., rT]
    returns: list of RTG values [RTG0, RTG1, ..., RTG_T]
    """
    rtg = []
    cumulative = 0

    for r in reversed(rewards):
        cumulative += float(r)
        rtg.append(cumulative)

    return list(reversed(rtg))

def heuristic_policy(state: List[float], K = 0.0) -> float:
    """
    state: [tilt angle, angular vel, cart position, cart velocity]
    return: external force
    """
    theta, *_ = state

    # proportional controller
    u = K * theta
    return u

def _softmax(M: np.array) -> np.array:
    b = np.max(M)
    e_x = np.exp(M - b)
    return e_x / np.sum(e_x)

def attention(Q: np.array, K: np.array, V: np.array):
    """
    Q: Qeury Matrix
    K: Key Matrix
    V: Value Matrix
    """
    result = np.dot(Q, K.T)
    normalised = _softmax(result)
    output = np.dot(normalised, V)

    return output

def multi_head_attention(x: np.array, head_count=4):
    """
    x: input sequence
    num_heads: number of attention heads
    
    Returns: combined output from all heads
    """
    seq_len , d_model = x.shape
    d_head = d_model // head_count

    outputs = []

    for head in range(head_count):
        x_head = x[:, head * d_head : (head+1) * d_head]

        W_q = np.random.randn(d_head, d_head)
        W_k = np.random.randn(d_head, d_head)
        W_v = np.random.randn(d_head, d_head)

        Q = x_head @ W_q.T
        K = x_head @ W_k.T
        V = x_head @ W_v.T

        head_output = attention(Q, K, V)
        outputs.append(head_output)

    concat_output = np.concatenate(outputs, axis=1)

    W_output = np.random.randn(d_model, d_model)
    result = concat_output @ W_output.T

    return result