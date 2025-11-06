import numpy as np
from src.transformer import (
    computeRTG,
    heuristic_policy,
    _softmax,
    attention,
    multi_head_attention
)


def test_rtg():
    rewards = [1, 1, 1, 0]
    rtg = computeRTG(rewards)
    assert rtg == [3, 2, 1, 0], "RTG Code error"

def test_heuristic():
    state = [0.1, 0, 0, 0]
    u = heuristic_policy(state, K = 20)
    assert u == 2.0, "Heuristic policy error"

def test_attention():
    Q = np.array([1, 1, 0])
    K = np.array([
        [2,3,1],
        [1,1,0]
    ])
    V = np.array([
        [69, 420],
        [100, 200]
    ])


    normalise = _softmax(Q)
    result = attention(Q, K, V)

    assert np.sum(normalise) == 1.0, "Softmaxing is not adding upto 1"
    assert result.shape == (2,), "attention shape not matching up"

def test_multihead():
    x = np.random.randn(10, 8)
    output = multi_head_attention(x, head_count=4)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (10, 8), "Shape mismatch!"
    print("✓ Multi-head attention works!")