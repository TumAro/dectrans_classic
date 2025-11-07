from typing import List, Tuple
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

# ============================================================================

def _softmax(M: np.ndarray) -> np.ndarray:
    b = np.max(M)
    e_x = np.exp(M - b)
    return e_x / np.sum(e_x)

def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    """
    Q: Qeury Matrix
    K: Key Matrix
    V: Value Matrix
    """
    result = np.dot(Q, K.T)
    normalised = _softmax(result)
    output = np.dot(normalised, V)

    return output

def multi_head_attention(x: np.ndarray, head_count=4):
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

# ============================================================================
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# x -> (token_number, dimension)
# Q,K,V -> (dimension, dimension)
# Attention(Q,K,V) -> (token_number, dimension)

class Attention(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.W_q = nn.Linear(dimension, dimension)
        self.W_k = nn.Linear(dimension, dimension)
        self.W_v = nn.Linear(dimension, dimension)


    def forward(self, x: torch.Tensor):
        """
        x:          (batch, seq_len, dimension) tensors
        output:     (batch, seq_len, dimension)
        """
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        query_match = Q @ K.transpose(-2, -1)
        similarities = F.softmax(query_match, dim=-1)
        output = similarities @ V
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dimension, head_count):
        super().__init__()

        self._d_head = dimension // head_count

        self._heads = nn.ModuleList(
            [Attention(self._d_head) for _ in range(head_count)]
        )
        self.W_o = nn.Linear(dimension, dimension)
        

    def forward(self, x: torch.Tensor):
        """
        x:    (batch, seq_len, dimension) tensors
        output:     (batch, seq_len, dimension)
        """

        concat= torch.cat(
            [head(x[:, :, i*self._d_head:(i+1)*self._d_head]) for i, head in enumerate(self._heads)],
            dim = -1
        )
        result = self.W_o(concat)
        return result

class Transformer(nn.Module):
    def __init__(self, dimension, head_count):
        super().__init__()

        self._attention = MultiHeadAttention(dimension, head_count)

        # * For MLP Pass
        self._norm_1 = nn.LayerNorm(dimension)
        self._mlp_1 = nn.Linear(dimension, dimension*4)
        self._mlp_2 = nn.Linear(dimension*4, dimension)
        self._norm_2 = nn.LayerNorm(dimension)



    def forward(self, x: torch.Tensor):
        """
        input:
            x -> (batch, seq_len, dimension) tensors
        output:

        """

        y = self._attention(x) + x      # * Multi Head Attention

        y_norm = self._norm_1(y)        # * Layer Norm

        y = self._mlp_1(y_norm)         # * Hidden Layer 1 | dimension -> dimension*4
        y = F.gelu(y)                   # * normalis
        y = self._mlp_2(y) + y_norm     # * Hidden Layer 2 | dimension*4 -> dimension

        y_norm = self._norm_2(y)        # * Layer Norm -> output

        return y_norm

# ============================================================================
# ============================================================================

class Tokenisation(nn.Module):
    def __init__(self, dimension: Tuple[int, int, int]) -> None:
        super().__init__()

        state_dim, action_dim, rtg_dim = dimension

        self.embed_state = nn.Linear(state_dim, 32)
        self.embed_action = nn.Linear(action_dim, 32)
        self.embed_rtg = nn.Linear(rtg_dim, 32)

        self.pos_encoding = nn.Embedding(20 * len(dimension), 32)

    def forward(self, states, actions, rtg):

        y_state = self.embed_state(states)
        y_action = self.embed_action(actions)
        y_rtg = self.embed_rtg(rtg)

        stacked = torch.stack([y_rtg, y_action, y_state], dim=2)
        interleaved = stacked.reshape(stacked.shape[0], -1, 32)

        seq_len = interleaved.shape[1]
        positions = torch.arange(seq_len)

        pos_embed = self.pos_encoding(positions)
        return interleaved + pos_embed

class Pipeline(nn.Module):
    def __init__(self, dimension: Tuple[int, int, int], head_count, layer_count = 2) -> None:
        super().__init__()

        self._tokeniser = Tokenisation(dimension)
        self._transformers = nn.ModuleList([
            Transformer(32, head_count) for _ in range(layer_count)
        ])

        self._output_layer = nn.Linear(32,1)

    def forward(self, states, actions, rtg):
        y = self._tokeniser(states, actions, rtg)
        for transformer in self._transformers:
            y = transformer(y)

        state_tokens = y[:, 1::3, :]
        output = self._output_layer(state_tokens)

        return output