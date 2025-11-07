import torch
import pytest
from src.transformer import Attention, MultiHeadAttention

def test_attention_output_shape():
    dimension = 128
    batch_size = 2
    seq_len = 10

    attn = Attention(dimension)
    x = torch.randn(batch_size, seq_len, dimension)
    output = attn(x)

    assert output.shape == (2, 10, 128), "Shape mismatch"


def test_attention_forward_pass():
    attn = Attention(64)
    x = torch.randn(1, 5, 64)
    output = attn(x)
    assert output is not None, "Error at forward pass"

def test_attention_parameters_learnable():
    """Test that parameters have gradients"""
    attn = Attention(32)
    x = torch.randn(1, 3, 32, requires_grad=True)
    output = attn(x)
    loss = output.sum()
    loss.backward()

    assert attn.W_q.weight.grad is not None, "No gradient found"

def test_multihead_attention():
    """Test multi-head attention"""
    d_model = 128
    num_heads = 4
    batch_size = 2
    seq_len = 10
    
    mha = MultiHeadAttention(dimension=d_model, head_count=num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    output = mha(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    print(f"✓ MultiHeadAttention works! {x.shape} → {output.shape}")
