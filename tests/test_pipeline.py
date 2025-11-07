from src.transformer import Pipeline
import torch

def test_pipeline():
    dimension = (4,1,1)
    head_count = 4
    layer_count = 2

    pipe = Pipeline(dimension, head_count, layer_count)

    state = torch.randn(2, 20, 4)
    action = torch.randn(2, 20, 1)
    rtg = torch.randn(2, 20, 1)

    output = pipe(state, action, rtg)

    assert output.shape == (2, 20, 1), "shape mismatch"