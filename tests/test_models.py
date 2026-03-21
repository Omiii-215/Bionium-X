import pytest
import torch
from src.models.cnn_1d import CNN1DModel

def test_cnn1d_forward_pass():
    model = CNN1DModel(input_length=1000, num_classes=5)
    model.eval()
    
    # Batch size 2, 1 channel, 1000 sequence length
    dummy_input = torch.randn(2, 1, 1000)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    # Output should exactly predict 5 target molecules for each batch row
    assert output.shape == (2, 5), f"Expected output shape (2, 5), got {output.shape}"
    
    # Because of Softmax / Sigmoid implementation, outputs should be bounded [0, 1] if normalized
    # But usually logits are pushed through sigmoid implicitly if it's BCE, or Explicitly in eval
    output_np = output.numpy()
    assert (output_np >= 0).all() and (output_np <= 1.0).all(), "Predictions are out of probabilistic bounds [0.0, 1.0]"
