"""Test code for timm_model.py."""

import torch
from fruity.models.timm_model import create_model, TIMMModule



def test_create_model() -> None:
    """Test model creating function."""
    # Setup
    model_name = "resnet18"  # replace with a model name from timm
    input_ch = 3
    num_cls = 10

    # Exercise
    result = create_model(model_name, input_ch, num_cls)

    # Verify
    assert result is not None


def test_timm_module() -> None:
    """Test TIMMModule class."""
    # Setup
    model = create_model("resnet18", 3, 10)  # replace with a model name from timm
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    module = TIMMModule(model, optimizer)

    # Exercise and Verify
    assert module.net == model

    # Mock batch
    batch = (torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,)))

    # Test forward
    output = module.forward(batch[0])
    assert output.shape == torch.Size([16, 10])

    # Test step
    loss, preds, targets = module.step(batch)
    assert loss.shape == torch.Size([])
    assert preds.shape == targets.shape

    # Test training_step
    result = module.training_step(batch, 0)
    assert "loss" in result
    assert "preds" in result
    assert "targets" in result

    # Test validation_step
    result = module.validation_step(batch, 0)
    assert "loss" in result
    assert "preds" in result
    assert "targets" in result

    # Test test_step
    result = module.test_step(batch, 0)
    assert "loss" in result
    assert "preds" in result
    assert "targets" in result

    # # Test configure_optimizers
    # result = module.configure_optimizers()
    # assert 'optimizer' in result
