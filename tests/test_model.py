# import os

# import torch

# from fruity.models.timm_model import TIMMModule

# class TestTimmModel:
#     """Tests for the TIMMModule class."""

#     def test_init(self) -> None:
#         """Test initialization of TIMMModule."""
#         net = "resnet18"
#         optimizer = torch.optim.Adam
#         # Act
#         model = TIMMModule(net, optimizer)

#         # Assert
#         assert model.hparams["net"] == net
#         assert model.hparams["optimizer"] == optimizer
#         assert model.net is not None
#         assert model.criterion is not None
#         assert model.train_acc is not None
#         assert model.val_acc is not None
#         assert model.test_acc is not None
#         assert model.train_loss is not None
#         assert model.val_loss is not None
#         assert model.test_loss is not None
#         assert model.val_acc_best is not None
#         assert model.predict_transform is not None

#     def test_forward(self) -> None:
#         """Test forward pass through the network with size 100x100."""
#         model = TIMMModule("resnet18", torch.optim.Adam)
#         x = torch.randn(1, 3, 100, 100)

#         # Act
#         out = model(x)

#         # Assert
#         assert out.shape == (1, 1000)


import torch
from unittest.mock import Mock
from fruity.models.timm_model import create_model, TIMMModule

def test_create_model():
    # Setup
    model_name = 'resnet18'  # replace with a model name from timm
    input_ch = 3
    num_cls = 10

    # Exercise
    result = create_model(model_name, input_ch, num_cls)

    # Verify
    assert result is not None

def test_timm_module():
    # Setup
    model = create_model('resnet18', 3, 10)  # replace with a model name from timm
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
    assert 'loss' in result
    assert 'preds' in result
    assert 'targets' in result

    # Test validation_step
    result = module.validation_step(batch, 0)
    assert 'loss' in result
    assert 'preds' in result
    assert 'targets' in result

    # Test test_step
    result = module.test_step(batch, 0)
    assert 'loss' in result
    assert 'preds' in result
    assert 'targets' in result

    # # Test configure_optimizers
    # result = module.configure_optimizers()
    # assert 'optimizer' in result