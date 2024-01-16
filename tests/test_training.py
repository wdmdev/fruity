"""Test code for train_model.py."""

from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from fruity.train import train

@patch('pytorch_lightning.Trainer')
@patch('pytorch_lightning.LightningModule')
@patch('pytorch_lightning.LightningDataModule')
@patch('pytorch_lightning.seed_everything')
@patch('hydra.utils.instantiate')
@patch('fruity.utils.instantiate_callbacks')
def test_train(mock_instantiate_callbacks,
               mock_instantiate,
               mock_seed_everything,
               mock_datamodule,
               mock_model,
               mock_trainer):
    """Test train function."""
    # Setup
    cfg = OmegaConf.create({'seed': 123,
                            'datamodule': {},
                            'model': {},
                            'callbacks': [],
                            'trainer': {},
                            'train': True,
                            'test': True})

    # Mocks
    mock_instantiate.return_value = mock_trainer
    mock_instantiate_callbacks.return_value = []
    mock_datamodule.return_value = MagicMock(spec=LightningDataModule)
    mock_model.return_value = MagicMock(spec=LightningModule)
    mock_trainer.return_value = MagicMock(spec=Trainer)

    # Exercise
    metric_dict, object_dict = train(cfg)

    # Verify
    mock_seed_everything.assert_called_once_with(123, workers=True)
    mock_instantiate.assert_called()
    mock_instantiate_callbacks.assert_called_once_with([])
    assert 'cfg' in object_dict
    assert 'datamodule' in object_dict
    assert 'model' in object_dict
    assert 'callbacks' in object_dict
    assert 'trainer' in object_dict

# @patch('fruity.train')
# @patch('fruity.utils.get_metric_value')
# def test_main(mock_get_metric_value, mock_train):
#     # Setup
#     cfg = OmegaConf.create({'optimized_metric': 'accuracy',
#                             'datamodule': {},})

#     # Mocks
#     mock_train.return_value = ({'accuracy': 0.9}, {})
#     mock_get_metric_value.return_value = 0.9

#     # Exercise
#     result = main(cfg)

#     # Verify
#     mock_train.assert_called_once_with(cfg)
#     mock_get_metric_value.assert_called_once_with(metric_dict={'accuracy': 0.9}, metric_name='accuracy')
#     assert result == 0.9
