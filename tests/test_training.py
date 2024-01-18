"""Test code for train_model.py."""

from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from fruity.train import train


@patch("pytorch_lightning.Trainer")
@patch("pytorch_lightning.LightningModule")
@patch("pytorch_lightning.LightningDataModule")
@patch("pytorch_lightning.seed_everything")
@patch("hydra.utils.instantiate")
@patch("fruity.utils.instantiate_callbacks")
def test_train(
    mock_instantiate_callbacks: MagicMock,
    mock_instantiate: MagicMock,
    mock_seed_everything: MagicMock,
    mock_datamodule: MagicMock,
    mock_model: MagicMock,
    mock_trainer: MagicMock,
) -> None:
    """Test train function.

    Args:
    ----
        mock_instantiate_callbacks (MagicMock): Mock of instantiate_callbacks.
        mock_instantiate (MagicMock): Mock of instantiate.
        mock_seed_everything (MagicMock): Mock of seed_everything.
        mock_datamodule (MagicMock): Mock of LightningDataModule.
        mock_model (MagicMock): Mock of LightningModule.
        mock_trainer (MagicMock): Mock of Trainer.

    """
    # Setup
    cfg = OmegaConf.create(
        {"seed": 123, "datamodule": {}, "model": {}, "callbacks": [],
         "trainer": {}, "train": True, "test": True, "logging": {}}
    )

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
    assert "cfg" in object_dict
    assert "datamodule" in object_dict
    assert "model" in object_dict
    assert "callbacks" in object_dict
    assert "trainer" in object_dict
