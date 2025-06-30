import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from pre_trained_model import PreTrainedModel


class TestPreTrainedModel(unittest.TestCase):
    @patch("tensorflow.keras.models.load_model")
    def test_prepare_tile(self, mock_load_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros((1, 256, 256, 3))
        mock_load_model.return_value = mock_model

        model = PreTrainedModel("dummy_path")
        img = np.random.rand(4, 256, 256).astype(np.float32)
        pre_img = model.prepare_tile(img.copy())
        self.assertEqual(pre_img.shape, (1, 256, 256, 4))

    @patch("tensorflow.keras.models.load_model")
    def test_predict(self, mock_load_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros((1, 256, 256, 3))
        mock_load_model.return_value = mock_model

        model = PreTrainedModel("dummy_path")
        img = np.random.rand(4, 256, 256).astype(np.float32)
        preds = model.predict(img)
        self.assertEqual(preds.shape[0], 1)

    @patch("tensorflow.keras.models.load_model")
    def test_prepare_tile_batch_and_predict_batch(self, mock_load_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros((2, 256, 256, 3))
        mock_load_model.return_value = mock_model

        model = PreTrainedModel("dummy_path")
        imgs = [np.random.rand(4, 256, 256).astype(np.float32) for _ in range(2)]
        batch = model.prepare_tile_batch(imgs)
        self.assertEqual(batch.shape, (2, 256, 256, 4))

        preds = model.predict_batch(imgs)
        self.assertEqual(preds.shape[0], 2)
