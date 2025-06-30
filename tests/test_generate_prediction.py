import unittest
from unittest.mock import MagicMock, patch
from affine import Affine
from rasterio.windows import Window
from generate_prediction import get_crop_window, get_tiles


class TestGeneratePrediction(unittest.TestCase):
    def test_get_crop_window_valid(self):
        win = Window(0, 0, 256, 256)
        cropped = get_crop_window(win, crop_amount=64)
        self.assertEqual(cropped.width, 128)
        self.assertEqual(cropped.height, 128)

    def test_get_crop_window_invalid(self):
        win = Window(0, 0, 128, 128)
        with self.assertRaises(ValueError):
            get_crop_window(win, crop_amount=64)

    @patch("rasterio.open")
    def test_get_tiles(self, mock_rasterio_open):
        mock_src = MagicMock()
        mock_src.meta = {"width": 512, "height": 512}

        # Provide a valid rasterio Affine transform
        mock_src.transform = Affine.translation(0, 0) * Affine.scale(1, -1)

        tiles = list(get_tiles(mock_src, width=256, height=256, stride=256))
        self.assertGreater(len(tiles), 0)
