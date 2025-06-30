import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
import tkinter as tk
from gui_prediction_app import PredictionApp
from unittest.mock import patch


class TestPredictionApp(unittest.TestCase):

    def setUp(self):
        self.root = tk.Tk()
        self.app = PredictionApp(self.root)
        self.tmp_dir = Path("temp_test_rasters")
        self.tmp_dir.mkdir(exist_ok=True)

        # Patch messagebox.showerror globally
        self.patcher = patch("tkinter.messagebox.showerror")
        self.mock_showerror = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        for f in self.tmp_dir.glob("*.tif"):
            f.unlink()
        self.tmp_dir.rmdir()
        self.root.destroy()

    def test_validate_input_raster_valid_uint8(self):
        valid_raster = self.tmp_dir / "valid_uint8.tif"
        data = np.random.randint(0, 255, (4, 10, 10)).astype(np.uint8)

        with rasterio.open(
            valid_raster,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=4,
            dtype="uint8",
            crs="EPSG:2230",
            transform=from_origin(0, 0, 0.5, 0.5),
        ) as dst:
            dst.write(data)

        self.assertTrue(self.app.validate_input_raster(valid_raster))
        self.mock_showerror.assert_not_called()

    def test_validate_input_raster_invalid_float32(self):
        invalid_raster = self.tmp_dir / "invalid_float32.tif"
        data = np.random.rand(4, 10, 10).astype(np.float32)

        with rasterio.open(
            invalid_raster,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=4,
            dtype="float32",
            crs="EPSG:2230",
            transform=from_origin(0, 0, 0.5, 0.5),
        ) as dst:
            dst.write(data)

        self.assertFalse(self.app.validate_input_raster(invalid_raster))
        self.mock_showerror.assert_called()

    def test_validate_input_raster_invalid_crs(self):
        invalid_raster = self.tmp_dir / "invalid_crs.tif"
        data = np.random.randint(0, 255, (4, 10, 10)).astype(np.uint8)

        with rasterio.open(
            invalid_raster,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=4,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(0, 0, 0.5, 0.5),
        ) as dst:
            dst.write(data)

        self.assertFalse(self.app.validate_input_raster(invalid_raster))
        self.mock_showerror.assert_called()

    def test_validate_input_raster_invalid_resolution(self):
        invalid_raster = self.tmp_dir / "invalid_res.tif"
        data = np.random.randint(0, 255, (4, 10, 10)).astype(np.uint8)

        with rasterio.open(
            invalid_raster,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=4,
            dtype="uint8",
            crs="EPSG:2230",
            transform=from_origin(0, 0, 1.0, 1.0),  # invalid resolution
        ) as dst:
            dst.write(data)

        self.assertFalse(self.app.validate_input_raster(invalid_raster))
        self.mock_showerror.assert_called()

    def test_validate_input_raster_invalid_bandcount(self):
        invalid_raster = self.tmp_dir / "invalid_bands.tif"
        data = np.random.randint(0, 255, (2, 10, 10)).astype(np.uint8)

        with rasterio.open(
            invalid_raster,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=2,
            dtype="uint8",
            crs="EPSG:2230",
            transform=from_origin(0, 0, 0.5, 0.5),
        ) as dst:
            dst.write(data)

        self.assertFalse(self.app.validate_input_raster(invalid_raster))
        self.mock_showerror.assert_called()


if __name__ == "__main__":
    unittest.main()
