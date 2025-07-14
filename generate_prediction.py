from dataclasses import dataclass
from pathlib import Path
import sys
import time

from affine import Affine
import numpy as np

import rasterio
from rasterio import windows
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

from pre_trained_model import PreTrainedModel

# Pyinstaller compatibility
# If running as a PyInstaller bundle, use the _MEIPASS attribute to find the base path
# Otherwise, use the current file's directory as the base path
if hasattr(sys, "_MEIPASS"):
    base_path = Path(sys._MEIPASS)
else:
    base_path = Path(__file__).parent

PRE_TRAINED_MODELS = [
    PreTrainedModel(
        model_path=base_path / "data" / "models" / "model_1" / "saved_model"
    ),
    PreTrainedModel(
        model_path=base_path / "data" / "models" / "model_2" / "saved_model"
    ),
    PreTrainedModel(
        model_path=base_path / "data" / "models" / "model_3" / "saved_model"
    ),
]


@dataclass
class Result:
    batch_size: int
    processing_time: int


def reclassify(arr):
    reclass_map = {0: 5, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 3, 7: 4, 8: 3, 9: 3}

    return np.vectorize(reclass_map.get)(arr)


def get_crop_window(window, crop_amount):
    """
    Return the rasterio window for the current tile.
    """
    if 2 * crop_amount >= window.height or 2 * crop_amount >= window.width:
        raise ValueError(
            f"Invalid crop size={crop_amount}. Results in window {window.height} with zero or negative area."
        )

    col_off = window.col_off + crop_amount
    row_off = window.row_off + crop_amount
    width = height = 2 * crop_amount

    return windows.Window(col_off=col_off, row_off=row_off, width=width, height=height)


def get_tiles(src, width: int = 256, height: int = 256, stride: int = 256):
    ncols, nrows = src.meta["width"], src.meta["height"]

    window_offsets = [
        (col, row)
        for col in range(0, ncols, stride)
        for row in range(0, nrows - 128, stride)
    ]

    overall_window = windows.Window(col_off=0, row_off=0, height=nrows, width=ncols)

    for col_off, row_off in window_offsets:
        window = windows.Window(
            col_off=col_off, row_off=row_off, width=width, height=height
        ).intersection(overall_window)

        transform = windows.transform(window, src.transform)

        yield window, transform


def generate_prediction(
    src,
    profile,
    out_prediction_tif: Path,
    models,
    windows: list[windows.Window],
    tile_size: int = 256,
    stride: int = 256,
    batch_size: int = 4,
    progress_callback=None,
    reclassify_values: bool = False,
):

    cell_height = cell_width = tile_size
    tif_profile = {
        "driver": "GTiff",
        "count": 1,
        "height": profile["height"],
        "width": profile["width"],
        "dtype": "uint8",
        "crs": src.profile["crs"],
        "transform": profile["transform"],
        "nodata": 255,
    }

    with rasterio.open(out_prediction_tif, "w", **tif_profile) as tile_dst:
        with WarpedVRT(src, **profile) as vrt:
            batch_imgs = []
            batch_windows = []

            for window in windows:
                if window.height != cell_height or window.width != cell_width:
                    continue

                tile_img = vrt.read((1, 2, 3, 4), window=window)

                if np.average(tile_img) == profile["nodata"]:
                    continue

                batch_imgs.append(tile_img)
                batch_windows.append(window)

                if len(batch_imgs) == batch_size:
                    batch_preds = []
                    for model in models:
                        preds = model.predict_batch(batch_imgs)
                        batch_preds.append(preds)

                    avg_preds = np.mean(batch_preds, axis=0)
                    batch_argmax = np.argmax(avg_preds, axis=3)

                    for i in range(batch_size):
                        crop_window = get_crop_window(batch_windows[i], crop_amount=64)
                        pred_crop = batch_argmax[i][64:192, 64:192]

                        if reclassify_values:
                            pred_crop = reclassify(pred_crop)

                        tile_dst.write(pred_crop, window=crop_window, indexes=1)

                        if progress_callback:
                            progress_callback()

                    batch_imgs.clear()
                    batch_windows.clear()

            if batch_imgs:
                batch_preds = []
                for model in models:
                    preds = model.predict_batch(batch_imgs)
                    batch_preds.append(preds)

                avg_preds = np.mean(batch_preds, axis=0)
                batch_argmax = np.argmax(avg_preds, axis=3)

                for i in range(len(batch_imgs)):
                    crop_window = get_crop_window(batch_windows[i], crop_amount=64)
                    pred_crop = batch_argmax[i][64:192, 64:192]

                    if reclassify_values:
                        pred_crop = reclassify(pred_crop)

                    tile_dst.write(pred_crop, window=crop_window, indexes=1)

                    if progress_callback:
                        progress_callback()
