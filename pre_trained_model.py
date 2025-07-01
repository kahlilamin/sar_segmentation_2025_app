import numpy as np
import tensorflow as tf
from typing import Union, Optional
from pathlib import Path

tf.get_logger().setLevel("ERROR")


class PreTrainedModel:
    """
    Wrapper class for loading a trained TensorFlow segmentation model
    and performing predictions on 4-band imagery tiles (e.g., from rasterio).

    Assumptions:
    - Each image tile has shape (4, 256, 256): 4 channels, height, width.
    - The model was trained with a SEResNet34 backbone.
    - Input expected by the model is shape (None, 256, 256, 4) with float values scaled to [0, 255].
    """

    def __init__(self, model_path: Union[str, Path]) -> None:
        """
        Load the trained TensorFlow model.

        Args:
            model_path (str or Path): Path to a SavedModel directory or .h5 file.
        """
        self.model_path = model_path

        self._model = None

    def load(self) -> None:
        if self._model is None:
            self._model = tf.keras.models.load_model(self.model_path, compile=False)

    @property
    def model(self) -> Optional[tf.keras.Model]:
        """
        Load and return the TensorFlow model.

        Returns:
            tf.keras.Model: The loaded model.
        """
        if self._model is None:
            self.load()
        return self._model

    def prepare_tile(self, img) -> np.ndarray:
        """
        Preprocess a single input tile for prediction.

        Steps:
        - Normalize each channel independently to 8-bit range [0, 255]
        - Add batch dimension (1, 4, H, W)
        - Transpose to TensorFlow format (1, H, W, 4)

        Args:
            img (np.ndarray): Input image tile of shape (4, 256, 256)

        Returns:
            np.ndarray: Preprocessed input with shape (1, 256, 256, 4)
        """
        # Normalize pixel values for each of the 4 bands to 8-bit range
        for i in range(4):
            img[i] = (img[i] * 255.0 / img[i].max()).astype(np.float16)

        # Add new dimension 4x256x256 -> 1x4x256x256
        rgbi = np.expand_dims(img, axis=0)

        # Re-arrange dimensions to 1x256x256x4. TF needs input in batches
        tf_rgbi = tf.transpose(rgbi, perm=[0, 2, 3, 1])

        # Note: segmentation-models seresnet does not require preprocessing
        # like normalization or scaling. The raw pixel values are used directly.

        return tf_rgbi

    def predict(self, img: np.ndarray) -> np.ndarray:
        """
        Run model inference on a single image tile.

        Args:
            img: Image of shape (4, 256, 256)

        Returns:
            Prediction output from the model
        """
        pre_img = self.prepare_tile(img)

        return self.model.predict(pre_img)

    def prepare_tile_batch(self, imgs: np.ndarray) -> np.ndarray:
        """
        Preprocess a batch of image tiles.

        Args:
            imgs: List or array of shape (B, 4, 256, 256)

        Returns:
            Preprocessed batch of shape (B, 256, 256, 4)
        """
        prepared_imgs = [self.prepare_tile(img) for img in imgs]

        return np.vstack(prepared_imgs)

    def predict_batch(self, imgs) -> np.ndarray:
        """
        Run model inference on a batch of image tiles.

        Args:
            imgs: Batch of images with shape (B, 4, 256, 256)

        Returns:
            Model predictions for the entire batch
        """
        batch_input = self.prepare_tile_batch(imgs)
        return self.model.predict(batch_input)
