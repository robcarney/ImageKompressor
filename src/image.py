import cv2
import numpy as np
from enum import Enum

class ImageDataFormat(Enum):
    """ImageDataFormat is an enum for specifying the format of image data you want.
    """
    DEFAULT = 1
    NORMALIZED = 2
    FLATTENED = 3
    FLATTENED_NORMALIZED = 4


class Image:
    """Image represents an image as a numpy array of data.
    """
    def __init__(self, image_path=None, image_data=np.empty(())):
        """Default Image constructor

        Args:
            image_path: file path for the image
            image_data: data for the image
        """
        if not image_data.any() and not image_path:
            raise ValueError("One of image_path or image_data must be present")
        elif image_path:
            img = cv2.imread(image_path)
        else:
            img = image_data
        if not img.any():
            print("Something went wrong when reading the image. Please ensure the file location is correct")
        self.data = img

    def shape(self):
        """shape retrieves the shape of the Image.

        Returns:
            Shape of the Image as a tuple.
        """
        return self.data.shape

    def raw_data(self, data_format=ImageDataFormat.DEFAULT):
        """raw_data returns the data for the Image, formatted as specified.

        Args:
            data_format: One of the ImageDataFormat enums, specifying the desired format.
        """
        if data_format == ImageDataFormat.DEFAULT:
            return self.data
        elif data_format == ImageDataFormat.FLATTENED:
            return self.data.flatten()
        elif data_format == ImageDataFormat.NORMALIZED:
            return self.data.astype(np.float32) / 256.
        else:
            return self.data.flatten().astype(np.float32) / 256.







