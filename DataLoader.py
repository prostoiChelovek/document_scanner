from typing import Optional

import cv2 as cv

import tempfile

import numpy as np

try:
    from pdf2image import convert_from_path

    class TestPdfLoader:
        def __init__(self, path: str, rotate: Optional[int] = None,
                     *args, **kwargs):
            self.rotate: Optional[int] = rotate

            kwargs["paths_only"] = True
            kwargs["fmt"] = "png"

            self._temp_directory = tempfile.TemporaryDirectory()
            kwargs["output_folder"] = self._temp_directory.name

            self._pdf = convert_from_path(path, *args, **kwargs)

        def __del__(self):
            self._temp_directory.cleanup()

        def __iter__(self):
            return self

        def __next__(self) -> np.array:
            for img_path in self._pdf:
                img = cv.imread(img_path)
                if self.rotate:
                    img = cv.rotate(img, self.rotate)

                return img

            raise StopIteration()
except ModuleNotFoundError:
    pass


class TestImagesLoader:
    def __init__(self, directory: str, rotate: Optional[int] = None):
        self.directory = directory

        self.rotate: Optional[int] = rotate

        self._current_img = 1

    def __iter__(self):
        self._current_img = 1
        return self

    def __next__(self) -> np.array:
        filename = f"{self.directory}/{self._current_img:02d}.png"

        img: Optional[np.array] = cv.imread(filename)
        if img is None or img.size == 0:
            raise StopIteration()

        if self.rotate:
            img = cv.rotate(img, self.rotate)

        self._current_img += 1

        return img
