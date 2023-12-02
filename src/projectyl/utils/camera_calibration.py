import cv2 as cv
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
from projectyl.utils.io import Image, Dump
from matplotlib import pyplot as plt
import logging
from tqdm import tqdm


def getcorners(
    img_in: np.array,
    checkerboardsize: Tuple[int, int] = (10, 7),
    resize: Optional[Tuple[int, int]] = None,
    show: bool = False
) -> Tuple[bool, np.array, np.array, Tuple[int, int]]:
    img_overlay = img_in.copy()
    resize_factor = 1.
    grayorig = cv.cvtColor(img_overlay, cv.COLOR_BGR2GRAY)
    if resize is not None:
        resize_factor = img_in.shape[1] / resize[0]
    gray = grayorig if (resize is None or resize is False) else cv.resize(grayorig, resize)
    ret, corners = cv.findChessboardCorners(gray, checkerboardsize, None)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    if ret:
        corners = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    else:
        logging.warning("Chessboard not found")
    if ret:
        img_overlay = cv.drawChessboardCorners(
            img_overlay, checkerboardsize, corners*resize_factor, ret)
    if show:
        plt.imshow(img_overlay)
        plt.show()
    return ret, None if not ret else corners*resize_factor, img_overlay, gray.shape[::-1]


def camera_calibration(img_list: List[Path], resize_factor=0.5, output_folder: Path = None, debug=True):
    if output_folder is not None:
        output_folder.mkdir(parents=True, exist_ok=True)
    for idx, img_path in tqdm(enumerate(img_list), desc="Camera calibration", total=len(img_list)):
        img_path = Path(img_path)
        corner_path = (output_folder / (img_path.name + "_corners")).with_suffix(".yaml")
        corner_list = []
        if corner_path.exists():
            corners = Dump.load_yaml(corner_path)
            logging.debug(f"Skipping {img_path} - corners already computed")
            continue
        else:
            img = Image.load(img_path)
            h, w = img.shape[:2]
            if resize_factor is None:
                ds_size = None
            else:
                ds_size = (int(w*resize_factor), int(h*resize_factor))
            ret, corners, img_overlay, checkerboard_shape = getcorners(img, resize=ds_size, show=False)
            if debug:
                if img_overlay is not None:
                    Image.write(output_folder/img_path.name, img_overlay)
            if ret:
                Dump.save_yaml(corners.tolist(), corner_path)
        corner_list.append(corners)
        # if idx > 20:
        #     break
