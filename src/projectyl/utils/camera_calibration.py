import cv2 as cv
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
from projectyl.utils.io import Image, Dump
from matplotlib import pyplot as plt
import logging
from tqdm import tqdm
from projectyl.utils.camera_projection import get_focal_from_full_frame_equivalent, rescale_focal, get_intrinic_matrix
from projectyl.video.props import INTRINSIC_MATRIX


def getcorners(
    img_in: np.array,
    checkerboardsize: Tuple[int, int] = (10, 7),
    resize: Optional[Tuple[int, int]] = None,
    decimate_points: int = 5,
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
        logging.debug("Chessboard not found")
    if ret:
        img_overlay = cv.drawChessboardCorners(
            img_overlay, checkerboardsize, corners*resize_factor, ret)
    if show:
        plt.imshow(img_overlay)
        plt.show()
    return ret, None if not ret else corners*resize_factor, img_overlay, gray.shape[::-1]


def camera_calibration(
    img_list: List[Path],
    resize_factor=0.5,
    output_folder: Path = None,
    debug=True,
    decimate: int = 5,
    checkerboardsize: Tuple[int, int] = (10, 7),
):
    if output_folder is not None:
        output_folder.mkdir(parents=True, exist_ok=True)
    # Save calibration
    cam_calib_path = output_folder/"camera_calibration.json"
    if cam_calib_path.exists():
        calib_dict = Dump.load_json(cam_calib_path)
        logging.debug(f"Camera calibration found {calib_dict}")
        return np.array(calib_dict[INTRINSIC_MATRIX])
    corner_list = []
    for idx, img_path in tqdm(enumerate(img_list), desc="Camera calibration", total=len(img_list)):
        img_path = Path(img_path)
        corner_path = (output_folder / (img_path.name + "_corners")).with_suffix(".json")
        if corner_path.exists():
            corners = Dump.load_json(corner_path)
        else:
            img = Image.load(img_path)
            h, w = img.shape[:2]
            if resize_factor is None:
                ds_size = None
            else:
                ds_size = (int(w*resize_factor), int(h*resize_factor))
            ret, corners, img_overlay, checkerboard_shape = getcorners(
                img, resize=ds_size, show=False, checkerboardsize=checkerboardsize)
            if not ret:
                corners = []
            if debug:
                if img_overlay is not None:
                    Image.write(output_folder/img_path.name, img_overlay)
            Dump.save_json([] if not ret else corners.tolist(), corner_path)
        corner_list.append(corners)
    corner_list_non_empty = [np.array(c).reshape(-1, 1, 2).astype(np.float32) for c in corner_list if len(c) > 0]
    # Remove a lot of corners to speed up calibration
    if decimate > 1:
        corner_list_non_empty = corner_list_non_empty[::decimate]
    img = Image.load(Path(img_list[0]))
    h, w = img.shape[:2]
    del img
    objp = np.zeros((checkerboardsize[1]*checkerboardsize[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboardsize[0], 0:checkerboardsize[1]].T.reshape(-1, 2)
    objp = objp.astype(np.float32)
    objpoints = []  # 3d point in real world space
    for _idx in range(len(corner_list_non_empty)):
        objpoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, corner_list_non_empty, (w, h), None, None)
    # Numerical sanity check
    expected_focal = rescale_focal(
        get_focal_from_full_frame_equivalent(),
        w_resized=max(w, h)*1.3  # Video mode crops around 30% margin for stabilization
    )
    theoretical_intrinsic = get_intrinic_matrix((h, w), fpix=expected_focal)
    print(f"{mtx}\n{theoretical_intrinsic}")
    Dump.save_json({INTRINSIC_MATRIX: mtx.tolist()}, cam_calib_path)
    return mtx
