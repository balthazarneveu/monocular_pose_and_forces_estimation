
import numpy as np
from typing import Tuple


def get_4D_homogeneous_vector(pos: np.ndarray) -> np.ndarray:
    """Permute coordinates and add a 1 at the end

    Args:
        pos (np.ndarray): [3] 3D position in the world coordinate system.

    Returns:
        np.ndarray: [4, 1] 3D position in permuted-camera-like coordinate system.
    ```
    # World coordinate system
    =========================
    Z (blue)
    ^
    |   . Y   (green)
    |  /
    | /
    . -------> X (red).



    Image arrays , OpenCV referential
    =================================

       . Z (optical axis)
      /
     /
    .------> X (horizontal) - columns
    |
    |
    v Y (vertical) - rows
    ```

    NOTE:
    =====
    This could be done by incorporating a permutation
    to the camera extrinsic matrix
    """

    return np.array([pos[0], -pos[2], pos[1], 1.]).reshape(4, 1)


def get_intrinic_matrix(img_shape: Tuple[int], fpix: float = 600.) -> np.ndarray:
    """Pinhole camera model `K` matrix.

    Get the intrinic matrix of the camera based on the image shape and focal length

    Args:
        img_shape (Tuple[int]): image shape (height, width) to compute a rough
        estimation of the optical center projection on the sensor.
        fpix (float, optional): Focal length expressed in pixels.
        Computed as a metric focal length/pixel pitch
        Defaults to 600..

    Returns:
        np.ndarray: [3,3] `K` camera matrix
    """
    intrinic_matrix = np.diag([fpix, fpix, 1.])
    intrinic_matrix[0, 2] = float(img_shape[1])/2.
    intrinic_matrix[1, 2] = float(img_shape[0])/2.
    return intrinic_matrix


def project_3D_point(
        pos: np.ndarray,
        intrinsic_matrix: np.ndarray,
        extrinsic_matrix: np.ndarray) -> np.ndarray:
    """Project a 3D point to the image plane

    Args:
        pos (np.ndarray): [3] 3D position in the world coordinate system.
        intrinsic_matrix (np.ndarray):  [3,3] `K` camera matrix
        extrinsic_matrix (np.ndarray): [4,3] `R|t` camera extrinsic matrix

    Returns:
        np.ndarray: [2] 2D coordinates in the image plane expressed in pixel coordinates.

    Pinhole camera geometry
    =======================
    From 3D metric coordinates to 2D pixel coordinates.
    Distortion is not considered here.
    For more details on projective geometry see:
    Hartley-Zisserman book https://www.robots.ox.ac.uk/~vgg/hzbook/
    """
    pos = get_4D_homogeneous_vector(pos)
    ext_vec = extrinsic_matrix.dot(pos)
    pos2d = intrinsic_matrix.dot(ext_vec)
    pos2d = pos2d[:2, 0] / pos2d[2, 0]
    return pos2d

# Utilities to get realistic camera parameters


def get_focal_from_full_frame_equivalent(
    focal_length_equivalent_24x36=0.024,
    w: int = 4000,  # image width [pixels]
    # h: int = 3000,
    pixel_pitch: float = 2.*1.4E-6  # 12Mpix (binned quadbayers)
) -> float:
    """Get the right focal in pixels for a given sensor size
    and 24x36mm full frame focal length equivalent
    Example for Xiaomi Mi 11 Ultra photo 12Mpix

    Args:
        focal_length_equivalent_24x36 (float, optional): 24mm full frame equivalent. Defaults to 0.024.
        w (int, optional): Image width 4000x3000=12Mpix -> 4000. Defaults to 4000.
        pixel_pitch (float): Size in meters (2.8µm = 2*1.4µm after binning)

    Returns:
        float: focal in pixels, ready to insert in the diagonals of the intrinic matrix
        ~ 2666 pixels
    """
    sensor_w = w*pixel_pitch
    # sensor_h = h*pixel_pitch
    full_frame_36mm_w = 36E-3
    # full_frame_24mm_h = 24E-3
    focal = focal_length_equivalent_24x36 * sensor_w / full_frame_36mm_w
    fpix = focal/pixel_pitch
    return fpix  # ~2600


def rescale_focal(
    fpix: float = get_focal_from_full_frame_equivalent(),
    w: int = 4000,  # full original size
    w_resized: int = 1920  # downscaled size
) -> float:
    """Provide the right focal length for a downsampled image
    Resizing the image will change the focal length in pixels

    Args:
        fpix (float, optional): Focal length in pixels. Defaults to get_focal_from_full_frame_equivalent().
        w (int, optional): _description_. Defaults to 4000. for 12Mpix photo mode
        w_resized (int, optional): Resized video size. Defaults to 1920 for Full HD 1080p.

    Returns:
        float: Resized focal length in pixels
    """
    virtual_pixel_pitch_ratio = w/w_resized  # Virtual equivalent bigger pixels pitch -> 4000/1920
    fpix_resized = fpix/virtual_pixel_pitch_ratio
    return fpix_resized

