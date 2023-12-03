import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.special import huber
from projectyl.utils.properties import ELBOW, SHOULDER, WRIST, RIGHT, LEFT
from projectyl.video.props import INTRINSIC_MATRIX, THUMBS, SIZE
from pathlib import Path
from projectyl.dynamics.inverse_kinematics import forward_kinematics
from projectyl.utils.arm import retrieve_arm_estimation
from projectyl.utils.camera_projection import project_3D_point
from projectyl.utils.pose_overlay import get_4D_homogeneous_vector
from projectyl.utils.io import Dump
from projectyl import root_dir
from projectyl.utils.camera_projection import get_4D_homogeneous_vector
from interactive_pipe.helper import _private
from scipy.optimize import least_squares
from typing import List, Tuple
from projectyl import root_dir
from projectyl.dynamics.armmodel import ArmRobot
from matplotlib import pyplot as plt
import logging
DEFAULT_CAMERA_CALIBRATION = root_dir/"calibration"/"camera_calibration_xiaomi_mi11_ultra_video_vertical.json"


def get_camera_pose_fit_data_from_files(
    video_path: Path,
    camera_calibration_path: Path = DEFAULT_CAMERA_CALIBRATION,
    ik_seq: str = "coarse_ik.pkl",
    pose_seq: str = "pose/pose_sequence.pkl",
    config_file: str = "config.yaml",
) -> tuple[List[dict], List[dict], np.ndarray]:
    """"Get 2D and 3D data from preprocessed files
    """
    data3d = Dump.load_pickle(video_path/ik_seq)
    data2d = Dump.load_pickle(video_path/pose_seq)
    calib_dict = Dump.load_json(camera_calibration_path)
    intrinsic_matrix = np.array(calib_dict[INTRINSIC_MATRIX])
    config = Dump.load_yaml(video_path/config_file, safe_load=False)
    h, w = int(config[THUMBS][SIZE][0]), int(config[THUMBS][SIZE][1])
    # logging.warning(f"Video size: {w}x{h}")
    assert h == 1920 and w == 1080, "Video size is not 1920x1080 - need to recompute camera calibration"
    return data3d, data2d, intrinsic_matrix, config


def build_3d_2d_data_arrays(data3d: List[dict], data2d: List[dict], h_w_size: Tuple[int, int], arm_side: str = RIGHT) -> tuple[np.ndarray, np.ndarray]:
    # INITIAL STATE LIST
    h, w = h_w_size
    q_states = np.array(data3d["q"])

    # 3D points
    p = ([
        np.array([el.translation for el in data3d["3dpoints"][member]]).T for member in [SHOULDER, ELBOW, WRIST]])
    gt_p3d_full = np.concatenate(p).T  # T, 9
    # Standardized arm 3D points -> not the results of IK

    # Retrieve mediapipe joints 2D points
    arm_2d = [retrieve_arm_estimation(data2d, frame_idx=t, arm_side=arm_side, key="pose_landmarks")
              for t in range(len(data2d))]
    p2d = [np.array([el[member] for el in arm_2d]).T for member in [SHOULDER, ELBOW, WRIST]]

    gt_p2d_full = np.concatenate(p2d).T  # T, 9
    gt_p2d_full[:, 2::3] = 1.  # Add homogeneous coordinate
    gt_p2d_full[:, 0::3] *= w  # Rescale x coordinates to image size in pixels
    gt_p2d_full[:, 1::3] *= h  # Rescale y coordinates to image size in pixels
    return (
        gt_p3d_full,  # (T, 9) 3D coordinates XYZ in world
        gt_p2d_full,  # (T, 9) homogeneous 2D xy1 coordinates
        q_states,  # (T, 5) joint angles
    )


def visualize_2d_trajectories(p2d_dict: dict, h_w_size: Tuple[int, int], title="2D image trajectories"):
    h, w = h_w_size
    plt.figure(figsize=(7, 7*h//w))
    for name, p2d in p2d_dict.items():
        plt.plot(p2d[:, 0], p2d[:, 1], "r-", label=f"{name} {SHOULDER}")
        plt.plot(p2d[:, 3], p2d[:, 4], "g-", label=f"{name} {ELBOW}")
        plt.plot(p2d[:, 6], p2d[:, 7], "b-", label=f"{name} {WRIST}")
    plt.grid()
    plt.legend()
    plt.xlim(0, w)
    plt.ylim(h, 0)
    plt.title(title)
    plt.show()


def get_2d_projection_from_arm_config_estimations(
    q_states: np.ndarray,
    arm_robot: ArmRobot,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix_list: List[np.ndarray],
    joints_list: List[str] = [SHOULDER, ELBOW, WRIST]
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward kinematics to get 3D points and project them to 2D

    From a sequence of arm configurations,
    get the 3D joints positions and the 2D projections of the arm joints
    get the 4D homogeneous coordinates of the 3D joints positions for optimization

    Args:
        q_states (np.ndarray): config states of the arm (T, 5)
        arm_robot (ArmRobot): arm robotic model
        intrinsic_matrix (np.ndarray): intrinsic camera matrix (3, 3)
        extrinsic_matrix_list (List[np.ndarray]): sequence of extrinsic camera matrices (T, 3, 4)
        joints_list (List[str], optional): joint names. Defaults to [SHOULDER, ELBOW, WRIST].

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
        - p2d_estim (T,9)
        - p3d_estim (T,9)
        - p4d_estim (T, 12)
    """
    # [ELBOW, SHOULDER, WRIST]
    p2d_estim = np.ones((len(q_states), 3*len(joints_list)))
    p3d_estim = np.empty((len(q_states), 3*len(joints_list)))
    p4d_estim = np.empty((len(q_states), 4*len(joints_list)))
    for time_idx in range(len(q_states)):
        extrinsic_matrix = extrinsic_matrix_list[time_idx]
        for member_idx, member in enumerate(joints_list):
            current_q = q_states[time_idx]
            point, jac = forward_kinematics(arm_robot, current_q, frame=member)
            p3d = point.translation
            p2d = project_3D_point(p3d, intrinsic_matrix, extrinsic_matrix)
            p4d = get_4D_homogeneous_vector(p3d)[:, 0]
            p2d_estim[time_idx, member_idx*3:member_idx*3+2] = p2d
            p3d_estim[time_idx, member_idx*3:member_idx*3+3] = p3d
            p4d_estim[time_idx, member_idx*4:member_idx*4+4] = p4d
    return (
        p2d_estim,  # (T, 9) homogeneous 2D xy1 coordinates
        p3d_estim,  # (T, 9) 3D coordinates XYZ in world
        p4d_estim,  # (T, 12) 3D coordinates XYZ1 in world
    )


def get_extrinsics_default(
    cam_position_3d_world: np.ndarray = [0., -3., 1.],
    cam_rotation: np.ndarray = np.eye(3)
) -> np.ndarray:
    """Get an extrinsic matrix from a camera position and rotation


    Args:
        cam_position_3d_world (np.ndarray, optional): Position of the camera focal point. 
        Defaults to [0., -3., 1.].
        cam_rotation (np.ndarray, optional): Identity - we assume camera cannot rotate. Defaults to np.eye(3).

    Returns:
        np.ndarray: camera extrinsic matrix (3, 4)
    """
    extrinsic_matrix_coarse_init = np.zeros((3, 4))
    extrinsic_matrix_coarse_init[:3, :3] = cam_rotation
    cam_pos = get_4D_homogeneous_vector(cam_position_3d_world)
    extrinsic_matrix_coarse_init[:3, -1] = -cam_pos[:3, 0]
    return extrinsic_matrix_coarse_init
