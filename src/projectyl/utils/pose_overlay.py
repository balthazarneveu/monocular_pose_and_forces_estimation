from projectyl.utils.properties import COLOR, POSITION, SIZE, ELBOW, SHOULDER, WRIST, CAMERA, LEFT, RIGHT
from projectyl.video.props import INTRINSIC_MATRIX, EXTRINSIC_MATRIX
from projectyl.utils.interactive import frame_selector, crop
from interactive_pipe import interactive, interactive_pipeline
from projectyl.algo.pose_estimation import draw_landmarks_on_image
from typing import Union, List
import numpy as np
from pathlib import Path
from projectyl.dynamics.inverse_kinematics import update_arm_model_inverse_kinematics, build_arm_model
from projectyl.utils.camera_projection import (
    project_3D_point, get_intrinic_matrix, get_4D_homogeneous_vector,
    get_focal_from_full_frame_equivalent, rescale_focal
)
from projectyl.dynamics.meshcat_viewer_wrapper import MeshcatVisualizer
from projectyl.dynamics.inverse_kinematics import forward_kinematics
from pinocchio import SE3
import cv2 as cv


@interactive(
    pose_overlay=(True, "pose_overlay"),
    arm_side=(RIGHT, [LEFT, RIGHT], "arm_side"),
    joint_id=(-1, [-2, 32, 1])
)
def overlay_pose(frame, pose_annotations, global_params={}, pose_overlay=True, joint_id=-1, arm_side=LEFT):
    if not pose_overlay:
        return frame
    frame_idx = global_params["frame_idx"]
    global_params["side"] = arm_side
    new_annot = draw_landmarks_on_image(
        frame, pose_annotations[frame_idx]["pose_landmarks"], joint_id=joint_id, left_arm=(arm_side == LEFT))
    return new_annot


@interactive(
    update_arm=(True, "update_arm"),
    fit_mode=(ELBOW+"+"+WRIST, [ELBOW, WRIST, ELBOW+"+"+WRIST], "fit_mode"),
    # scale_constant=(1., [0.1, 10.], "scale_constant"),
)
def update_arm_model_filter(
    body_pose_full,
    update_arm=True,
    fit_mode=WRIST,
    scale_constant=1.,
    global_params={}
):

    if update_arm:
        build_arm_model(global_params=global_params)
        update_arm_model_inverse_kinematics(body_pose_full, global_params=global_params,
                                            fit_elbow=ELBOW in fit_mode, fit_wrist=WRIST in fit_mode,
                                            scale_constant=scale_constant, arm_side=global_params.get("side", LEFT))


def make_a_scene_in_3D(object_list, viz: MeshcatVisualizer = None) -> MeshcatVisualizer:
    """Make the 3D scene with the given objects in Meshcat

    Args:
        object_list (List[dict]): List of dictionaries {name, size, color, position}
        viz (MeshcatVisualizer, optional): meshcat visualizer. Defaults to None.
    """
    if viz is None:
        viz = MeshcatVisualizer()
    for name, obj in object_list.items():
        viz.addBox(
            name,
            obj[SIZE],
            obj[COLOR]
        )
        viz.applyConfiguration(name, SE3(np.eye(3), np.array(obj[POSITION])))
    return viz


def get_camera_config(w, h, viz=None):
    object_list = {
        CAMERA: {
            COLOR: [1., 0.5, 0.5, 1.],
            POSITION: [0., -1.6, 1.],
            SIZE: [0.05, 0.2, 0.05]
        }
    }
    if viz is not None:
        make_a_scene_in_3D(object_list, viz)
    # w, h = 1920, 1080  # Full HD 1080p

    fpix = rescale_focal(
        fpix=get_focal_from_full_frame_equivalent(),
        w_resized=w
    )
    k = get_intrinic_matrix((h, w), fpix)
    extrinsic_matrix = np.zeros((3, 4))
    extrinsic_matrix[:3, :3] = np.eye(3)
    cam_pos = get_4D_homogeneous_vector(object_list[CAMERA][POSITION])
    extrinsic_matrix[:3, -1] = -cam_pos[:3, 0]
    return k, extrinsic_matrix


def forward_camera_projection(img_ref, global_params={}):
    build_arm_model(global_params=global_params)
    viz = global_params.get("viz", None)
    arm = global_params["arm"]
    q = global_params.get("q", arm.q0)
    h, w = img_ref.shape[:2]
    k, extrinsic_matrix = global_params.get(INTRINSIC_MATRIX, None), global_params.get(EXTRINSIC_MATRIX, None)
    if k is None or extrinsic_matrix is None:
        k, extrinsic_matrix = get_camera_config(w, h, viz=viz)
    p2d_list = {}
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for frame_idx, frame in enumerate([SHOULDER, ELBOW, WRIST]):
        point, jac = forward_kinematics(arm, q, frame=frame)
        p3d = point.translation
        p2d = project_3D_point(p3d, k, extrinsic_matrix)
        p2d_list[frame] = {"2d": p2d, COLOR: colors[frame_idx]}

    img = 10*np.ones((h, w, 3))  # 720p frame
    cv.line(img, (0, h//2), (w, h//2), (255, 255, 255), 2)
    cv.line(img, (w//2, 0), (w//2, h), (255, 255, 255), 2)
    cv.circle(img, (w//2, h//2), 5, (255, 0, 0), -1)
    for frame_idx, frame in enumerate(p2d_list):
        p2d = p2d_list[frame]["2d"]
        cv.circle(img, (int(p2d[0]), int(p2d[1])), 20, p2d_list[frame][COLOR], -1)
        cv.putText(
            img,
            f"{frame}",
            (int(p2d[0]+5), int(p2d[1])-20),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            p2d_list[frame][COLOR],
            2,
            cv.LINE_AA
        )
        if frame_idx > 0:
            p2d_prev = p2d_list[list(p2d_list.keys())[frame_idx-1]]["2d"]
            cv.line(
                img,
                (int(p2d[0]), int(p2d[1])),
                (int(p2d_prev[0]), int(p2d_prev[1])),
                p2d_list[frame][COLOR], 2
            )
    img = img.clip(0, 255).astype(np.uint8)
    return cv.resize(img, None, fx=0.3, fy=0.3) / 255.


@interactive()
def get_camera_config_filter(img_ref, camera_config, global_params={}):
    h, w = img_ref.shape[:2]
    k_default, extrinsic_matrix = get_camera_config(w, h)
    if EXTRINSIC_MATRIX in camera_config.keys():
        extrinsic_params = camera_config[EXTRINSIC_MATRIX][global_params["frame_idx"]]
        extrinsic_matrix = np.zeros((3, 4))
        extrinsic_matrix[:3, :3] = np.eye(3)
        cam_pos = get_4D_homogeneous_vector(extrinsic_params)
        extrinsic_matrix[:3, -1] = -cam_pos[:3, 0]
        object_list = {
            CAMERA: {
                COLOR: [1., 0.5, 0.5, 1.],
                POSITION: extrinsic_params,
                SIZE: [0.05, 0.2, 0.05]
            }
        }
        viz = global_params.get("viz", None)
        if viz is not None:
            make_a_scene_in_3D(object_list, viz)
    k = camera_config.get(INTRINSIC_MATRIX, k_default) if camera_config is not None else k_default
    global_params[INTRINSIC_MATRIX] = k
    global_params[EXTRINSIC_MATRIX] = extrinsic_matrix
    pass


def visualize_pose(sequence, pose_annotations, camera_config):
    frame = frame_selector(sequence)
    get_camera_config_filter(frame, camera_config)
    frame_overlay = overlay_pose(frame, pose_annotations)
    update_arm_model_filter(pose_annotations)
    reproj = forward_camera_projection(frame_overlay)
    cropped = crop(frame_overlay)
    return cropped, reproj


def interactive_visualize_pose(sequence: Union[Path, List[np.ndarray]], pose_annotations, camera_config={}):
    int_viz = interactive_pipeline(gui="auto", cache=True)(visualize_pose)
    int_viz(sequence, pose_annotations, camera_config)
