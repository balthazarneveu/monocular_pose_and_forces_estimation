from time import sleep
from projectyl.utils.interactive import frame_selector, crop
from interactive_pipe import interactive, interactive_pipeline
from projectyl.algo.pose_estimation import draw_landmarks_on_image
from typing import Union, List
import numpy as np
from pathlib import Path
from projectyl.dynamics.inverse_kinematics import update_arm_model, build_arm_model


@interactive(
    pose_overlay=(True, "pose_overlay"),
    joint_id=(-1, [-2, 32, 1])
)
def overlay_pose(frame, pose_annotations, global_params={}, pose_overlay=True, joint_id=-1):
    if not pose_overlay:
        return frame
    frame_idx = global_params["frame_idx"]
    new_annot = draw_landmarks_on_image(frame, pose_annotations[frame_idx], joint_id=joint_id)
    return new_annot


@interactive(
    update_arm=(True, "update_arm"),
    fit_elbow=(False, "fit_elbow"),
    scale_constant=(5., [0.1, 10.], "scale_constant"),
)
def update_arm_model_filter(body_pose_full, update_arm=True, fit_elbow=False, scale_constant=5., global_params={}):
    if update_arm:
        build_arm_model(global_params=global_params)
        update_arm_model(body_pose_full, global_params=global_params,
                         fit_elbow=fit_elbow, scale_constant=scale_constant)




def visualize_pose(sequence, pose_annotations):
    frame = frame_selector(sequence)
    frame_overlay = overlay_pose(frame, pose_annotations)
    cropped = crop(frame_overlay)
    update_arm_model_filter(pose_annotations)
    return cropped


def interactive_visualize_pose(sequence: Union[Path, List[np.ndarray]], pose_annotations):
    int_viz = interactive_pipeline(gui="auto", cache=True)(visualize_pose)
    int_viz(sequence, pose_annotations)
