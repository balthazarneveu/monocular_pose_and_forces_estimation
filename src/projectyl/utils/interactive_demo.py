from projectyl.utils.properties import SIZE, RIGHT
from projectyl.utils.interactive import frame_selector, crop
from interactive_pipe import interactive_pipeline
from typing import Union, List
import numpy as np
from pathlib import Path
from projectyl.dynamics.inverse_kinematics import build_arm_model
import projectyl.utils.fit_camera_pose as fit_cam
from projectyl.utils.pose_overlay import get_camera_config_filter, overlay_pose, forward_camera_projection


def update_arm_robot_live(q_states, global_params={}):
    build_arm_model(global_params=global_params)
    # arm = global_params.get("arm", None)
    viz = global_params.get("viz", None)
    frame_idx = global_params["frame_idx"]
    current_state = q_states[frame_idx]
    viz.display(current_state)
    global_params["q"] = current_state


def demo_pipeline(sequence, pose_annotations, camera_config, q_states):
    frame = frame_selector(sequence)
    get_camera_config_filter(frame, camera_config)
    frame_overlay = overlay_pose(frame, pose_annotations)
    update_arm_robot_live(q_states)
    reproj = forward_camera_projection(frame_overlay)
    cropped = crop(frame_overlay)
    return cropped, reproj


def interactive_demo(sequence: Union[Path, List[np.ndarray]], pose_annotations, states_sequences=None, camera_config={}):
    (h, w) = camera_config[SIZE]
    arm_side = RIGHT
    p3d, p2d, q_states = fit_cam.build_3d_2d_data_arrays(states_sequences, pose_annotations, (h, w), arm_side=arm_side)
    int_demo = interactive_pipeline(gui="auto", cache=True)(demo_pipeline)

    int_demo(sequence, pose_annotations, camera_config, q_states)
