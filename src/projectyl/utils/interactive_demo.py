from projectyl.utils.properties import SIZE, RIGHT
from projectyl.utils.interactive import frame_selector, crop
from interactive_pipe import interactive_pipeline
from typing import Union, List
import numpy as np
from pathlib import Path
from projectyl.dynamics.inverse_kinematics import build_arm_model
import projectyl.utils.fit_camera_pose as fit_cam
from projectyl.utils.pose_overlay import get_camera_config_filter, overlay_pose, forward_camera_projection
from interactive_pipe.data_objects.curves import Curve, SingleCurve
from interactive_pipe import interactive


def update_arm_robot_live(q_states, global_params={}):
    build_arm_model(global_params=global_params)
    viz = global_params.get("viz", None)
    frame_idx = global_params["frame_idx"]
    current_state = q_states[frame_idx]
    viz.display(current_state)
    global_params["q"] = current_state

# @interactive(
#     show_legend=(False, "show_legend", ["z", "z"])
# )


def trajectories(p2d_dict, camera_config, global_params={}):
    show_legend = False
    (h, w) = camera_config[SIZE]
    all_curves = []
    for name, v in p2d_dict.items():
        if "fit" not in name:
            alpha = 0.1
            style = "--"
        else:
            alpha = 0.3
            style = "-"
        c1 = SingleCurve(v[:, 0], v[:, 1], style=f"r{style}", alpha=alpha,
                         label=f"{name} shoulder" if show_legend else None)
        c2 = SingleCurve(v[:, 3], v[:, 4], style=f"g{style}", alpha=alpha,
                         label=f"{name} elbow" if show_legend else None)
        c3 = SingleCurve(v[:, 6], v[:, 7], style=f"b{style}", alpha=alpha,
                         label=f"{name} wrist" if show_legend else None)
        all_curves.extend([c1, c2, c3])
    min_x = min([c.x.min() for c in all_curves])
    max_x = max([c.x.max() for c in all_curves])
    min_y = min([c.y.min() for c in all_curves])
    max_y = max([c.y.max() for c in all_curves])
    curves = Curve(
        all_curves,
        xlim=(min_x, max_x), ylim=(max_y, min_y),
        # xlim=(0, w), ylim=(h, 0),
        grid=True,
        title="2D Trajectories"
    )
    return curves


def update_traj(current_curve_in: Curve, p2d_dict, global_params={}):
    frame_idx = global_params["frame_idx"]
    from copy import deepcopy
    current_curve = deepcopy(current_curve_in)
    for name, v in p2d_dict.items():
        if "fit" not in name:
            alpha = 0.5
            style = "+"
            line_style = "--"
        else:
            alpha = 1.
            style = "o"
            line_style = "-"
        c1 = SingleCurve(v[frame_idx:frame_idx+1, 0], v[frame_idx:frame_idx+1, 1], style=f"r{style}", alpha=alpha)
        c2 = SingleCurve(v[frame_idx:frame_idx+1, 3], v[frame_idx:frame_idx+1, 4], style=f"g{style}", alpha=alpha)
        c3 = SingleCurve(v[frame_idx:frame_idx+1, 6], v[frame_idx:frame_idx+1, 7], style=f"b{style}", alpha=alpha)

        upper_arm = SingleCurve(
            [v[frame_idx, 0], v[frame_idx, 3]],
            [v[frame_idx, 1], v[frame_idx, 4]],
            style="g"+line_style,
            alpha=alpha,
            linewidth=4
        )
        fore_arm = SingleCurve(
            [v[frame_idx, 3], v[frame_idx, 6]],
            [v[frame_idx, 4], v[frame_idx, 7]],
            style="b"+line_style,
            alpha=alpha,
            linewidth=4
        )

        current_curve.append(c1)
        current_curve.append(c2)
        current_curve.append(c3)
        current_curve.append(upper_arm)
        current_curve.append(fore_arm)
    return current_curve


def demo_pipeline(sequence, pose_annotations, camera_config, q_states, p2d_dict):
    traj = trajectories(p2d_dict, camera_config)
    frame = frame_selector(sequence)
    trajectory = update_traj(traj, p2d_dict)
    get_camera_config_filter(frame, camera_config)
    frame_overlay = overlay_pose(frame, pose_annotations)
    update_arm_robot_live(q_states)
    reprojection = forward_camera_projection(frame_overlay)
    pose_estimation = crop(frame_overlay)
    return pose_estimation, trajectory #, trajectory
    # return pose_estimation, reprojection, trajectory
    # return [[pose_estimation, reprojection], [trajectory, None]]


def interactive_demo(sequence: Union[Path, List[np.ndarray]], pose_annotations, p2d_dict=None, states_sequences=None, camera_config={}):
    (h, w) = camera_config[SIZE]
    arm_side = RIGHT
    p3d, p2d, q_states = fit_cam.build_3d_2d_data_arrays(states_sequences, pose_annotations, (h, w), arm_side=arm_side)
    int_demo = interactive_pipeline(gui="auto", cache=True)(demo_pipeline)

    int_demo(sequence, pose_annotations, camera_config, q_states, p2d_dict)
