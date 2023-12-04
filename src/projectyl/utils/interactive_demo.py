import math
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
from interactive_pipe import KeyboardControl
from copy import deepcopy


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


CANVAS_DICT = {
    "vision": [["pose_estimation", "arm_states"]],
    "trajectories": [["trajectory",]],
    "full": [["pose_estimation", "reprojection", "trajectory"]],
}
CANVAS = list(CANVAS_DICT.keys())


@interactive(
    # canvas=KeyboardControl(0, [0, len(CANVAS)], name="canvas", keyup="p", modulo=True)
    canvas=KeyboardControl(CANVAS[0], CANVAS, name="canvas", keyup="p", modulo=True)
)
def morph_canvas(canvas=CANVAS[0], global_params={}):
    global_params["__pipeline"].outputs = CANVAS_DICT[canvas]
    return None


def euler_from_quaternion(x, y, z, w):
    """
    https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 * (t2 > +1.0) + t2 * (t2 <= +1.0)
    t2 = -1.0 * (t2 < -1.0) + t2 * (t2 >= -1.0)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


@interactive(
    euler=KeyboardControl(False, name="euler", keyup="e")
)
def plot_states_sequence(q_states, q_angles, euler=False, global_params={}):
    global_params["euler"] = euler
    if euler:
        csx = SingleCurve(y=q_angles[:, 0], style="r-", alpha=0.5)  # , label="shoulder X")
        csy = SingleCurve(y=q_angles[:, 1], style="g-", alpha=0.5)  # , label="shoulder Y")
        csz = SingleCurve(y=q_angles[:, 2], style="b-", alpha=0.5)  # , label="shoulder Z")
        # csz = SingleCurve(y=90*q_states[:, 3], style="k--", alpha=0.1)  # , label="shoulder Z")
        ce = SingleCurve(y=np.rad2deg(q_states[:, 4]), style="y-", alpha=0.5)  # , label="elbow")
    else:
        csx = SingleCurve(y=q_states[:, 0], style="r-", alpha=0.5)
        csy = SingleCurve(y=q_states[:, 1], style="g-", alpha=0.5)
        csz = SingleCurve(y=q_states[:, 2], style="b-", alpha=0.5)
        ce = SingleCurve(y=q_states[:, 4], style="y-", alpha=0.5)  # , label="elbow")

    return Curve(
        [csx, csy, csz, ce],
        title="Joint euler angles (°)" if euler else "Joints orientations",
        grid=True,
        ylim=(-180, 180) if euler else (-np.pi, np.pi),
        xlabel="Time (Frame)",
        ylabel="Angle (°)" if euler else "Quaternions"
    )


def update_plot_states_sequence(q_states, q_angles, current_curve_in: Curve, global_params={}):
    frame_idx = global_params["frame_idx"]
    euler = global_params["euler"]
    q_plot = q_angles if euler else q_states
    current_curve = deepcopy(current_curve_in)
    elbow_angle = np.rad2deg(q_states[frame_idx, 4]) if euler else q_states[frame_idx, 4]
    csx = SingleCurve(x=[frame_idx], y=[q_plot[frame_idx, 0]], style="ro", label="shoulder X")
    csy = SingleCurve(x=[frame_idx], y=[q_plot[frame_idx, 1]], style="go", label="shoulder Y")
    csz = SingleCurve(x=[frame_idx], y=[q_plot[frame_idx, 2]], style="bo", label="shoulder Z")
    ce = SingleCurve(x=[frame_idx], y=[elbow_angle], style="yo", label="elbow")
    current_curve.append(csx)
    current_curve.append(csy)
    current_curve.append(csz)
    current_curve.append(ce)
    return current_curve


def demo_pipeline(sequence, pose_annotations, camera_config, q_states, q_angles, p2d_dict):
    traj = trajectories(p2d_dict, camera_config)
    arm_states = plot_states_sequence(q_states, q_angles)
    frame = frame_selector(sequence)
    trajectory = update_traj(traj, p2d_dict)
    arm_states = update_plot_states_sequence(q_states, q_angles, arm_states)
    get_camera_config_filter(frame, camera_config)
    frame_overlay = overlay_pose(frame, pose_annotations)
    update_arm_robot_live(q_states)
    reprojection = forward_camera_projection(frame_overlay)
    pose_estimation = crop(frame_overlay)

    morph_canvas()
    return pose_estimation, trajectory


def interactive_demo(sequence: Union[Path, List[np.ndarray]], pose_annotations, p2d_dict=None, states_sequences=None, camera_config={}):
    (h, w) = camera_config[SIZE]
    arm_side = RIGHT
    p3d, p2d, q_states = fit_cam.build_3d_2d_data_arrays(states_sequences, pose_annotations, (h, w), arm_side=arm_side)
    if states_sequences is not None:
        # states_sequences = np.array(states_sequences)

        q_states_quat = q_states.copy()
        q_states_quat[:, :4] /= q_states_quat[:, 3:4]
        q_angles = euler_from_quaternion(
            q_states_quat[:, 0], q_states_quat[:, 1], q_states_quat[:, 2], q_states_quat[:, 3])
        q_angles = np.stack(q_angles, axis=1)
        q_angles = np.rad2deg(q_angles)
        # print(q_angles.shape)
    else:
        q_angles = None
    int_demo = interactive_pipeline(gui="auto", cache=True)(demo_pipeline)

    int_demo(sequence, pose_annotations, camera_config, q_states, q_angles, p2d_dict)
