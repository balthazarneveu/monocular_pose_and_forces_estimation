from projectyl.utils.properties import LEFT, ELBOW, SHOULDER, WRIST
import numpy as np
import logging
from time import sleep
from interactive_pipe import interactive, interactive_pipeline
from interactive_pipe.data_objects.curves import Curve, SingleCurve
import matplotlib.pyplot as plt
FROZEN_COLOR_CODE = ["r", "g", "b", "y", "m", "c", "k"]


def retrieve_arm_estimation(body_pose_full: list, frame_idx: int, arm_side: str = LEFT, key="pose_world_landmarks") -> dict:
    arm_estim = {}
    if len(body_pose_full[frame_idx][key]) == 0:
        logging.warning(f"Empty pose at frame {frame_idx}")
        return None
    body_pose = body_pose_full[frame_idx][key][0]
    correspondance = [(SHOULDER, 11), (ELBOW, 13), (WRIST, 15)] if arm_side == LEFT else [
        (SHOULDER, 12), (ELBOW, 14), (WRIST, 16)]
    for joint_name, current_joint_id in correspondance:
        arm_estim[joint_name] = [
            body_pose[current_joint_id].x,
            body_pose[current_joint_id].y,
            body_pose[current_joint_id].z
        ]

    upper_arm = np.array(arm_estim[SHOULDER]) - np.array(arm_estim[ELBOW])
    upper_arm = np.sqrt((upper_arm**2).sum())
    fore_arm = np.array(arm_estim[ELBOW]) - np.array(arm_estim[WRIST])
    fore_arm = np.sqrt((fore_arm**2).sum())
    logging.debug(f"UPPER ARM = {100.*upper_arm:.1f}cm, FORE ARM {100.*fore_arm:.1f}cm")
    return arm_estim


def backward_project(arm_estim, intrisic_matrix):
    pass


def replay_whole_sequence(q_list, viz):
    for q in q_list:
        viz.display(q)
        sleep(1/30.)


@interactive(
    # frame_idx=(0., [0., 1.], "frame_idx",  0.000001),
    frame_idx=[0, 500, 1, 0],
    mode=(0, [0, 10], "index [0 for estimation -1 for GT]"),
)
def seclect_replay(q_dict, frame_idx=0., mode=0, global_params={}):
    q_keys = list(q_dict.keys())
    q_key = q_keys[min(mode, len(q_keys)-1)]
    q_list = q_dict[q_key]
    logging.info(f"Replay mode {q_key} at index {frame_idx}")
    global_params["frame_idx"] = int(min(frame_idx, len(q_list)-1))  # int(frame_idx*(len(q_list)-1))
    global_params["q_key"] = q_key
    return q_list,


@interactive()
def meshcat_replay(q_list, viz,  global_params={}):
    frame_idx = global_params["frame_idx"]
    q = q_list[frame_idx]
    viz.display(q)


@interactive()
def display_curve(q_dict, global_params={}):
    ts = global_params["frame_idx"]
    q_key = global_params["q_key"]
    q_array = np.array(q_dict[q_key])
    curves = [
        SingleCurve(
            y=q_array[:, dim],
            style="-"+FROZEN_COLOR_CODE[dim],
            label=f"{q_key} shoulder {dim}")
        for dim in range(3)
    ]
    curves.append(SingleCurve(
        y=q_array[:, 4],
        style="-"+FROZEN_COLOR_CODE[3],
        label=f"{q_key} elbow"))
    curves.append(SingleCurve(x=[ts, ts], y=[-1, 1], style="--k", label=None))
    return Curve(
        curves,
        title="Q state [rad]", xlabel=f"Time index - {ts:d}", ylabel="Q [rad]",
        grid=True
    )


def replay_sequence(q_dict, viz):
    q_list = seclect_replay(q_dict)
    meshcat_replay(q_list, viz)
    curve = display_curve(q_dict)

    return curve


def interactive_replay_sequence(q_dict, viz):
    interactive_pipeline(safe_input_buffer_deepcopy=False)(replay_sequence)(q_dict, viz)


def plot_optimization_curves(states_to_plot: list, mode: str = "qvt", title: str = "Optimization problem"):
    n_fig = len(mode)
    fig, axs = plt.subplots(1, n_fig, figsize=(n_fig*5, 5))
    if n_fig == 1:
        axs = [axs]

    for state, label, style in states_to_plot:
        graph_id = 0
        if "q" in mode:
            ax = axs[graph_id]
            for dim in range(3):
                ax.plot(state[:, 0+dim], style+FROZEN_COLOR_CODE[dim],
                        label=f"q shoulder {label} {dim}")  # skip index 3 (quaternion normalization)
            ax.plot(state[:, 4], style+FROZEN_COLOR_CODE[3], label=f"q elbow {label}")
            ax.set_title("Q state [rad] (?)")
            graph_id += 1
        if "v" in mode:
            ax = axs[graph_id]
            for dim in range(3):
                ax.plot(state[:, 4+1+dim], style+FROZEN_COLOR_CODE[dim], label=f"q' shoulder {label} {dim}")
            ax.plot(state[:, 4+3+1], style+FROZEN_COLOR_CODE[3], label=f"q' elbow {label}")
            ax.set_title("Velocity [rad/s] (?)")
            graph_id += 1
        if "t" in mode:
            ax = axs[graph_id]
            for dim in range(3):
                ax.plot(state[:, 4+3+2+dim], style+FROZEN_COLOR_CODE[dim],
                        label=f"Torque shoulder {label} {dim}")
            ax.set_title("Torque")
            ax.plot(state[:, -1], style+FROZEN_COLOR_CODE[3], label=f"Torque elbow {label}")
            graph_id += 1
    for i in range(len(axs)):
        axs[i].legend()
        axs[i].grid()
        axs[i].set_xlabel("Time index")
    plt.suptitle(title)
    plt.show()


def plot_ik_states(conf_list):
    plot_optimization_curves(
        [(np.array(conf_list["q"]), "estimation", "--")],
        mode="q",
        title="State estimation by Inverse kinematics optimization"
    )
