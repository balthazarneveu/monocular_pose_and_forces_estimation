from projectyl.utils.properties import LEFT, ELBOW, SHOULDER, WRIST
import numpy as np
import logging
from time import sleep
from interactive_pipe import interactive, interactive_pipeline
import matplotlib.pyplot as plt


def retrieve_arm_estimation(body_pose_full: list, frame_idx: int, arm_side: str = LEFT) -> dict:
    arm_estim = {}
    body_pose = body_pose_full[frame_idx]["pose_world_landmarks"][0]
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
    frame_idx=(0., [0., 1.], "frame_idx"),
)
def interactive_replay(q_list, viz, frame_idx=0., global_params={}):
    q = q_list[int(frame_idx*(len(q_list)-1))]
    viz.display(q)
    sleep(1/30.)


def replay_sequence(q_list, viz):
    interactive_replay(q_list, viz)


def interactive_replay_sequence(q_list, viz):
    interactive_pipeline(safe_input_buffer_deepcopy=False)(replay_sequence)(q_list, viz)


def plot_optimization_curves(states_to_plot: list, mode: str = "qvt", title: str = "Optimization problem"):
    n_fig = len(mode)
    fig, axs = plt.subplots(1, n_fig, figsize=(n_fig*5, 5))
    if n_fig == 1:
        axs = [axs]
    frozen_color_code = ["r", "g", "b", "y", "m", "c", "k"]

    for state, label, style in states_to_plot:
        graph_id = 0
        if "q" in mode:
            ax = axs[graph_id]
            for dim in range(3):
                ax.plot(state[:, 0+dim], style+frozen_color_code[dim],
                        label=f"q shoulder {label} {dim}")  # skip index 3 (quaternion normalization)
            ax.plot(state[:, 4], style+frozen_color_code[3], label=f"q elbow {label}")
            graph_id += 1
        if "v" in mode:
            ax = axs[graph_id]
            for dim in range(3):
                ax.plot(state[:, 4+1+dim], style+frozen_color_code[dim], label=f"q' shoulder {label} {dim}")
            ax.plot(state[:, 4+3+1], style+frozen_color_code[3], label=f"q' elbow {label}")
            ax.set_title("Velocity [rad/s] (?)")
            graph_id += 1
        if "t" in mode:
            ax = axs[graph_id]
            for dim in range(3):
                ax.plot(state[:, 4+3+1+dim], style+frozen_color_code[dim],
                        label=f"Torque shoulder {label} {dim}")
            ax.set_title("Torque")
            ax.plot(state[:, 4+3+1], style+frozen_color_code[3], label=f"Torque elbow {label}")
            graph_id += 1
    for i in range(len(axs)):
        axs[i].legend()
        axs[i].grid()
    plt.suptitle(title)
    plt.show()


# plot_optimization_curves([(gt_sol, "[gt]", "--"),])


def plot_ik_states(conf_list):
    plot_optimization_curves(
        [(np.array(conf_list["q"]), "estimation", "--")],
        mode="q",
        title="State estimation by Inverse kinematics optimization"
    )
