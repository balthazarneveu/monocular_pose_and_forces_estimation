from projectyl.utils.properties import LEFT, ELBOW, SHOULDER, WRIST
import numpy as np
import logging
from time import sleep
from interactive_pipe import interactive, interactive_pipeline


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
