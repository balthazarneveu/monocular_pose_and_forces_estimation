from projectyl.utils.interactive import frame_selector, crop
from interactive_pipe import interactive, interactive_pipeline
from projectyl.algo.pose_estimation import draw_landmarks_on_image
from typing import Union, List
import numpy as np
from pathlib import Path


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


def visualize_pose(sequence, pose_annotations):
    frame = frame_selector(sequence)
    frame_overlay = overlay_pose(frame, pose_annotations)
    cropped = crop(frame_overlay)
    return cropped


def interactive_visualize_pose(sequence: Union[Path, List[np.ndarray]], pose_annotations):
    int_viz = interactive_pipeline(gui="auto", cache=True)(visualize_pose)
    int_viz(sequence, pose_annotations)
