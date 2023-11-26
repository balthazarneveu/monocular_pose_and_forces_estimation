from projectyl.utils.properties import LEFT, ELBOW, SHOULDER, WRIST


def retrieve_arm_estimation(body_pose_full: list, frame_idx: int, arm_side: str= LEFT) -> dict:
    arm_estim = {}
    body_pose = body_pose_full[frame_idx][0]
    correspondance = [(SHOULDER, 11), (ELBOW, 13), (WRIST, 15)] if arm_side == LEFT else [
        (SHOULDER, 12), (ELBOW, 14), (WRIST, 16)]
    for joint_name, current_joint_id in correspondance:
        arm_estim[joint_name] = [
            body_pose[current_joint_id].x,
            body_pose[current_joint_id].y,
            body_pose[current_joint_id].z
        ]
    return arm_estim
