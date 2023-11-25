import pinocchio as pin
from tqdm import tqdm
import numpy as np
from time import sleep
from projectyl.dynamics.armmodel import ArmRobot
from projectyl.dynamics.meshcat_viewer_wrapper import MeshcatVisualizer
from typing import Union, Optional, Tuple


def get_frame_id(arm: ArmRobot, frame: Union[str, int]) -> int:
    if isinstance(frame, str):
        frame_id = arm.model.getFrameId(frame)
    elif isinstance(frame, int):
        frame_id = frame
    return frame_id


def forward_kinematics(
    arm: ArmRobot, q: np.ndarray,
    frame: Optional[Union[str, int]] = "end_effector"
) -> Tuple[pin.SE3, np.ndarray]:
    frame_id = get_frame_id(arm, frame)
    pin.framesForwardKinematics(arm.model, arm.data, q)
    o_Mtool = arm.data.oMf[frame_id].copy()
    o_Jtool = pin.computeFrameJacobian(arm.model, arm.data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
    return o_Mtool, o_Jtool


def inverse_kinematics(
    arm: ArmRobot,
    viz: MeshcatVisualizer,
    target_position: pin.SE3,
    q_init: np.ndarray = None,
    joint_name: Optional[Union[str, int]] = "end_effector"
):
    if q_init is None:
        q_init = pin.randomConfiguration(arm.model)
    q = q_init.copy()
    DT = 1e-2
    for i in tqdm(range(500)):
        o_Mtool, o_Jtool = forward_kinematics(arm, q, joint_name)
        o_Jtool3 = o_Jtool[:3, :]
        o_TG = target_position.translation - o_Mtool.translation
        vq = np.linalg.pinv(o_Jtool3) @ o_TG[:3]
        q = pin.integrate(arm.model, q, vq*DT)
        if i % 50 == 0:
            viz.display(q)
            sleep(1e-3)
    return q


def build_arm_model(global_params: dict = {}):
    arm = global_params.get("arm", None)
    if arm is None:
        arm = ArmRobot(upper_arm_length=0.3, forearm_length=0.25)
        global_params["arm"] = arm
    viz = global_params.get("viz", None)
    if viz is None:
        viz = MeshcatVisualizer(arm)
        viz.display(arm.q0)
        global_params["viz"] = viz


def update_arm_model(body_pose_full, global_params={}, fit_elbow=False, scale_constant=1.):
    # shoulder, elbow, wrist
    SHOULDER, ELBOW, WRIST = "shoulder", "elbow", "wrist"
    COLORS = {
        SHOULDER: [.5, .1, .1, 1.],
        ELBOW: [.1, 1., .1, 1.],
        WRIST:  [.1, .1, 1., 1.],
    }
    arm_estim = {}
    frame_idx = global_params["frame_idx"]
    body_pose = body_pose_full[frame_idx][0]
    correspondance = [(SHOULDER, 11), (ELBOW, 13), (WRIST, 15)]
    for joint_name, current_joint_id in correspondance:
        arm_estim[joint_name] = [
            body_pose[current_joint_id].x,
            body_pose[current_joint_id].y,
            body_pose[current_joint_id].z
        ]
    arm = global_params.get("arm", None)
    viz = global_params.get("viz", None)
    shoulder_pos = np.array([0., arm_estim[SHOULDER][0], -arm_estim[SHOULDER][1]])

    def get_estimated_arm_se3(joint_pos):
        target_position = np.array([0., joint_pos[0], -joint_pos[1]])
        target_position = scale_constant*(target_position - shoulder_pos) + np.array([0., 0., 1.])
        new_pose = pin.SE3()
        new_pose.rotation = np.eye(3)
        new_pose.translation = target_position
        return new_pose
    joint_estimated_poses = {}
    for joint_name, joint_pos in arm_estim.items():
        joint_estimated_pose = get_estimated_arm_se3(joint_pos)
        viz.addBox(joint_name, [.05, .05, .05], COLORS[joint_name])
        viz.applyConfiguration(joint_name, joint_estimated_pose)
        joint_estimated_poses[joint_name] = joint_estimated_pose
    q = global_params.get("q", None)
    # CALL IK TWICE - once for wrist, once for elbow
    q = inverse_kinematics(arm, viz, joint_estimated_poses["wrist"], q_init=q, joint_name="end_effector")
    if fit_elbow:
        q = inverse_kinematics(arm, viz, joint_estimated_poses["elbow"], q_init=q, joint_name="elbow")
    global_params["q"] = q
