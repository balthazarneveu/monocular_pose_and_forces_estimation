import pinocchio as pin
from tqdm import tqdm
import numpy as np
from time import sleep
from projectyl.dynamics.armmodel import ArmRobot
from projectyl.dynamics.meshcat_viewer_wrapper import MeshcatVisualizer
from typing import Union, Optional, Tuple, List
import logging


def extract_dim(vec: np.ndarray, start: int, end: int) -> np.ndarray:
    return vec[start:end, ...]


def get_config_velocity_update_translation_with_proj(
    q: np.ndarray,
    arm: pin.RobotWrapper,
    index_object: int,
    o_M_target: pin.SE3,
    constraints: Tuple[int, int] = (0, 3),
    projector: np.ndarray = None,
    vq_prev: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Get a configuration update (velocity vq) to move the object to the target position.
    When projector and vq_prev are None, this function behaves like the first iteration
    Which means that Identity projects into the whole space without constraint.
    Warning: Calling this function assumes that Foward Kinematics data is up to date.

    Args:
        q (np.ndarray): current configuration state.
            - (joint angles, basis position, etc...)
        arm (pin.RobotWrapper): Robot instance.
        index_object (int): index of an object in the robot model (like effector or basis).
        o_M_target (pin.SE3): Target object position. SE(3) used here simply for its translation.
        constraints (Tuple[int, int], optional): Constrain only certain dimension of the target vector (from a to b).
            - Defaults to (0, 3) meaning no constraint.
            - (0,1) means constraining on the x axis.
            - (1,2) means constraining on the y axis.
            - (2,3) means constraining on the z axis.
            - (0,2) means constraining on the x & y axis.
        projector (np.ndarray, optional): Previous task projector matrix. Defaults to None.
            - Required not to deviate from the previous task direction - only evolve in the orthogonal space.
        vq_prev (np.ndarray, optional): Previous task velocity update. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: vq, projector
    """
    if projector is None:
        projector = np.eye(arm.nv)  # Identity matrix
    if vq_prev is None:
        vq_prev = np.zeros(arm.nv)  # Null vector

    # Current object location -> o_Mcurrent
    o_Mcurrent = arm.data.oMf[index_object]

    # Compute the error between the current object and the target object -> obj_2_goal
    obj_2_goal = (o_M_target.translation - o_Mcurrent.translation)
    obj_2_goalC = extract_dim(obj_2_goal, *constraints)  # constraint on some specific dimensions

    # Compute the jacobian of the object -> o_J_obj , constrained on specific dimensions.
    o_J_obj = pin.computeFrameJacobian(arm.model, arm.data, q, index_object, pin.LOCAL_WORLD_ALIGNED)
    o_J_objC = extract_dim(o_J_obj, *constraints)  # + constraint on some specific dimensions

    new_error = (obj_2_goalC - o_J_objC @ vq_prev)

    J = o_J_objC @ projector
    Jinv = np.linalg.pinv(J)  # pinv(J2@P1)

    vq = vq_prev + Jinv @ new_error
    # Compute updated projector.

    new_proj = projector - Jinv @ J
    # Note the special case when projector is the identity matrix,
    # we get the same result as the first iteration.

    return vq, new_proj


def solve_tasks(
    task_list: List[Tuple[int, pin.SE3, Tuple[int, int]]],
    # DT: float = 3e-2,
    DT: float = 3e-3,
    Niter: int = 500,
    viz=None,
    q_init=None,
    arm=None
) -> np.ndarray:
    if q_init is None:
        logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Random init")
        q_init = pin.randomConfiguration(arm.model)
    q = q_init.copy()
    viz.display(q_init)

    for it in tqdm(range(Niter), desc="IK"):  # Integrate over 2 second of rob life
        pin.framesForwardKinematics(arm.model, arm.data, q)  # update Forward kinematic
        vq, p = None, None

        for frame_id, o_M_target, constraints in task_list:
            # Iterate over the tasks
            vq, p = get_config_velocity_update_translation_with_proj(
                q, arm, frame_id, o_M_target, constraints=constraints,
                vq_prev=vq, projector=p
            )
        q = pin.integrate(arm.model, q, vq*DT)
        if it % 50 == 0:
            viz.display(q)
            sleep(1.E-3)
    viz.display(q)
    return q


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


def update_arm_model(body_pose_full, global_params={}, fit_wrist=True, fit_elbow=False, scale_constant=1.):
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
    task_list = []
    if fit_wrist:
        task_list.append(
            (arm.model.getFrameId("end_effector"), joint_estimated_poses[WRIST], (0, 3)),
        )
    if fit_elbow:
        task_list.append(
            (arm.model.getFrameId("elbow"), joint_estimated_poses[ELBOW], (0, 3)),
        )
    q = global_params.get("q", None)
    q = solve_tasks(
        task_list,
        viz=viz,
        q_init=q,
        arm=arm
    )
    global_params["q"] = q
