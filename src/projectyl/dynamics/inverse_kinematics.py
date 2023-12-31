import pinocchio as pin
from tqdm import tqdm
import numpy as np
from time import sleep
from projectyl.dynamics.armmodel import ArmRobot
from projectyl.dynamics.meshcat_viewer_wrapper import MeshcatVisualizer
from projectyl.utils.arm import retrieve_arm_estimation, interactive_replay_sequence
from typing import Union, Optional, Tuple, List
import logging
from projectyl.utils.properties import SHOULDER, ELBOW, WRIST, LEFT, RIGHT


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
    arm=None,
    progress_bar=True
) -> np.ndarray:
    if q_init is None:
        logging.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Random init")
        q_init = pin.randomConfiguration(arm.model)
    q = q_init.copy()
    if viz is not None:
        viz.display(q_init)

    for it in (range(Niter) if not progress_bar else tqdm(range(Niter), desc="IK")):  # Integrate over 2 second of rob life
        pin.framesForwardKinematics(arm.model, arm.data, q)  # update Forward kinematic
        vq, p = None, None

        for frame_id, o_M_target, constraints in task_list:
            # Iterate over the tasks
            vq, p = get_config_velocity_update_translation_with_proj(
                q, arm, frame_id, o_M_target, constraints=constraints,
                vq_prev=vq, projector=p
            )
        q = pin.integrate(arm.model, q, vq*DT)
        if viz is not None:
            if it % 50 == 0:
                viz.display(q)
                sleep(1.E-3)
    if viz is not None:
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
    frame: Optional[Union[str, int]] = WRIST,
    forward_update: bool = True
) -> Tuple[pin.SE3, np.ndarray]:
    frame_id = get_frame_id(arm, frame)
    if forward_update:
        pin.framesForwardKinematics(arm.model, arm.data, q)
    o_Mtool = arm.data.oMf[frame_id].copy()
    o_Jtool = pin.computeFrameJacobian(arm.model, arm.data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
    return o_Mtool, o_Jtool


def build_arm_model(global_params: dict = {}, headless=False, free_elbow=False):
    arm = global_params.get("arm", None)
    if arm is None:
        arm = ArmRobot(upper_arm_length=0.23, forearm_length=0.27, headless=headless, free_elbow=free_elbow)
        global_params["arm"] = arm
    viz = global_params.get("viz", None)
    if viz is None and not headless:
        viz = MeshcatVisualizer(arm)
        viz.display(arm.q0)
        global_params["viz"] = viz


def permute_estimated_poses(joint_pos):
    target_position = np.array([joint_pos[0], joint_pos[2], -joint_pos[1]])
    return target_position


def update_arm_model_inverse_kinematics(
    body_pose_full, global_params={}, fit_wrist=True,
    fit_elbow=False, scale_constant=1., arm_side=LEFT,
    progress_bar=False
):
    # shoulder, elbow, wrist
    COLORS_LIST = {
        SHOULDER: [.5, .1, .1, 1.],
        ELBOW: [.1, 1., .1, 1.],
        WRIST:  [.1, .1, 1., 1.],
    }
    frame_idx = global_params["frame_idx"]
    arm_estim = retrieve_arm_estimation(body_pose_full, frame_idx, arm_side)
    if arm_estim is None:
        return global_params.get("q", None), None, None
    # backward_project(arm_estim, global_params["intrisic_matrix"])
    arm = global_params.get("arm", None)
    viz = global_params.get("viz", None)
    shoulder_pos = permute_estimated_poses(arm_estim[SHOULDER])

    def get_estimated_arm_se3(joint_pos):
        target_position = permute_estimated_poses(joint_pos)
        target_position = scale_constant*(target_position - shoulder_pos) + np.array([0., 0., 1.])
        new_pose = pin.SE3()
        new_pose.rotation = np.eye(3)
        new_pose.translation = target_position
        return new_pose
    joint_estimated_poses = {}
    for joint_name, joint_pos in arm_estim.items():
        joint_estimated_pose = get_estimated_arm_se3(joint_pos)
        joint_estimated_poses[joint_name] = joint_estimated_pose

    def standardize_length(
        joint_estimated_poses: dict, forced_length: float,
        start_name: str = SHOULDER, end_name: str = ELBOW,
        sanity_check=False
    ) -> None:
        """Force length between 3D points
        - move end
        - keep start
        - distance = ||start-end||

        This trick allows respecting the arm possible workspace.
        And therefor grants correct IK solution.
        """
        start_pos = joint_estimated_poses[start_name].translation
        end_pos = joint_estimated_poses[end_name].translation
        length = np.sqrt(((end_pos - start_pos)**2).sum())
        new_end_pos = start_pos + (end_pos - start_pos)/length*forced_length
        joint_estimated_poses[end_name].translation = new_end_pos
        if sanity_check:
            start_pos = joint_estimated_poses[start_name].translation
            end_pos = joint_estimated_poses[end_name].translation
            length = np.sqrt(((end_pos - start_pos)**2).sum())
            assert np.isclose(length, forced_length), f"Length is {length} instead of {forced_length}"
    original_joint_3d_positions = joint_estimated_poses.copy()
    standardize_length(joint_estimated_poses, start_name=SHOULDER, end_name=ELBOW, forced_length=arm.upper_arm_length)
    standardize_length(joint_estimated_poses, start_name=ELBOW, end_name=WRIST, forced_length=arm.forearm_length)
    joint_3d_positions = joint_estimated_poses.copy()
    if viz is not None:
        for joint_name, joint_pos in arm_estim.items():
            viz.addBox(joint_name, [.05, .05, .05], COLORS_LIST[joint_name])
            viz.applyConfiguration(joint_name, joint_estimated_poses[joint_name])

    task_list = []
    if fit_elbow:
        task_list.append(
            # (arm.model.getFrameId(ELBOW), joint_estimated_poses[ELBOW], (0, 2) if fit_wrist else (0, 3)),
            (arm.model.getFrameId(ELBOW), joint_estimated_poses[ELBOW], (0, 3) if fit_wrist else (0, 3)),
        )
    if fit_wrist:
        task_list.append(
            (arm.model.getFrameId(WRIST), joint_estimated_poses[WRIST], (0, 3)),
        )
    q = global_params.get("q", None)
    q = solve_tasks(
        task_list,
        viz=viz,
        q_init=q,
        arm=arm,
        progress_bar=progress_bar
    )
    global_params["q"] = q
    return q, joint_3d_positions, original_joint_3d_positions


def coarse_inverse_kinematics_initialization(
    estimated_poses: List,
    arm_side: str = RIGHT,
    visualize_ik_iterations: bool = True
) -> Tuple[List[dict], dict]:
    """Coarse initialization using inverse kinematics (fit shoulder and elbow 3D positions)

    Args:
        estimated_poses (List): List of estimated poses from mediapipe
        arm_side (str, optional): right or left. Defaults to RIGHT.
        visualize_ik_iterations (bool, optional): Not recommended. View the IK iterations while fitting
        Defaults to True.

    Returns:
        Tuple[List[np.ndarray], dict]: q_list, global_params
        list of states, and global params in case you need to retieve
        the arm model or visualization afterwards
    """
    # @TODO: go backwards to refine
    global_params = {}
    q_list = []
    joint_3d_positions_list = []
    original_joint_3d_positions_list = []
    valid_frames_list = []
    invalid_frames_list = []
    build_arm_model(global_params=global_params, headless=visualize_ik_iterations)
    for frame_idx in tqdm(range(len(estimated_poses))):
        global_params["frame_idx"] = frame_idx
        q_estimation, joint_3d_positions, original_joint_3d_positions = update_arm_model_inverse_kinematics(
            estimated_poses, global_params=global_params, arm_side=arm_side,
            fit_elbow=True, fit_wrist=True, scale_constant=1., progress_bar=False)
        if joint_3d_positions is not None:  # SUPPORT EMPTY FRAMES
            valid_frames_list.append(frame_idx)
            q_list.append(q_estimation)
            joint_3d_positions_list.append(joint_3d_positions)
            original_joint_3d_positions_list.append(original_joint_3d_positions)
    config = {
        "invalid_frames_list": invalid_frames_list,
        "valid_frames_list": valid_frames_list,
        "q": q_list,
        "3dpoints": {
            "list": joint_3d_positions_list,
            SHOULDER: [dic[SHOULDER] for dic in joint_3d_positions_list],
            ELBOW: [dic[ELBOW] for dic in joint_3d_positions_list],
            WRIST: [dic[WRIST] for dic in joint_3d_positions_list],
        },
        "3dpoints_original": {
            "list": original_joint_3d_positions_list,
        }
    }
    return config, global_params


def coarse_inverse_kinematics_visualization(q_list: List[np.ndarray], global_params: dict) -> None:
    """Visualize step by step the IK solution

    Args:
        q_list (List[np.ndarray]): List of q states
        global_params (dict): dict containing viz and arm in case these already exist

    """
    build_arm_model(global_params=global_params, headless=False)
    interactive_replay_sequence(
        {"prediction": q_list},
        global_params["viz"]
    )
    return global_params
