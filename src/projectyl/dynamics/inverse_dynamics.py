import pinocchio as pin
import numpy as np
from typing import Tuple
from projectyl.dynamics.armmodel import ArmRobot
from projectyl.utils.properties import ELBOW, SHOULDER, WRIST


shoulder_frame_id, elbow_frame_id, wrist_frame_id = None, None, None  # 1, 2, 3


def get_frame_id(arm_robot: ArmRobot):
    global shoulder_frame_id, elbow_frame_id, wrist_frame_id
    shoulder_frame_id = arm_robot.model.getFrameId(SHOULDER)
    elbow_frame_id = arm_robot.model.getFrameId(ELBOW)
    wrist_frame_id = arm_robot.model.getFrameId(WRIST)


def full_body_dynamics(
    tq: np.ndarray,
    tvq: np.ndarray,
    taq: np.ndarray,
    T: int,
    arm_robot: ArmRobot
) -> np.ndarray:
    """Retrieve the torque from RNEA

    Args:
        tq (np.ndarray): serie of configuration states q
        tvq (np.ndarray): serie of configuration states velocities vq
            (dq/dt, basically axis angles velocities when q is a quaternion)
        taq (np.ndarray): serie of configuration states accelerations aq
            (d²q/dt²)
        T (int): length of the series
        arm_robot (ArmRobot): Arm robot model

    Returns:
        np.ndarray: Torque from RNEA

    Remarks
    =======
    If the Lagrange dynamics formulation cannot be totally satisfied,
    one can relax by minimizing the difference between:
    - the Lagrange dynamics torques tau_rec predicited from q, vq and aq.
    - the current predicted torques ttauq
    middle point is used here.
    """
    nv = arm_robot.model.nv
    ttau = np.empty((T - 4, nv))

    for i in range(2, T - 2):
        ttau[i - 2] = pin.rnea(arm_robot.model, arm_robot.data, tq[i], tvq[i - 1], taq[i - 2])

    return ttau.flatten()


def get_3D_pose_velocity_acceleration(
    tq: np.ndarray, T: int, DT: float, arm_robot: ArmRobot
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve 3D pose, velocity, and acceleration from forward kinematics.

    Args:
        tq (np.ndarray): Array of joint angles.
        T (int): Number of time steps.
        DT (float): Time step size.
        arm_robot (ArmRobot): Instance of the ArmRobot class.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the 3D pose, velocity, and acceleration.

    """
    global shoulder_frame_id, elbow_frame_id, wrist_frame_id
    if shoulder_frame_id is None or elbow_frame_id is None or wrist_frame_id is None:
        get_frame_id(arm_robot)
    # [p_shoulder_1, p_elbow_1, p_wrist_1, ..., p_shoulder_T, p_elbow_T, p_wrist_T]

    tshoulder_p = np.empty((T, 3))
    telbow_p = np.empty((T, 3))
    twrist_p = np.empty((T, 3))

    for i in range(T):
        # Forward kinematics
        pin.framesForwardKinematics(arm_robot.model, arm_robot.data, tq[i])

        # Predicted 3D points
        tshoulder_p[i] = arm_robot.data.oMf[shoulder_frame_id].translation
        telbow_p[i] = arm_robot.data.oMf[elbow_frame_id].translation
        twrist_p[i] = arm_robot.data.oMf[wrist_frame_id].translation

    # Computes speed and acceleration
    tshoulder_v = (tshoulder_p[2:] - tshoulder_p[:-2]) / (2 * DT)
    telbow_v = (telbow_p[2:] - telbow_p[:-2]) / (2 * DT)
    twrist_v = (twrist_p[2:] - twrist_p[:-2]) / (2 * DT)

    tshoulder_a = (tshoulder_v[2:] - tshoulder_v[:-2]) / (2 * DT)
    telbow_a = (telbow_v[2:] - telbow_v[:-2]) / (2 * DT)
    twrist_a = (twrist_v[2:] - twrist_v[:-2]) / (2 * DT)

    data_pos3D = np.concatenate((tshoulder_p, telbow_p, twrist_p), axis=1).flatten()
    velocity = np.concatenate(
        (tshoulder_v, telbow_v, twrist_v), axis=0).flatten()
    acceleration = np.concatenate(
        (tshoulder_a, telbow_a, twrist_a), axis=0).flatten()
    return data_pos3D, velocity, acceleration


def get_velocity_acceleration(tq: np.ndarray, T: int, DT: float, arm_robot: ArmRobot) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve angular velocity and acceleration from joint angles.
    Use centered finite differences

    Args:
        tq (np.ndarray): Array of joint angles.
        T (int): total number of samples
        DT (float): timestep
        arm_robot (ArmRobot): Robot arm mode

    Returns:
        Tuple[np.ndarray, np.ndarray]: tvq, taq: angular velocity and acceleration
    """
    nv = arm_robot.model.nv
    tvq = np.empty((T - 2, nv))
    # nv=4

    # Pourquoi ces opérations ne sont pas vectorizés dans pinocchio ...
    for i in range(1, T - 1):
        tvq[i - 1] = pin.difference(arm_robot.model, tq[i - 1], tq[i + 1]) / (2 * DT)

    taq = (tvq[2:] - tvq[:-2]) / (2 * DT)

    return tvq, taq


def process_var(var, T, nq) -> Tuple[np.ndarray, np.ndarray]:
    # [q_1, q_2, ..., q_T, tauq_3, tauq_4, ..., tauq_T-2]

    # [ q1 , q2 ,   q3  ,                       , qT-1, qT ]
    # [ -  , -  , tauq_3, tauq_4, ..., tauq_T-2 ,  -  , -  ]
    tq_unnormalized = var[:T * nq]
    tq_unnormalized = tq_unnormalized.reshape(T, nq)
    # tq =[q_1, q_2, ..., q_T]
    ttauq = var[T * nq:]  # ttauq = [tauq_3, tauq_4, ..., tauq_T-2]

    # Get tq
    # Normalize shoulder quaternion
    shoulder_quaternion_unnormalized = tq_unnormalized[:, :4]  # Shouler quaternion 4 first values of q

    shoulder_quaternion_norm = np.linalg.norm(
        shoulder_quaternion_unnormalized,
        axis=1,
        keepdims=True
    )

    shoulder_quaternion_normalized = shoulder_quaternion_unnormalized / shoulder_quaternion_norm

    # Normalize elbow
    if nq == 6:
        elbow_angle_unnormalized = tq_unnormalized[:, 4:].reshape(T, 2)

        elbow_angle_norm = np.linalg.norm(
            elbow_angle_unnormalized,
            axis=1,
            keepdims=True
        )

        elbow_angle_normalized = elbow_angle_unnormalized / elbow_angle_norm
    elif nq == 5:
        elbow_angle_normalized = tq_unnormalized[:, 4:]

    tq = np.concatenate(
        (
            shoulder_quaternion_normalized,
            elbow_angle_normalized
        ),
        axis=1
    )
    assert tq.shape == tq_unnormalized.shape

    return tq, ttauq


# Build the cost function
def objective(var, observed_p, T, DT, arm_robot: ArmRobot, debug=False) -> np.ndarray:
    if debug:
        print(f"{T}, {DT}, var = {var.shape}, observed_p = {observed_p.shape}")
    nq = arm_robot.model.nq
    tq, ttauq = process_var(var, T, nq)
    tvq, taq = get_velocity_acceleration(tq, T, DT, arm_robot)
    tp, tv, ta = get_3D_pose_velocity_acceleration(tq, T, DT, arm_robot)
    ttau = full_body_dynamics(tq, tvq, taq, T, arm_robot)

    if debug:
        print("Diff between 3D pose :", np.linalg.norm(observed_p - tp))
        print("Smooth velocity :", np.linalg.norm(tv))
        print("Smooth acceleration :", np.linalg.norm(ta))
        print("Smooth torque :", np.linalg.norm(ttauq))
        print("Dynamics :", np.linalg.norm(ttau - ttauq))

    res_p = observed_p - tp
    mask_p = np.abs(res_p) > 1
    res_p[mask_p] = 2 * np.sqrt(np.abs(res_p[mask_p])) - 1

    res = np.concatenate([
        (res_p),
        0.6 * tv,
        0.6 * ta,
        0.6 * ttauq,
        2 * (ttau - ttauq),
    ])

    return res
