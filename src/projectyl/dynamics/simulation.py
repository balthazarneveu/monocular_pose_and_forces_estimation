from projectyl.dynamics.armmodel import ArmRobot
from projectyl.utils.properties import ELBOW, SHOULDER, WRIST
import numpy as np
import pinocchio as pin
from typing import Tuple


def rk2_step(
        arm_robot: ArmRobot,
        q: np.ndarray,
        vq: np.ndarray,
        tauq: np.ndarray,
        dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Runge-Kutta 2nd order integration step

    Args:
        arm_robot (ArmRobot): _description_
        q (np.ndarray): config state
        vq (np.ndarray): velocity state (dq/dt)
        tauq (np.ndarray): torque
        dt (float): time interval

    Returns:
        Tuple[np.ndarray, np.ndarray]: q_new, vq_new
    """
    # First step
    M = pin.crba(arm_robot.model, arm_robot.data, q)
    b = pin.nle(arm_robot.model, arm_robot.data, q, vq)
    aq1 = np.linalg.solve(M, tauq - b)

    # Intermediate values
    q_mid = pin.integrate(arm_robot.model, q, vq * dt / 2)
    vq_mid = vq + aq1 * dt / 2

    # Second step
    M_mid = pin.crba(arm_robot.model, arm_robot.data, q_mid)
    b_mid = pin.nle(arm_robot.model, arm_robot.data, q_mid, vq_mid)
    aq2 = np.linalg.solve(M_mid, tauq - b_mid)

    # Update state
    q_new = pin.integrate(arm_robot.model, q, vq_mid * dt)
    vq_new = vq + aq2 * dt

    return q_new, vq_new


def build_simulation(
    arm_robot: ArmRobot,
    T: int = 30,
    DT: float = 1e-2,
    friction_coefficient: float = 0.1,
    initial_torque_modulation: float = 0.0,
    static: bool = False,
) -> Tuple[list, list, list, list, list, list, list]:
    """Record ground truth values during a motion simulation

    Args:
        arm_robot (ArmRobot): Arm robot model
        T (int, optional): Total duration (in intervals). Defaults to 30.
        DT (float, optional): time interval. Defaults to 1e-2.
        friction_coefficient (float, optional): Friction coefficient. Defaults to 0.1
        Note: use friction=0 to do a free fall.
        initial_torque_modulation (float, optional): Initial torque modulation. Defaults to 0.0.
    Returns:
        Tuple[list, list, list, list, list, list, list]:
        gt_q, gt_vq, gt_aq, gt_tauq,
        gt_shoulder_p, gt_elbow_p, gt_wrist_p
    """
    # Set initial conditions
    q = arm_robot.q0.copy()
    if arm_robot.free_elbow:
        q[4:] = np.sqrt(2) / 2
    else:
        q[4] = np.pi / 4
    vq = np.zeros(arm_robot.model.nv)
    aq = np.zeros(arm_robot.model.nv)
    np.random.seed(42)  # For reproducibility
    tauq = initial_torque_modulation * np.random.rand(arm_robot.model.nv)

    shoulder_frame_id = arm_robot.model.getFrameId(SHOULDER)
    elbow_frame_id = arm_robot.model.getFrameId(ELBOW)
    wrist_frame_id = arm_robot.model.getFrameId(WRIST)

    gt_q = []
    gt_vq = []
    gt_aq = []
    gt_tauq = []
    gt_shoulder_p = []
    gt_elbow_p = []
    gt_wrist_p = []

    for _ in range(T):
        if static:
            tauq = pin.rnea(arm_robot.model, arm_robot.data, q, vq, aq)
        # Iterative forward dynamics
        # Compute mass and non linear effects
        M = pin.crba(arm_robot.model, arm_robot.data, q)
        b = pin.nle(arm_robot.model, arm_robot.data, q, vq)

        # Compute accelleration
        aq = np.linalg.solve(M, tauq - b)

        # Retrieve 3D points (forward kinematics)
        pin.framesForwardKinematics(arm_robot.model, arm_robot.data, q)
        shoulder_p = arm_robot.data.oMf[shoulder_frame_id].translation
        elbow_p = arm_robot.data.oMf[elbow_frame_id].translation
        wrist_p = arm_robot.data.oMf[wrist_frame_id].translation

        # Store ground truth var value
        gt_q.append(q.copy())
        gt_vq.append(vq.copy())
        gt_aq.append(aq.copy())
        gt_tauq.append(tauq.copy())
        gt_shoulder_p.append(shoulder_p.copy())
        gt_elbow_p.append(elbow_p.copy())
        gt_wrist_p.append(wrist_p.copy())

        q, vq = rk2_step(arm_robot, q, vq, tauq, DT)
        
        if not static:
            tauq = - friction_coefficient * vq  # Free fall with friction
    return (gt_q, gt_vq, gt_aq, gt_tauq,
            gt_shoulder_p, gt_elbow_p, gt_wrist_p)
