# # def build
# global_params = {}
# build_arm_model(global_params, headless=False)

from projectyl.dynamics.armmodel import ArmRobot
from projectyl.utils.properties import ELBOW, SHOULDER, WRIST
import numpy as np
import pinocchio as pin
# arm_robot.model.createData()


def build_simulation(arm_robot: ArmRobot, DT=1e-2, T=30):
    q = arm_robot.q0.copy()
    q[-1] = 0.5
    vq = np.zeros(arm_robot.model.nv)
    aq = np.zeros(arm_robot.model.nv)
    tauq = 0 * np.random.rand(arm_robot.model.nv)

    shoulder_frame_id = arm_robot.model.getFrameId(SHOULDER)
    elbow_frame_id = arm_robot.model.getFrameId(ELBOW)
    wrist_frame_id = arm_robot.model.getFrameId(WRIST)

    rec_ground_truth_q = []
    rec_ground_truth_vq = []
    rec_ground_truth_aq = []
    rec_ground_truth_tauq = []
    rec_ground_truth_shoulder_p = []
    rec_ground_truth_elbow_p = []
    rec_ground_truth_wrist_p = []

    for _ in range(T):
        # Iterative forward dynamics

        # Compute mass and non linear effects
        M = pin.crba(arm_robot.model, arm_robot.data, q)
        b = pin.nle(arm_robot.model, arm_robot.data, q, vq)

        # Compute accelleration
        aq = np.linalg.solve(M, tauq - b)

        # tauq = pin.rnea(arm_robot.model, arm_robot.data, q, vq, aq)

        # Retrieve 3D points (forward kinematics)
        pin.framesForwardKinematics(arm_robot.model, arm_robot.data, q)
        shoulder_p = arm_robot.data.oMf[shoulder_frame_id].translation
        elbow_p = arm_robot.data.oMf[elbow_frame_id].translation
        wrist_p = arm_robot.data.oMf[wrist_frame_id].translation

        # Store ground truth var value
        rec_ground_truth_q.append(q.copy())
        rec_ground_truth_vq.append(vq.copy())
        rec_ground_truth_aq.append(aq.copy())
        rec_ground_truth_tauq.append(tauq.copy())
        rec_ground_truth_shoulder_p.append(shoulder_p.copy())
        rec_ground_truth_elbow_p.append(elbow_p.copy())
        rec_ground_truth_wrist_p.append(wrist_p.copy())
        vq += aq * DT
        q = pin.integrate(arm_robot.model, q, vq * DT)
        tauq *= 0.1
    return (rec_ground_truth_q, rec_ground_truth_vq, rec_ground_truth_aq, rec_ground_truth_tauq,
            rec_ground_truth_shoulder_p, rec_ground_truth_elbow_p, rec_ground_truth_wrist_p)
