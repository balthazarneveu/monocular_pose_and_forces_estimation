import pinocchio as pin
import numpy as np
from projectyl.dynamics.armmodel import ArmRobot


def get_pose_velocity_from_state(tq_normalized, tvq, T, DT, arm_robot: ArmRobot):
    # Forward pass
    rec_p = []
    rec_v = []
    shoulder_frame_id = arm_robot.shoulder_frame_id
    elbow_frame_id = arm_robot.elbow_frame_id
    wrist_frame_id = arm_robot.wrist_frame_id

    for t in range(T):
        q = tq_normalized[t]
        vq = tvq[t]
        # (forward kinematics)
        pin.forwardKinematics(arm_robot.model, arm_robot.data, q, vq)
        pin.updateFramePlacements(arm_robot.model, arm_robot.data)
        # Predicted 3D points -> 3x3 points
        shoulder_p = arm_robot.data.oMf[shoulder_frame_id].translation
        elbow_p = arm_robot.data.oMf[elbow_frame_id].translation
        end_effector_p = arm_robot.data.oMf[wrist_frame_id].translation
        # Predicted 3D velocities -> ingnore shoulder 2x3 points
        shoulder_v = pin.getFrameVelocity(arm_robot.model, arm_robot.data,
                                          shoulder_frame_id, pin.ReferenceFrame.WORLD).linear
        elbow_v = pin.getFrameVelocity(arm_robot.model, arm_robot.data, elbow_frame_id, pin.ReferenceFrame.WORLD).linear
        end_effector_v = pin.getFrameVelocity(arm_robot.model, arm_robot.data,
                                              wrist_frame_id, pin.ReferenceFrame.WORLD).linear

        p = np.concatenate([shoulder_p, elbow_p, end_effector_p])
        rec_p.append(p)

        v = np.concatenate([shoulder_v, elbow_v, end_effector_v])
        rec_v.append(v)

    tp = np.vstack(rec_p)
    tv = np.vstack(rec_v)

    ta = (tv[1:] - tv[:-1]) / DT

    return tp, tv, ta


def diff_3D(tp, tp_observed):
    diff = tp - tp_observed
    diff = diff.flatten()

    return diff


def smooth_velocity_acceleration(tv, ta):
    rv = tv.flatten()
    ra = ta.flatten()

    return np.concatenate([rv, ra])


def smooth_torque(ttauq):
    return ttauq.flatten()

# If the Lagrange dynamics formulation cannote be totally satisfied,
# one can relax by minimizing the difference between:
# - the Lagrange dynamics torques tau_rec predicited from q, vq and aq.
# - the current predicted torques ttauq
# question: why not also going backward in time ?


def full_body_dynamics(tq_normalized, tvq, taq, ttauq, T, arm_robot: ArmRobot):

    tau_rec = []

    for t in range(1, T):
        tau = pin.rnea(arm_robot.model, arm_robot.data, tq_normalized[t], tvq[t], taq[t - 1])
        # taq[t - 1] ?

        tau_rec.append(tau)

    ttau = np.vstack(tau_rec)

    diff = ttau - ttauq[1:]
    # ttauq[1:] - cannot compute ttau for t=0

    diff = diff.flatten()

    return diff

# Build the cost function


def objective(txuc: np.ndarray, tp_observed: np.ndarray, T: float, DT: float, arm_robot: ArmRobot, debug=False, ) -> np.ndarray:
    txuc_r = txuc.reshape(T, -1)

    tq = txuc_r[:, :arm_robot.model.nq]
    norm_quat = np.linalg.norm(tq[:, :-1], axis=1, keepdims=True)
    tq_normalized = tq.copy()
    tq_normalized[:, :-1] /= norm_quat

    tvq = txuc_r[:, arm_robot.model.nq: arm_robot.model.nq + arm_robot.model.nv]

    taq = (tvq[1:] - tvq[:-1]) / DT

    ttauq = txuc_r[:, arm_robot.model.nq + arm_robot.model.nv:]

    tp, tv, ta = get_pose_velocity_from_state(tq_normalized, tvq, T, DT, arm_robot)

    # DEBUG
    if debug:
        print("3D joint positions", np.linalg.norm(diff_3D(tp, tp_observed)))
        print("Smooth velocity acceleration", np.linalg.norm(smooth_velocity_acceleration(tv, ta)))
        print("Smooth torque", np.linalg.norm(smooth_torque(ttauq)))
        print("Full body dynamics", np.linalg.norm(full_body_dynamics(tq_normalized, tvq, taq, ttauq, T, arm_robot)))

    res = np.concatenate([
        diff_3D(tp, tp_observed),
        0.1 * smooth_velocity_acceleration(tv, ta),
        0.1 * smooth_torque(ttauq),
        10 * full_body_dynamics(tq_normalized, tvq, taq, ttauq, T, arm_robot),
    ])

    return res
