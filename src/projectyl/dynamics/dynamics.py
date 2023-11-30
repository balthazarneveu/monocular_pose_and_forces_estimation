import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.special import huber
from projectyl.utils.properties import ELBOW, SHOULDER, WRIST
from projectyl.utils.arm import plot_optimization_curves

from projectyl.dynamics.inverse_kinematics import build_arm_model
from projectyl.utils.arm import interactive_replay_sequence
from projectyl.utils.io import Dump
from interactive_pipe.helper import _private
from typing import List, Tuple, Dict, Any


def process_var(var, T, arm_robot) -> Tuple[np.ndarray, np.ndarray]:
    # [q_1, q_2, ..., q_T, tauq_3, tauq_4, ..., tauq_T-2]

    # [ q1 , q2    q3  ,                       , qT-1, qT]
    # [ -    - , tauq_3, tauq_4, ..., tauq_T-2 ,  -   -  ]
    nq = arm_robot.model.nq
    tq_unnormalized = var[:T * nq].reshape(T, nq)  # tq =[q_1, q_2, ..., q_T]
    ttauq = var[T * nq:]  # ttauq = [tauq_3, tauq_4, ..., tauq_T-2]

    # Get tq
    shoulder_quaternion_unnormalized = tq_unnormalized[:, :4]  # Shouler quaternion 4 premières valeurs de q
    elbow_angle = tq_unnormalized[:, 4].reshape(T, 1)

    shoulder_quaternion_norm = np.linalg.norm(
        shoulder_quaternion_unnormalized,
        axis=1,
        keepdims=True
    )

    shoulder_quaternion_normalized = shoulder_quaternion_unnormalized / shoulder_quaternion_norm

    tq = np.concatenate(
        (
            shoulder_quaternion_normalized,
            elbow_angle
        ),
        axis=1
    )
    assert tq.shape == tq_unnormalized.shape

    return tq, ttauq


def get_velocity_acceleration(arm_robot, tq, T, nv, DT):

    tvq = np.empty((T - 2, nv))
    # nv=4

    # Pourquoi ces opérations ne sont pas vectorizés dans pinocchio ...
    for i in range(1, T - 1):
        tvq[i - 1] = pin.difference(arm_robot.model, tq[i - 1], tq[i + 1]) / (2 * DT)

    taq = (tvq[2:] - tvq[:-2]) / (2 * DT)

    return tvq, taq
