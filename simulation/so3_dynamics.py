from enum import Enum, auto
from dataclasses import dataclass
from typing import Callable

import torch
import roma


class TorqueFrame(Enum):
    """Specifies the reference frame of a torque function's output"""
    BODY = auto()    # Torque is in body frame
    WORLD = auto()   # Torque is in world frame


@dataclass
class TorqueFunction:
    """Wrapper for torque functions with frame specification"""
    fn: Callable
    frame: TorqueFrame


class RigidBody:
    def __init__(self, moment_of_inertia, initial_orientation, initial_angular_velocity):
        """
        Args:
            moment_of_inertia: [3] or [B, 3] tensor
            initial_orientation: [B, 4] tensor
            initial_angular_velocity: [B, 3] tensor
        """
        batch_size = initial_orientation.shape[0]

        def expand_matrix_to_batch(x):
            return x.unsqueeze(0).expand(batch_size, -1, -1)
        
        I_inv = torch.inverse(moment_of_inertia)
        # Handle broadcasting of moment of inertia
        if moment_of_inertia.dim() == 2:
            moment_of_inertia = expand_matrix_to_batch(moment_of_inertia)
            I_inv = expand_matrix_to_batch(I_inv)


        self.I = moment_of_inertia  # [B, 3, 3]
        self.I_inv = I_inv  # [B, 3, 3]
        
        self.state = torch.cat([
            initial_orientation,  # [B, 4]
            initial_angular_velocity  # [B, 3]
        ], dim=-1)  # [B, 7]


def dynamics(t, state, rigid_body, torque_fn_wrapper=TorqueFunction):
    """Vectorized rotational dynamics on SO(3)
    
    All scenarios reduce to the same fundamental equations:
    dq/dt = 1/2 * q ⊗ ω
    dω/dt = I⁻¹(τ_external - ω × (Iω))
    
    For forced motion scenarios:
    - Setting I = I (identity) and τ_external = Ia gives dω/dt = a for constant acceleration
    - Setting I = I and τ_external = I(Aω + b) gives dω/dt = Aω + b for linear acceleration
    """
    batch_size = state.shape[0]
    orientation, angular_velocity = torch.split(state, [4, 3], dim=-1)
    
    # Normalize quaternion
    orientation = orientation / torch.norm(orientation, dim=-1, keepdim=True)

    external_torque = torque_fn_wrapper.fn(t, {
        'orientation': orientation,
        'angular_velocity': angular_velocity,
        'I': rigid_body.I,
        'I_inv': rigid_body.I_inv,
    })
    
    # Convert world frame torques to body frame if needed
    if torque_fn_wrapper.frame == TorqueFrame.WORLD:
        R = roma.unitquat_to_rotmat(orientation)
        external_torque = torch.bmm(R.transpose(1, 2), external_torque.unsqueeze(-1)).squeeze(-1)

    
    # Angular acceleration from Euler's equations
    Iw = torch.bmm(rigid_body.I, angular_velocity.unsqueeze(-1)).squeeze(-1)
    gyro_torque = -torch.cross(angular_velocity, Iw, dim=-1)
    total_torque = external_torque + gyro_torque
    angular_acceleration = torch.bmm(rigid_body.I_inv, total_torque.unsqueeze(-1)).squeeze(-1)
    
    # Quaternion kinematics
    pure_quat = torch.cat([
        angular_velocity,
        torch.zeros(batch_size, 1, device=state.device)
    ], dim=-1)
    q_dot = 0.5 * roma.quat_product(pure_quat, orientation)
    
    return torch.cat([q_dot, angular_acceleration], dim=-1)
