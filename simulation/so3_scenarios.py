import torch
import roma
from dataclasses import dataclass
from typing import Optional, Dict, Any

from simulation.so3_dynamics import TorqueFrame, TorqueFunction

@dataclass
class ScenarioConfig:
    """Configuration class for simulation scenarios"""
    torque_fn: TorqueFunction
    description: str
    params: Dict[str, Any]


def create_magnetic_torque(B_field: torch.Tensor, magnetic_moment: Optional[torch.Tensor] = None):
    """Physical magnetic torque calculated in world frame
    Returns: TorqueFunction with frame specification"""
    if magnetic_moment is None:
        magnetic_moment = torch.tensor([1.0, 0.0, 0.0])
        
    def torque_fn(t, state):
        device = state['orientation'].device
        R = roma.unitquat_to_rotmat(state['orientation'])  # [B, 3, 3]
        m = magnetic_moment.to(device).expand(R.shape[0], -1)  # [B, 3]
        m_world = torch.bmm(R, m.unsqueeze(-1)).squeeze(-1)  # [B, 3]
        return torch.cross(m_world, B_field.to(device).expand(R.shape[0], -1))
    
    return TorqueFunction(fn=torque_fn, frame=TorqueFrame.WORLD)


def create_orientation_dependent_torque(K: float, reference_quat: Optional[torch.Tensor] = None):
    """Spring-like restoring torque"""
    def torque_fn(t, state):
        device = state['orientation'].device
        q = state['orientation']  # [B, 4]
        if reference_quat is not None:
            q_rel = roma.quat_product(q, roma.quat_inverse(reference_quat.to(device)))
            return -K * roma.unitquat_to_rotvec(q_rel)
        return -K * roma.unitquat_to_rotvec(q)  # [B, 3]
    return TorqueFunction(fn=torque_fn, frame=TorqueFrame.WORLD)

def create_constant_acceleration(acceleration: torch.Tensor):
    """Constant acceleration torque: τ = I⋅a"""
    def torque_fn(t, state):
        device = state['orientation'].device
        batch_size = state['orientation'].shape[0]
        a = acceleration.to(device).expand(batch_size, -1)  # [B, 3]
        # Convert desired acceleration to appropriate torque
        return torch.bmm(state['I'], a.unsqueeze(-1)).squeeze(-1)
    return TorqueFunction(fn=torque_fn, frame=TorqueFrame.BODY)

def create_linear_acceleration(A_matrix: torch.Tensor, bias: Optional[torch.Tensor] = None):
    """Linear acceleration torque: τ = I⋅(Aω + b)"""
    def torque_fn(t, state):
        device = state['orientation'].device
        omega = state['angular_velocity']  # [B, 3]
        batch_size = omega.shape[0]
        
        if A_matrix.dim() == 2:
            A = A_matrix.to(device).expand(batch_size, -1, -1)  # [B, 3, 3]
        else:
            A = A_matrix.to(device)
            
        # Compute desired acceleration Aω + b
        acc = torch.bmm(A, omega.unsqueeze(-1)).squeeze(-1)  # [B, 3]
        if bias is not None:
            acc = acc + bias.to(device).expand(batch_size, -1)
            
        # Convert desired acceleration to appropriate torque
        return torch.bmm(state['I'], acc.unsqueeze(-1)).squeeze(-1)
    return TorqueFunction(fn=torque_fn, frame=TorqueFrame.BODY)

def create_damped_acceleration(damping_matrix: torch.Tensor, natural_frequency: Optional[torch.Tensor] = None):
    """Damped motion torque calculated in body frame"""
    def torque_fn(t, state):
        device = state['orientation'].device
        omega = state['angular_velocity']  # [B, 3]
        batch_size = omega.shape[0]
        
        if damping_matrix.dim() == 2:
            D = damping_matrix.to(device).expand(batch_size, -1, -1)  # [B, 3, 3]
        else:
            D = damping_matrix.to(device)
            
        acc = torch.bmm(D, omega.unsqueeze(-1)).squeeze(-1)  # [B, 3]
        if natural_frequency is not None:
            w = natural_frequency.to(device).expand(batch_size, -1)
            acc = acc + torch.sin(w * t) * w.abs()
            
        return torch.bmm(state['I'], acc.unsqueeze(-1)).squeeze(-1)
    
    return TorqueFunction(fn=torque_fn, frame=TorqueFrame.BODY)


def create_combined_torque(torque_fns: list, weights: Optional[list] = None):
    """Combines multiple torque functions with frame awareness"""
    if weights is None:
        weights = [1.0] * len(torque_fns)
    
    def torque_fn(t, state):
        device = state['orientation'].device

        R = roma.unitquat_to_rotmat(state['orientation'])  # [B, 3, 3]
        R_T = R.transpose(1, 2)
        
        total_body_torque = 0
        for fn_wrapper, w in zip(torque_fns, weights):
            torque = fn_wrapper.fn(t, state)
            # Convert world frame torques to body frame
            if fn_wrapper.frame == TorqueFrame.WORLD:
                torque = torch.bmm(R_T, torque.unsqueeze(-1)).squeeze(-1)
            total_body_torque = total_body_torque + w * torque
        return total_body_torque
    
    return TorqueFunction(fn=torque_fn, frame=TorqueFrame.BODY)


def create_scenarios(config: Dict[str, Any]) -> Dict[str, ScenarioConfig]:
    """Creates scenario dictionary based on full configuration
    
    Args:
        config: Full configuration dictionary containing all scenario parameters
    """
    def to_tensor(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return torch.tensor(x, dtype=torch.float32)
        return x

    scenarios = {
        'FREE_ROTATION': ScenarioConfig(
            torque_fn=TorqueFunction(fn=lambda t, state: torch.zeros_like(state['angular_velocity']), frame=TorqueFrame.BODY),
            description='Free rotation without external torques',
            params={}
        ),
        'MAGNETIC': ScenarioConfig(
            torque_fn=create_magnetic_torque(
                B_field=to_tensor(config['MAGNETIC'].get('B_FIELD', [0., 0., 1.])),
                magnetic_moment=to_tensor(config['MAGNETIC'].get('MAGNETIC_MOMENT', None))
            ),
            description='Magnetic torque with configurable B-field',
            params={'B_FIELD': config['MAGNETIC'].get('B_FIELD'),
                   'MAGNETIC_MOMENT': config['MAGNETIC'].get('MAGNETIC_MOMENT')}
        ),
        'DAMPED': ScenarioConfig(
            torque_fn=create_damped_acceleration(
                damping_matrix=to_tensor(config['DAMPED'].get('DAMPING_MATRIX', [[-0.1, 0.0, 0.0],
                                                                                [0.0, -0.1, 0.0],
                                                                                [0.0, 0.0, -0.1]])),
                natural_frequency=to_tensor(config['DAMPED'].get('NATURAL_FREQUENCY', None))
            ),
            description='Damped motion with optional oscillation',
            params=dict(config['DAMPED'])
        ),
        'DAMPED_MAGNETIC': ScenarioConfig(
            torque_fn=create_combined_torque(
                torque_fns=[
                    create_damped_acceleration(
                        damping_matrix=to_tensor(config['DAMPED'].get('DAMPING_MATRIX')),
                        natural_frequency=to_tensor(config['DAMPED'].get('NATURAL_FREQUENCY'))
                    ),
                    create_magnetic_torque(
                        B_field=to_tensor(config['MAGNETIC'].get('B_FIELD')),
                        magnetic_moment=to_tensor(config['MAGNETIC'].get('MAGNETIC_MOMENT'))
                    )
                ],
                weights=config['DAMPED_MAGNETIC'].get('WEIGHTS', [1.0, 1.0])
            ),
            description='Combined damped motion with magnetic torque',
            params={
                **config['DAMPED'],
                **config['MAGNETIC'],
                'WEIGHTS': config['DAMPED_MAGNETIC'].get('WEIGHTS')
            }
        ),
        'CONSTANT_ACC': ScenarioConfig(
            torque_fn=create_constant_acceleration(
                acceleration=to_tensor(config['CONSTANT_ACC'].get('ACCELERATION', torch.tensor([0.1, 0.0, 0.0])))
            ),
            description='Constant angular acceleration (τ = I⋅a)',
            params={
                **config['CONSTANT_ACC']
            }
        ),
        'LINEAR_ACC': ScenarioConfig(
            torque_fn=create_linear_acceleration(
                A_matrix=to_tensor(config['LINEAR_ACC'].get('A_MATRIX', torch.tensor([[-0.1, 0.0, 0.0],
                                                                                      [0.0, -0.1, 0.0],
                                                                                      [0.0, 0.0, -0.1]]))),
                bias=to_tensor(config['LINEAR_ACC'].get('BIAS', None))
            ),
            description='Linear angular acceleration (τ = I⋅(Aω + b))',
            params={
                **config['LINEAR_ACC']
            }
        ),
    }

    return scenarios
