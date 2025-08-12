import torch
import numpy as np
import roma

from utils.lietools import map_to_lie_algebra


def interpolate_quaternions(q_sparse: torch.Tensor, n_points: int) -> torch.Tensor:
    """
    Interpolate between sparse quaternions to get smooth trajectory using Roma's slerp
    
    Args:
        q_sparse: Sparse quaternions [B, T, 4]
        n_points: Number of points desired in interpolated trajectory
        
    Returns:
        q_interp: Interpolated quaternions [B, n_points, 4]
    """
    batch_size, n_sparse, _ = q_sparse.shape
    assert n_points % (n_sparse - 1) == 0, "n_points must be divisible by (n_sparse - 1)"
    points_per_segment = n_points // (n_sparse - 1)
    
    # Ensure quaternion continuity
    q_continuous = ensure_quaternion_continuity(q_sparse)
    
    # Initialize output tensor
    q_interp = torch.zeros(batch_size, n_points, 4, device=q_sparse.device)
    
    # Create interpolation steps for each segment
    steps = torch.linspace(0, 1, points_per_segment, device=q_sparse.device)
    
    # Interpolate between each pair of quaternions
    for i in range(n_sparse - 1):
        q0 = q_continuous[:, i]     # [B, 4]
        q1 = q_continuous[:, i+1]   # [B, 4]
        
        # Use Roma's slerp
        segment_interp = roma.utils.unitquat_slerp(
            q0, 
            q1, 
            steps,
            shortest_arc=True
        )  # [steps, B, 4]
        
        # Store interpolated values
        start_idx = i * points_per_segment
        end_idx = (i + 1) * points_per_segment
        q_interp[:, start_idx:end_idx] = segment_interp.permute(1, 0, 2)
    
    # Add the last point
    q_interp[:, -1] = q_continuous[:, -1]
    
    return q_interp


def ensure_quaternion_continuity(quaternions: torch.Tensor, start_in_north=True) -> torch.Tensor:
    """
    Ensures a continuous quaternion path by selecting the quaternion sign that
    maximizes the dot product with the previous quaternion in the sequence.
    Additionally ensures the first quaternion starts in the northern hemisphere (w >= 0).
    Flips all components of the quaternion to maintain the same rotation path.
    Vectorized implementation for efficiency.
    
    Args:
        quaternions: Tensor of shape (..., N, 4) where N is the sequence length
                    and 4 represents quaternion components in XYZW format
    
    Returns:
        Tensor of same shape with continuous quaternion path starting in northern hemisphere
    """
    # Handle empty sequences or single quaternions
    if quaternions.shape[-2] <= 1:
        return quaternions.clone()
    
    result = quaternions.clone()
    
    # First ensure the trajectory starts in the northern hemisphere

    if start_in_north:
        start_in_south = result[..., 0, 3] < 0
        result[..., 0, :] = torch.where(
            start_in_south.unsqueeze(-1).expand_as(result[..., 0, :]),
            -result[..., 0, :],
            result[..., 0, :]
        )
    
    # Compute dot products between consecutive quaternions
    # Shape: (..., N-1)
    dots = torch.sum(
        result[..., :-1, :] * result[..., 1:, :],
        dim=-1
    )
    
    # Where dot product is negative, we'll need a flip
    # Shape: (..., N-1)
    flip_mask = dots < 0
    
    # Convert to ±1 for multiplication
    flip_signs = torch.ones_like(flip_mask, dtype=torch.float32)
    flip_signs[flip_mask] = -1
    
    # Compute cumulative product of signs to determine total flips needed
    # Shape: (..., N-1)
    cumulative_signs = torch.cumprod(flip_signs, dim=-1)
    
    # Pad the cumulative signs to match sequence length
    # Shape: (..., N)
    flip_signs_full = torch.nn.functional.pad(
        cumulative_signs,
        (1, 0),  # Pad at the beginning
        value=1.0  # First quaternion keeps its sign from hemisphere check
    )
    
    # Apply all flips in one operation
    result = result * flip_signs_full.unsqueeze(-1)
    
    return result


def project_q_onto_S3(q):
    # NOTE: uses ROMA notation of XYZW
    # for [WXYZ] notation we would want to take
    # q[:, 1:]
    return q[:, :3] / np.linalg.norm(q[:, :3], axis=1, keepdims=True)


def project_to_northern_hem(q):
    """
    Project quaternions onto the northern hemisphere.
    
    Args:
    q (torch.Tensor): Input quaternions in XYZW format. Shape: (..., 4)
    
    Returns:
    torch.Tensor: Projected quaternions in XYZW format. Shape: (..., 4)
    """
    # Check if W (last component) is negative
    w_negative = q[..., -1:] < 0
    
    # Create a tensor with -1 where W is negative, 1 otherwise
    multiplier = torch.where(w_negative, -1.0, 1.0)
    
    # Multiply the quaternions by the multiplier
    projected_q = q * multiplier
    
    return projected_q


def normalize_quaternion(q):
    return q / torch.linalg.norm(q, dim=-1, keepdim=True, ord=2)


def quaternion_derivative(q, omega):
    # Assuming omega is a tensor of shape (..., 3)
    omega_quat = torch.cat([torch.zeros_like(omega[..., :1]), omega], dim=-1)
    return 0.5 *roma.quat_product(q, omega_quat)


def random_quaternion(N=1):
    if N>1:
        q = torch.randn((N,4))
    else:
        q = torch.randn(4)
    return normalize_quaternion(q)


def set_seed(seed_value):
    torch.manual_seed(seed_value)


def quaternion_from_axis_angle(axis, angle):
    # Ensure the axis tensor has the same number of dimensions as the angle tensor
    #axis = axis.expand_as(angle)
    axis = axis /torch.linalg.norm(axis, dim=-1, keepdim=True, ord=2)
    w = torch.cos(angle / 2.0)
    xyz = axis * torch.sin(angle / 2.0)
    return torch.cat([w, xyz], dim=-1)


def add_noise_to_quaternion(quaternion, noise_level):
    """
    Add noise to a quaternion using Roma library, handling the double cover properly.
    
    Args:
        quaternion: Quaternion tensor of shape (..., 4) in XYZW format
        noise_level: Standard deviation of the angular noise in radians
    
    Returns:
        Noisy quaternion tensor of same shape as input
    """
    batch_shape = quaternion.shape[:-1]  # Get all dimensions except the last
    
    # Generate random axis (uniformly distributed on unit sphere)
    random_axis = torch.randn(*batch_shape, 3, device=quaternion.device)  # Shape: (..., 3)
    random_axis = random_axis / torch.norm(random_axis, dim=-1, keepdim=True)
    
    # Generate random angles from normal distribution
    angles = torch.normal(
        mean=0.0,
        std=noise_level,
        size=(*batch_shape, 1),  # Match batch dimensions
        device=quaternion.device
    )
    
    # Create noise rotation (using axis-angle)
    noise_quaternion = roma.rotvec_to_unitquat(random_axis * angles)
    
    # Apply noise through quaternion multiplication
    noisy_quaternion = roma.quat_product(quaternion, noise_quaternion)
    
    # Handle double cover: choose the quaternion representation that's closest
    # to the original by checking the dot product
    dot_product = torch.sum(quaternion * noisy_quaternion, dim=-1, keepdim=True)
    
    # Only flip the w coordinate when choosing the other hemisphere
    flipped_quaternion = noisy_quaternion.clone()
    flipped_quaternion[..., 3] = -noisy_quaternion[..., 3]  # Flip only w coordinate (index 3 in XYZW)
    
    noisy_quaternion = torch.where(
        dot_product < 0,
        flipped_quaternion,
        noisy_quaternion
    )
    
    return noisy_quaternion


# def add_noise_to_quaternion(q, noise_level):
#     # Determine the shape for noise generation
#     shape = q.shape[:-1] if q.dim() > 1 else (1,)
# 
#     noise_rotvec = noise_level * torch.randn(shape + (3,), device=q.device)
#     q_rotvec = roma.unitquat_to_rotvec(q)
#     
#     return roma.rotvec_to_unitquat(roma.rotvec_composition([q_rotvec,noise_rotvec]))


def add_noise_to_quaternion_2(q, noise_level):
    # Determine the shape for noise generation
    shape = q.shape[:-1] if q.dim() > 1 else (1,)

    # Generate small rotation quaternions for each axis
    axis_x = torch.tensor([1.0, 0.0, 0.0], device=q.device).repeat(shape + (1,)) * noise_level * torch.randn(shape + (1,), device=q.device)
    axis_y = torch.tensor([0.0, 1.0, 0.0], device=q.device).repeat(shape + (1,)) * noise_level * torch.randn(shape + (1,), device=q.device)
    axis_z = torch.tensor([0.0, 0.0, 1.0], device=q.device).repeat(shape + (1,)) * noise_level * torch.randn(shape + (1,), device=q.device)

    noise_x = roma.rotvec_to_unitquat(axis_x) 
    noise_y = roma.rotvec_to_unitquat(axis_y) 
    noise_z = roma.rotvec_to_unitquat(axis_z) 
    
    # Combine the noise quaternions with the original quaternion
    q_noisy = q
    for noise_q in [noise_x, noise_y, noise_z]:
        q_noisy = normalize_quaternion(roma.quat_product(q_noisy, noise_q))
    
    return q_noisy


def quat_omega(w: torch.Tensor) -> torch.Tensor:
    """
    Function to generate the \Omega(\omega) matrix in the kinematic differential equations for quaternions.
    
    ...
    
    Parameters
    ----------
    w : torch.Tensor
        Angular velocity
        
    Returns
    -------
    Q : torch.Tensor
        Matrix for KDEs of quaternions
        
    Notes
    -----
    Q = \Omega(w) = \[-S(w)  w \] \in su(2)
                    \[-w^{T} 0 \]
                    
    """
    shape = w.shape[:-1] + (4,4)
    S_w = map_to_lie_algebra(v=w)
    
    Q = torch.zeros(shape, device=w.device)
    Q[..., :3, :3] = -S_w
    Q[..., -1, :3] = -w
    Q[..., :3, -1] = w
    
    return Q


def quaternion_derivative_2(q, omega):
    # Assuming omega is a tensor of shape (..., 3)
    omega_ = quat_omega(omega).float()
    
    return 0.5 * torch.einsum("...ab,...b-> ...a",omega_,q.float())
