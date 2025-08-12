import torch

def map_to_lie_algebra(v):
    """Map a point in R^N to the tangent space at the identity, i.e.
    to the Lie Algebra
    Arg:
        v = vector in R^N, (..., 3) in our case
    Return:
        R = v converted to Lie Algebra element, (3,3) in our case"""

    # make sure this is a sample from R^3
    assert v.size()[-1] == 3

    R_x = v.new_tensor([[ 0., 0., 0.],
                        [ 0., 0.,-1.],
                        [ 0., 1., 0.]])

    R_y = v.new_tensor([[ 0., 0., 1.],
                        [ 0., 0., 0.],
                        [-1., 0., 0.]])

    R_z = v.new_tensor([[ 0.,-1., 0.],
                        [ 1., 0., 0.],
                        [ 0., 0., 0.]])

    R = R_x * v[..., 0, None, None] + \
        R_y * v[..., 1, None, None] + \
        R_z * v[..., 2, None, None]

    return R


def map_to_lie_vector(X):
    """Map Lie algebra in ordinary (3, 3) matrix rep to vector.

    In literature known as 'vee' map.

    inverse of map_to_lie_algebra
    """
    return torch.stack((-X[..., 1, 2], X[..., 0, 2], -X[..., 0, 1]), -1)


def rodrigues(v):
    theta = v.norm(p=2, dim=-1, keepdim=True)
    
    # Handle the zero vector case (no rotation)
    if torch.allclose(theta, torch.zeros_like(theta)):
        return torch.eye(3, device=v.device, dtype=v.dtype)
    
    # normalize K
    K = map_to_lie_algebra(v / theta)
    I = torch.eye(3, device=v.device, dtype=v.dtype)
    R = I + torch.sin(theta)[..., None]*K \
        + (1. - torch.cos(theta))[..., None]*(K@K)
    return R


def log_map(rotmat):
    """Convert rotation matrices to Lie algebra vectors (logarithm map)."""
    # Compute the angle of rotation from trace
    trace = rotmat[..., 0, 0] + rotmat[..., 1, 1] + rotmat[..., 2, 2]
    cos_theta = (trace - 1) / 2
    
    # Clamp to avoid numerical issues
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    
    # Handle small rotations and singularities
    small_angle_mask = theta < 1e-6
    
    # For small angles, use first-order approximation
    sin_theta = torch.sin(theta)
    
    # Handle division by zero for sin(theta)
    factor = torch.where(
        small_angle_mask,
        torch.ones_like(theta) / 2,
        (theta / (2 * sin_theta))
    )
    
    # Extract the skew-symmetric component
    skew = (rotmat - rotmat.transpose(-1, -2)) * factor.unsqueeze(-1).unsqueeze(-1)
    
    # Convert to vector form [x, y, z] using the lietools function
    log_rot = map_to_lie_vector(skew)
    
    return log_rot
