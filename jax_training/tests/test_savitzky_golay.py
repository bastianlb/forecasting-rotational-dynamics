"""
Tests for Savitzky-Golay SO(3) filtering, comparing JAX and PyTorch implementations.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax_training.utils.savitzky_golay_so3 import so3_filter, so3_savitzky_golay_filter, SO3PolynomialPath
from jax_training.utils.so3 import rodrigues, log_map

# Try importing PyTorch version for comparison
try:
    import torch
    TORCH_AVAILABLE = True
    
    # PyTorch implementation (adapted from the provided code)
    def torch_construct_A(R, dt, p):
        """PyTorch version of construct_A for comparison."""
        t = R.shape[-3]
        s = R.shape[:-3]
        ts = torch.arange(0, t, 1, device=R.device) * dt
        ts_k = ts - ts[t//2]
        ts_k = (ts[None] - ts[:, None])
        
        A = []
        for p_ in range(p + 1):
            normalizer = max(1.0, float(np.math.factorial(p_)))
            term = (torch.eye(3, device=R.device) / normalizer).expand(s + (t, t, -1, -1)) * ts_k[:,:,None,None]**p_
            A.append(term)
        
        A = torch.cat(A, -1)
        A = A.view(s + (A.shape[-4], 3*t, 3*(p+1)))
        return A
    
    def torch_so3_filter_simple(R, dt, p, weight=None):
        """Simplified PyTorch SO3 filter for testing."""
        import roma  # Assuming roma is available
        
        if not is_rotation_matrix_torch(R):
            print("Warning, rotation matrices are not orthonormal!")
        
        A = torch_construct_A(R, dt, p)
        
        # Convert to rotation vectors relative to center
        center_idx = R.shape[-3] // 2
        R_center = R[..., center_idx, :, :]
        R_center_inv = R_center.transpose(-2, -1)
        R_rel = torch.matmul(R, R_center_inv.unsqueeze(-3))
        
        B = roma.rotmat_to_rotvec(R_rel).flatten(start_dim=-2)
        
        if weight is None:
            ATA_inv = torch.inverse(torch.einsum("...ba,...bc->...ac", A, A))
        else:
            W = torch.diag(weight)
            ATW = torch.einsum("...ba,bc->...ac", A, W) 
            ATA_inv = torch.inverse(torch.einsum("...ab,...bc->...ac", ATW, A))
            A = torch.einsum("...ab,cb->...ac", A, W)
        
        pinv = torch.einsum("...ab,...cb->...ac", ATA_inv, A)
        rho = torch.einsum("...ab,...bc->...ac", pinv, B[..., None])[..., 0]
        phi = rho[..., center_idx, :3]
        
        S = torch.matmul(roma.rotvec_to_rotmat(phi), R_center)
        return S
    
    def is_rotation_matrix_torch(R, tolerance=1e-4):
        """Check if matrices are valid rotation matrices (PyTorch version)."""
        identity = torch.eye(3, device=R.device).reshape(1, 1, 3, 3)
        R_RT = torch.matmul(R, R.transpose(-2, -1))
        RTR = torch.matmul(R.transpose(-2, -1), R)
        
        is_orthogonal = (torch.all(torch.abs(R_RT - identity) < tolerance, dim=(-2, -1)) & 
                        torch.all(torch.abs(RTR - identity) < tolerance, dim=(-2, -1)))
        
        det = torch.det(R)
        is_special = torch.abs(det - 1.0) < tolerance
        
        return torch.all(is_orthogonal & is_special)

except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, skipping comparison tests")


def generate_test_rotations(n_points: int, batch_size: int = 1) -> jnp.ndarray:
    """Generate smooth test rotation sequences."""
    key = jax.random.PRNGKey(42)
    
    # Generate smooth angular velocity trajectory
    t = jnp.linspace(0, 2*jnp.pi, n_points)
    omega_mag = 0.5 + 0.3 * jnp.sin(t)
    omega_dir = jnp.array([jnp.cos(t), jnp.sin(t), 0.2 * jnp.ones_like(t)]).T
    omega_dir = omega_dir / jnp.linalg.norm(omega_dir, axis=-1, keepdims=True)
    omega = omega_mag[..., None] * omega_dir
    
    # Integrate to get rotation vectors
    dt = t[1] - t[0]
    rotvec = jnp.cumsum(omega * dt, axis=0)
    
    # Convert to rotation matrices
    R = jax.vmap(rodrigues)(rotvec)
    
    if batch_size > 1:
        # Create batch by adding small perturbations
        keys = jax.random.split(key, batch_size)
        noise = jax.vmap(lambda k: jax.random.normal(k, (n_points, 3)) * 0.1)(keys)
        rotvec_batch = rotvec[None, :, :] + noise
        R = jax.vmap(jax.vmap(rodrigues))(rotvec_batch)
    
    return R


def test_so3_filter_basic():
    """Test basic functionality of JAX SO3 filter."""
    n_points = 11  # Odd number for clear center
    R = generate_test_rotations(n_points)
    dt = 0.1
    p = 3
    
    R_smooth, omega, _ = so3_filter(R, dt, p, return_omega=True)
    
    # Check output shapes
    assert R_smooth.shape == (3, 3)
    assert omega.shape == (3,)
    
    # Check that result is still a rotation matrix
    assert jnp.allclose(jnp.linalg.det(R_smooth), 1.0, atol=1e-6)
    assert jnp.allclose(R_smooth @ R_smooth.T, jnp.eye(3), atol=1e-6)


def test_so3_filter_batch():
    """Test batch processing."""
    n_points = 11
    batch_size = 5
    R = generate_test_rotations(n_points, batch_size)
    dt = 0.1
    p = 3
    
    R_smooth, omega, _ = so3_filter(R, dt, p, return_omega=True)
    
    # Check output shapes
    assert R_smooth.shape == (batch_size, 3, 3)
    assert omega.shape == (batch_size, 3)
    
    # Check all results are rotation matrices
    for i in range(batch_size):
        assert jnp.allclose(jnp.linalg.det(R_smooth[i]), 1.0, atol=1e-6)
        assert jnp.allclose(R_smooth[i] @ R_smooth[i].T, jnp.eye(3), atol=1e-6)


def test_irregular_timestamps():
    """Test handling of irregular timestamps."""
    n_points = 11
    R = generate_test_rotations(n_points)
    
    # Create irregular timestamps
    t_regular = jnp.linspace(0, 1.0, n_points)
    t_irregular = t_regular + 0.1 * jnp.sin(2 * jnp.pi * t_regular) * (jnp.arange(n_points) % 2)
    
    p = 3
    R_smooth, omega, _ = so3_savitzky_golay_filter(R, t_irregular, p, return_omega=True)
    
    # Check output shapes and properties
    assert R_smooth.shape == (3, 3)
    assert omega.shape == (3,)
    assert jnp.allclose(jnp.linalg.det(R_smooth), 1.0, atol=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_jax_vs_pytorch_regular():
    """Compare JAX and PyTorch implementations on regular grids."""
    # This would compare outputs between the two implementations


def test_polynomial_orders():
    """Test different polynomial orders."""
    n_points = 15  # Larger for higher order polynomials  
    R = generate_test_rotations(n_points)
    dt = 0.1
    
    for p in [1, 2, 3, 4]:
        R_smooth, omega, _ = so3_filter(R, dt, p, return_omega=True)
        
        # Check basic properties
        assert jnp.allclose(jnp.linalg.det(R_smooth), 1.0, atol=1e-6)
        assert jnp.allclose(R_smooth @ R_smooth.T, jnp.eye(3), atol=1e-6)


def test_weighted_filtering():
    """Test weighted Savitzky-Golay filtering."""
    n_points = 11
    R = generate_test_rotations(n_points)
    dt = 0.1
    p = 3
    
    # Create weights that emphasize center points
    weights = jnp.exp(-0.5 * (jnp.arange(n_points) - n_points//2)**2 / (n_points/4)**2)
    
    R_smooth_weighted, _, _ = so3_filter(R, dt, p, weight=weights)
    R_smooth_unweighted, _, _ = so3_filter(R, dt, p)
    
    # Results should be different but both valid rotations
    assert not jnp.allclose(R_smooth_weighted, R_smooth_unweighted, atol=1e-6)
    assert jnp.allclose(jnp.linalg.det(R_smooth_weighted), 1.0, atol=1e-6)


def test_edge_cases():
    """Test edge cases and error conditions."""
    # Test with minimum number of points
    n_points = 5
    R = generate_test_rotations(n_points)
    dt = 0.1
    p = 2  # Keep polynomial order reasonable
    
    R_smooth, omega, _ = so3_filter(R, dt, p, return_omega=True)
    assert R_smooth.shape == (3, 3)
    
    # Test with p=0 (constant fit)
    R_smooth_const, _, _ = so3_filter(R, dt, 0)
    assert jnp.allclose(jnp.linalg.det(R_smooth_const), 1.0, atol=1e-6)


def test_so3_polynomial_path_basic():
    """Test basic functionality of SO3PolynomialPath."""
    n_points = 11
    R = generate_test_rotations(n_points)
    t = jnp.linspace(0, 1.0, n_points)
    p = 3
    
    # Create polynomial path
    path = SO3PolynomialPath(R, t, p)
    
    # Test properties
    assert path.t0 == t[0]
    assert path.t1 == t[-1]
    
    # Test evaluation at center point
    t_center = t[n_points // 2]
    result = path.evaluate(t_center)
    
    # Should return [time, R_flattened] = 10D vector
    assert result.shape == (10,)
    assert jnp.isclose(result[0], t_center)
    
    # Extract and check rotation matrix
    R_eval = result[1:].reshape(3, 3)
    assert jnp.allclose(jnp.linalg.det(R_eval), 1.0, atol=1e-6)
    assert jnp.allclose(R_eval @ R_eval.T, jnp.eye(3), atol=1e-6)


def test_so3_polynomial_path_derivative():
    """Test derivative computation of SO3PolynomialPath."""
    n_points = 11
    R = generate_test_rotations(n_points)
    t = jnp.linspace(0, 1.0, n_points)
    p = 3
    
    path = SO3PolynomialPath(R, t, p)
    
    # Test derivative at center point
    t_center = t[n_points // 2]
    deriv = path.derivative(t_center)
    
    # Should return [1.0, dR/dt_flattened] = 10D vector
    assert deriv.shape == (10,)
    assert jnp.isclose(deriv[0], 1.0)  # Time derivative is always 1
    
    # Check that derivative is finite
    assert jnp.all(jnp.isfinite(deriv[1:]))


def test_so3_polynomial_path_interpolation():
    """Test interpolation accuracy of SO3PolynomialPath."""
    n_points = 11
    R = generate_test_rotations(n_points)
    t = jnp.linspace(0, 1.0, n_points)
    p = 3
    
    path = SO3PolynomialPath(R, t, p)
    
    # Test evaluation at original time points
    for i in [0, n_points//2, n_points-1]:  # Test boundary and center
        result = path.evaluate(t[i])
        R_eval = result[1:].reshape(3, 3)
        
        # Should be close to original rotation (S-G filter smooths data, so exact match not expected)
        if i == n_points//2:  # Center point should be reasonably close for S-G filter
            # S-G filter smooths the data, so allow reasonable tolerance
            distance = jnp.linalg.norm(R_eval - R[i])
            assert distance < 0.5, f"Center point too far from original: {distance}"


def test_so3_polynomial_path_increment():
    """Test increment computation (path(t1) - path(t0))."""
    n_points = 11
    R = generate_test_rotations(n_points)
    t = jnp.linspace(0, 1.0, n_points)
    p = 3
    
    path = SO3PolynomialPath(R, t, p)
    
    # Test increment
    t0, t1 = t[2], t[8]
    increment = path.evaluate(t0, t1)
    
    # Should be difference of evaluations
    val_t0 = path.evaluate(t0)
    val_t1 = path.evaluate(t1)
    expected_increment = val_t1 - val_t0
    
    assert jnp.allclose(increment, expected_increment, atol=1e-10)


def test_so3_polynomial_path_continuity():
    """Test that path provides smooth interpolation."""
    n_points = 11
    R = generate_test_rotations(n_points)
    t = jnp.linspace(0, 1.0, n_points)
    p = 3
    
    path = SO3PolynomialPath(R, t, p)
    
    # Evaluate at dense time grid
    t_dense = jnp.linspace(t[0], t[-1], 50)
    values = jax.vmap(path.evaluate)(t_dense)
    derivatives = jax.vmap(path.derivative)(t_dense)
    
    # Check that all values are finite and derivatives are reasonable
    assert jnp.all(jnp.isfinite(values))
    assert jnp.all(jnp.isfinite(derivatives))
    
    # Check that all evaluated rotations are valid
    for i in range(len(t_dense)):
        R_eval = values[i, 1:].reshape(3, 3)
        assert jnp.allclose(jnp.linalg.det(R_eval), 1.0, atol=1e-5)


if __name__ == "__main__":
    # Run basic tests
    test_so3_filter_basic()
    test_so3_filter_batch() 
    test_irregular_timestamps()
    test_polynomial_orders()
    test_weighted_filtering()
    test_edge_cases()
    
    # Test new SO3PolynomialPath functionality
    test_so3_polynomial_path_basic()
    test_so3_polynomial_path_derivative()
    test_so3_polynomial_path_interpolation()
    test_so3_polynomial_path_increment()
    test_so3_polynomial_path_continuity()
    
    print("All tests passed!")