"""
Micro-optimized derivative computation for SO3PolynomialPath.

Goal: Eliminate the 1.2ms overhead by minimizing intermediate arrays and function calls.
This probably won't solve the 467x integration slowdown, but worth testing.
"""

import jax
import jax.numpy as jnp
# Will be imported in test function


@jax.jit
def optimized_second_order_derivative(
    phi_coeffs_single,      # (p+1, 3)
    R_center_single,        # (3, 3) 
    first_coeffs,           # (p, 3)
    second_coeffs,          # (p-1, 3)
    t_rel,                  # scalar
    powers_phi,             # (p+1,)
    phi_normalizers,        # (p+1,)
    powers_first,           # (p,)
    powers_second           # (p-1,)
):
    """Ultra-optimized second-order derivative computation.
    
    Minimizes intermediate arrays and function calls by doing everything inline.
    """
    
    # === POLYNOMIAL EVALUATION (inline, minimal allocations) ===
    
    # Phi computation (base polynomial)
    t_powers_phi = t_rel ** powers_phi  # (p+1,)
    phi = jnp.sum(phi_coeffs_single * (t_powers_phi / phi_normalizers)[:, None], axis=0)  # (3,)
    
    # First derivative computation (phi_dot)
    if first_coeffs.shape[0] > 0:
        t_powers_first = t_rel ** powers_first  # (p,) - reuse powers
        phi_dot = jnp.sum(first_coeffs * t_powers_first[:, None], axis=0)  # (3,)
    else:
        phi_dot = jnp.zeros(3)
    
    # Second derivative computation (phi_ddot) 
    if second_coeffs.shape[0] > 0:
        t_powers_second = t_rel ** powers_second  # (p-1,) - reuse powers
        phi_ddot = jnp.sum(second_coeffs * t_powers_second[:, None], axis=0)  # (3,)
    else:
        phi_ddot = jnp.zeros(3)
    
    # === SO(3) MANIFOLD OPERATIONS (inline where possible) ===
    
    # Angular velocity from coefficients (inline computation)
    omega = compute_angular_velocity_from_coeffs(phi, phi_dot)
    
    # Rodrigues and current rotation
    R_correction = rodrigues(phi)
    R_current = R_correction @ R_center_single
    
    # Lie algebra mapping and first derivative
    omega_hat = map_to_lie_algebra(omega)
    dR_dt = omega_hat @ R_current
    
    # === SECOND DERIVATIVE COMPUTATION ===
    
    # ddexp_so3 term (this is expensive but necessary)
    ddexp_term = ddexp_so3(phi, phi_dot) @ phi_dot
    
    # Second angular velocity term
    dexp_term = compute_angular_velocity_from_coeffs(phi, phi_ddot)
    
    # Combined alpha term
    alpha = ddexp_term + dexp_term
    alpha_hat = map_to_lie_algebra(alpha)
    
    # omega_hat squared (reuse omega_hat)
    omega_hat_sq = omega_hat @ omega_hat
    
    # Second derivative matrix operation
    d2R_dt2 = (alpha_hat + omega_hat_sq) @ R_current
    
    # === FINAL CONCATENATION (single allocation) ===
    
    # Create result arrays directly - minimize allocations
    first_deriv = jnp.concatenate([jnp.array([1.0]), dR_dt.flatten()])
    second_deriv = jnp.concatenate([jnp.array([0.0]), d2R_dt2.flatten()])
    
    return first_deriv, second_deriv


@jax.jit  
def optimized_first_order_derivative(
    phi_coeffs_single,      # (p+1, 3)
    R_center_single,        # (3, 3)
    first_coeffs,           # (p, 3)
    t_rel,                  # scalar
    powers_phi,             # (p+1,)
    phi_normalizers,        # (p+1,)
    powers_first            # (p,)
):
    """Ultra-optimized first-order derivative computation."""
    
    # Phi computation
    t_powers_phi = t_rel ** powers_phi
    phi = jnp.sum(phi_coeffs_single * (t_powers_phi / phi_normalizers)[:, None], axis=0)
    
    # First derivative computation
    if first_coeffs.shape[0] > 0:
        t_powers_first = t_rel ** powers_first
        phi_dot = jnp.sum(first_coeffs * t_powers_first[:, None], axis=0)
    else:
        phi_dot = jnp.zeros(3)
    
    # SO(3) operations (inline)
    omega = compute_angular_velocity_from_coeffs(phi, phi_dot)
    R_correction = rodrigues(phi)
    R_current = R_correction @ R_center_single
    omega_hat = map_to_lie_algebra(omega)
    dR_dt = omega_hat @ R_current
    
    # Result
    return jnp.concatenate([jnp.array([1.0]), dR_dt.flatten()])


# Pre-compiled vmapped versions
optimized_first_order_vmap = jax.jit(jax.vmap(
    optimized_first_order_derivative, 
    in_axes=(0, 0, 0, None, None, None, None)
))

optimized_second_order_vmap = jax.jit(jax.vmap(
    optimized_second_order_derivative,
    in_axes=(0, 0, 0, 0, None, None, None, None, None)
))


def create_optimized_derivative_method(path_object):
    """Create an optimized derivative method for an SO3PolynomialPath object."""
    
    def optimized_derivative(t, left=True, order=1):
        """Optimized derivative computation with minimal overhead."""
        del left  # Not used
        
        if order == 2 and not path_object.second_order:
            raise ValueError("Second-order derivatives requested but second_order=False")
        
        if order not in [1, 2]:
            raise ValueError(f"Only derivative orders 1 and 2 are supported, got {order}")
        
        # Relative time
        t_rel = t - path_object.t_center
        
        # DIRECT call to optimized functions - minimal method overhead
        if order == 1:
            result = optimized_first_order_vmap(
                path_object.phi_coeffs, path_object.R_center, path_object.first_deriv_coeffs,
                t_rel, path_object.powers_phi, path_object.phi_normalizers, path_object.powers_first
            )
            return result[0] if path_object.batch_size == 1 else result
        else:
            result = optimized_second_order_vmap(
                path_object.phi_coeffs, path_object.R_center, 
                path_object.first_deriv_coeffs, path_object.second_deriv_coeffs,
                t_rel, path_object.powers_phi, path_object.phi_normalizers, 
                path_object.powers_first, path_object.powers_second
            )
            first_result, second_result = result
            if path_object.batch_size == 1:
                return first_result[0], second_result[0]
            else:
                return first_result, second_result
    
    return optimized_derivative


def test_optimization():
    """Test the optimized derivative method."""
    
    print("="*70)
    print("OPTIMIZED DERIVATIVE METHOD TEST")
    print("="*70)
    print("🎯 Goal: Eliminate 1.2ms overhead with micro-optimizations")
    
    # Import required modules
    import sys
    from pathlib import Path
    import time
    
    # Add path and import
    parent_dir = str(Path(__file__).resolve().parent)
    sys.path.insert(0, parent_dir)
    
    from savitzky_golay_so3 import SO3PolynomialPath, compute_angular_velocity_from_coeffs
    from so3 import rodrigues, ddexp_so3, map_to_lie_algebra
    
    # Update global functions for JIT compilation
    global compute_angular_velocity_from_coeffs, rodrigues, ddexp_so3, map_to_lie_algebra
    
    seq_len = 12
    t = jnp.linspace(0.0, 1.2, seq_len)
    
    key = jax.random.key(42)
    angles = jax.random.normal(key, (1, seq_len, 3)) * 0.01
    
    def angle_to_rotmat(angle):
        angle_norm = jnp.linalg.norm(angle)
        k = jnp.where(angle_norm < 1e-6, jnp.array([0., 0., 1.]), angle / angle_norm)
        K = jnp.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return jnp.where(angle_norm < 1e-6, jnp.eye(3),
                        jnp.eye(3) + jnp.sin(angle_norm) * K + (1 - jnp.cos(angle_norm)) * (K @ K))
    
    R = jax.vmap(jax.vmap(angle_to_rotmat))(angles)
    sg_path = SO3PolynomialPath(R=R, t=t, p=3, weight=None, second_order=True)
    
    # Create optimized derivative method
    optimized_deriv = create_optimized_derivative_method(sg_path)
    
    print(f"📊 Created optimized derivative method")
    
    # Test original vs optimized
    print(f"\n🔧 Comparing original vs optimized...")
    
    # Original method timing
    _ = sg_path.derivative(0.5, order=2)  # Warmup
    
    original_times = []
    for _ in range(1000):
        jax.device_get(jnp.array(0.0))
        start = time.perf_counter()
        result_orig = sg_path.derivative(0.5, order=2)
        jax.device_get(result_orig[0])
        elapsed = time.perf_counter() - start
        original_times.append(elapsed)
    
    avg_original = sum(original_times) / len(original_times) * 1_000_000
    
    # Optimized method timing
    _ = optimized_deriv(0.5, order=2)  # Warmup
    
    optimized_times = []
    for _ in range(1000):
        jax.device_get(jnp.array(0.0))
        start = time.perf_counter()
        result_opt = optimized_deriv(0.5, order=2)
        jax.device_get(result_opt[0])
        elapsed = time.perf_counter() - start
        optimized_times.append(elapsed)
    
    avg_optimized = sum(optimized_times) / len(optimized_times) * 1_000_000
    
    # Verify results are the same
    orig_first, orig_second = sg_path.derivative(0.5, order=2)
    opt_first, opt_second = optimized_deriv(0.5, order=2)
    
    first_close = jnp.allclose(orig_first, opt_first, atol=1e-6)
    second_close = jnp.allclose(orig_second, opt_second, atol=1e-6)
    
    print(f"\n📊 RESULTS:")
    print(f"  Original method:    {avg_original:.1f} μs")
    print(f"  Optimized method:   {avg_optimized:.1f} μs")
    print(f"  Speedup:            {avg_original/avg_optimized:.1f}x")
    print(f"  Results identical:  {first_close and second_close}")
    
    if avg_optimized < avg_original * 0.5:
        print(f"\n✅ SIGNIFICANT IMPROVEMENT: {avg_original/avg_optimized:.1f}x speedup!")
    elif avg_optimized < avg_original * 0.8:
        print(f"\n🟡 MODERATE IMPROVEMENT: {avg_original/avg_optimized:.1f}x speedup")
    else:
        print(f"\n❌ MINIMAL IMPROVEMENT: Only {avg_original/avg_optimized:.1f}x speedup")
    
    print(f"\n🔍 ANALYSIS:")
    if avg_optimized > 100:  # Still > 100μs
        print(f"   Even optimized method takes {avg_optimized:.1f} μs")
        print(f"   → Still much slower than expected microsecond performance")
        print(f"   → The fundamental issue is not micro-optimizations")
    else:
        print(f"   Optimized method achieves microsecond performance!")
        print(f"   → Micro-optimizations successfully eliminated overhead")
    
    return {
        'original_us': avg_original,
        'optimized_us': avg_optimized,
        'speedup': avg_original/avg_optimized,
        'results_match': first_close and second_close
    }


if __name__ == "__main__":
    import time
    results = test_optimization()