"""
Simple Euclidean Savitzky-Golay path for isolating polynomial math performance.

This eliminates all SO(3) manifold operations to test if the bottleneck is in:
- SG polynomial mathematics (this will test)
- SO(3) operations like rodrigues, ddexp_so3 (eliminated here)
"""

import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
from typing import Optional


class EuclideanSGPath(diffrax.AbstractPath):
    """Simple Euclidean Savitzky-Golay path - no SO(3) operations."""
    
    p: int
    t_center: float
    data_center: jnp.ndarray  # (input_dim,)
    _t0: float
    _t1: float
    batch_size: int
    input_dim: int
    second_order: bool
    
    # Cached polynomial coefficients
    coeffs: jnp.ndarray  # (batch_size, p+1, input_dim)
    powers: jnp.ndarray  # (p+1,)
    normalizers: jnp.ndarray  # (p+1,)
    
    def __init__(self, data: jnp.ndarray, t: jnp.ndarray, p: int, second_order: bool = False):
        """Initialize Euclidean SG path.
        
        Args:
            data: Input data, shape (batch_size, n_points, input_dim) or (n_points, input_dim)
            t: Time points, shape (n_points,)
            p: Polynomial order
            second_order: Whether to support second-order derivatives
        """
        if data.ndim == 2:
            # Single sequence case: add batch dimension
            data = data[None, ...]  # (1, n_points, input_dim)
            
        if data.ndim != 3:
            raise ValueError(f"Expected data shape (batch_size, n_points, input_dim), got {data.shape}")
            
        batch_size, n_points, input_dim = data.shape
        center_idx = n_points // 2
        
        # Store basic parameters
        t_center = t[center_idx]
        data_center = data[:, center_idx]  # (batch_size, input_dim)
        _t0 = t[0]
        _t1 = t[-1]
        
        # Simple polynomial fitting using least squares
        # Construct basis matrix: each row is [1, t, t^2, ..., t^p] for one time point
        t_rel = t - t_center  # Relative to center
        
        # Build polynomial basis matrix
        powers = jnp.arange(p + 1, dtype=jnp.float32)  # [0, 1, 2, ..., p]
        normalizers = jnp.maximum(1.0, powers)  # [1, 1, 2, 3, ..., p] for numerical stability
        
        # Basis matrix: (n_points, p+1)
        A = (t_rel[:, None] ** powers[None, :]) / normalizers[None, :]
        
        # Fit polynomial coefficients for each batch element and dimension
        # data_rel = data - data_center (relative to center)
        data_rel = data - data_center[:, None, :]  # (batch_size, n_points, input_dim)
        
        # Solve least squares for each batch and dimension
        # A: (n_points, p+1), data_rel: (batch_size, n_points, input_dim)
        # Want coeffs: (batch_size, p+1, input_dim)
        
        AtA = A.T @ A  # (p+1, p+1)
        AtA_inv = jnp.linalg.inv(AtA)  # (p+1, p+1)
        
        # For each batch element: coeffs = AtA_inv @ A.T @ data_rel
        # Use einsum for batch processing
        AtB = jnp.einsum('tp,bti->bpi', A, data_rel)  # (batch_size, p+1, input_dim)
        coeffs = jnp.einsum('pq,bqi->bpi', AtA_inv, AtB)  # (batch_size, p+1, input_dim)
        
        # Initialize attributes
        self.p = p
        self.t_center = t_center
        self.data_center = data_center
        self._t0 = _t0
        self._t1 = _t1
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.second_order = second_order
        
        # Cache polynomial evaluation components
        self.coeffs = coeffs
        self.powers = powers
        self.normalizers = normalizers
    
    @property
    def t0(self):
        return self._t0
        
    @property 
    def t1(self):
        return self._t1
        
    def evaluate(self, t0, t1=None, left=True):
        """Evaluate the path at time t0 or increment between t0 and t1."""
        del left  # Not used for deterministic paths
        
        if t1 is not None:
            # Return increment: path(t1) - path(t0)
            return self._evaluate_single(t1) - self._evaluate_single(t0)
        else:
            # Return path value at t0
            return self._evaluate_single(t0)
    
    def _evaluate_single(self, t_val):
        """Evaluate path at a single time point."""
        t_rel = t_val - self.t_center
        
        # Evaluate polynomial: sum(coeffs * (t_rel^powers / normalizers))
        # coeffs: (batch_size, p+1, input_dim)
        # powers: (p+1,), normalizers: (p+1,)
        
        t_powers = t_rel ** self.powers  # (p+1,)
        t_powers_norm = t_powers / self.normalizers  # (p+1,)
        
        # Polynomial evaluation for each batch element
        values = jnp.einsum('bpi,p->bi', self.coeffs, t_powers_norm)  # (batch_size, input_dim)
        
        # Add back the center value
        result = values + self.data_center  # (batch_size, input_dim)
        
        # Return with time prepended (diffrax convention): (batch_size, 1 + input_dim)
        time_col = jnp.full((self.batch_size, 1), t_val)
        result_with_time = jnp.concatenate([time_col, result], axis=1)
        
        if self.batch_size == 1:
            return result_with_time[0]  # (1 + input_dim,)
        else:
            return result_with_time  # (batch_size, 1 + input_dim)
    
    def derivative(self, t, left=True, order=1):
        """Compute time derivative(s) - pure polynomial math, no SO(3) operations."""
        del left  # Not used for deterministic paths
        
        if order == 2 and not self.second_order:
            raise ValueError("Second-order derivatives requested but second_order=False")
        
        if order not in [1, 2]:
            raise ValueError(f"Only derivative orders 1 and 2 are supported, got {order}")
        
        t_rel = t - self.t_center
        
        if order == 1:
            return self._compute_first_derivative(t_rel)
        else:
            return self._compute_second_derivative(t_rel)
    
    def _compute_first_derivative(self, t_rel):
        """Compute first derivative using simple polynomial differentiation."""
        if self.p == 0:
            # Constant polynomial has zero derivative
            zero_deriv = jnp.zeros((self.batch_size, self.input_dim))
            time_col = jnp.ones((self.batch_size, 1))  # dt/dt = 1
            result = jnp.concatenate([time_col, zero_deriv], axis=1)
            return result[0] if self.batch_size == 1 else result
        
        # First derivative: d/dt sum(c_k * t^k / k!) = sum(c_k * k * t^(k-1) / k!)
        # = sum(c_k * t^(k-1) / (k-1)!) for k >= 1
        
        first_powers = self.powers[1:]  # [1, 2, 3, ..., p] - skip constant term
        first_coeffs = self.coeffs[:, 1:, :]  # (batch_size, p, input_dim) - skip constant
        first_normalizers = self.normalizers[1:]  # [1, 2, 3, ..., p]
        
        if len(first_powers) == 0:
            # No first-order terms
            zero_deriv = jnp.zeros((self.batch_size, self.input_dim))
            time_col = jnp.ones((self.batch_size, 1))
            result = jnp.concatenate([time_col, zero_deriv], axis=1)
            return result[0] if self.batch_size == 1 else result
        
        # Evaluate derivative polynomial
        t_powers = t_rel ** (first_powers - 1)  # t^(k-1)
        derivative_terms = first_powers / first_normalizers  # k / k! = 1 / (k-1)!
        t_powers_weighted = t_powers * derivative_terms  # (p,)
        
        # Sum over polynomial terms
        values = jnp.einsum('bpi,p->bi', first_coeffs, t_powers_weighted)  # (batch_size, input_dim)
        
        # Return with time derivative prepended (dt/dt = 1)
        time_col = jnp.ones((self.batch_size, 1))
        result = jnp.concatenate([time_col, values], axis=1)
        
        return result[0] if self.batch_size == 1 else result
    
    def _compute_second_derivative(self, t_rel):
        """Compute both first and second derivatives."""
        first_deriv = self._compute_first_derivative(t_rel)
        
        if self.p <= 1:
            # Not enough terms for second derivative
            zero_deriv = jnp.zeros((self.batch_size, self.input_dim))
            time_col = jnp.zeros((self.batch_size, 1))  # d2t/dt2 = 0
            second_deriv = jnp.concatenate([time_col, zero_deriv], axis=1)
            
            if self.batch_size == 1:
                return first_deriv, second_deriv[0]
            else:
                return first_deriv, second_deriv
        
        # Second derivative: d2/dt2 sum(c_k * t^k / k!) = sum(c_k * k * (k-1) * t^(k-2) / k!)
        second_powers = self.powers[2:]  # [2, 3, 4, ..., p] - skip constant and linear
        second_coeffs = self.coeffs[:, 2:, :]  # (batch_size, p-1, input_dim)
        second_normalizers = self.normalizers[2:]  # [2, 3, 4, ..., p]
        
        if len(second_powers) == 0:
            # No second-order terms
            zero_deriv = jnp.zeros((self.batch_size, self.input_dim))
            time_col = jnp.zeros((self.batch_size, 1))
            second_deriv = jnp.concatenate([time_col, zero_deriv], axis=1)
            
            if self.batch_size == 1:
                return first_deriv, second_deriv[0]
            else:
                return first_deriv, second_deriv
        
        # Evaluate second derivative polynomial
        t_powers = t_rel ** (second_powers - 2)  # t^(k-2)
        second_derivative_terms = (second_powers * (second_powers - 1)) / second_normalizers  # k*(k-1)/k!
        t_powers_weighted = t_powers * second_derivative_terms  # (p-1,)
        
        # Sum over polynomial terms
        values = jnp.einsum('bpi,p->bi', second_coeffs, t_powers_weighted)  # (batch_size, input_dim)
        
        # Return with time derivative prepended (d2t/dt2 = 0)
        time_col = jnp.zeros((self.batch_size, 1))
        second_deriv = jnp.concatenate([time_col, values], axis=1)
        
        if self.batch_size == 1:
            return first_deriv, second_deriv[0]
        else:
            return first_deriv, second_deriv


def test_euclidean_sg_performance():
    """Test performance of simple Euclidean SG path vs SO(3) version."""
    
    print("="*70)
    print("EUCLIDEAN SG PATH PERFORMANCE TEST")
    print("="*70)
    print("🎯 Goal: Isolate polynomial math vs SO(3) operations performance")
    
    # Create test data - simple vectors instead of rotation matrices
    seq_len = 12
    input_dim = 9  # Same as flattened rotation matrix
    batch_size = 1
    
    t = jnp.linspace(0.0, 1.2, seq_len)
    
    # Create smooth vector data (simulating flattened rotation matrices)
    key = jax.random.key(42)
    data = jax.random.normal(key, (batch_size, seq_len, input_dim)) * 0.1
    
    # Create Euclidean SG path
    euclidean_path = EuclideanSGPath(data, t, p=3, second_order=True)
    
    print(f"📊 Created Euclidean SG path: data={data.shape}, t={t.shape}")
    
    # Test 1: Isolated derivative calls
    print(f"\n🔧 Test 1: Isolated derivative calls...")
    
    # Warmup
    _ = euclidean_path.derivative(0.5, order=2)
    
    # Time isolated calls
    n_calls = 100
    times = []
    
    for _ in range(n_calls):
        jax.device_get(jnp.array(0.0))
        start = time.perf_counter()
        first_deriv, second_deriv = euclidean_path.derivative(0.5, order=2)
        jax.device_get(first_deriv)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_isolated = sum(times) / len(times) * 1000
    print(f"  Average isolated call: {avg_isolated:.3f} ms")
    print(f"  First deriv shape: {first_deriv.shape}")
    print(f"  Second deriv shape: {second_deriv.shape}")
    
    # Test 2: Integration context
    print(f"\n🔧 Test 2: Integration context...")
    
    def ode_with_euclidean_sg(t, z, args):
        """ODE function using Euclidean SG derivatives."""
        first_deriv, second_deriv = euclidean_path.derivative(t, order=2)
        # Combine derivatives (skip time channel)
        combined = first_deriv[1:] + second_deriv[1:]
        return jnp.ones_like(z) * jnp.sum(combined) * 1e-6
    
    # Setup integration
    z0 = jnp.ones(10) * 0.1
    t_span = (0.0, 0.6)
    dt0 = 0.05
    
    # Count calls
    call_count = {'count': 0}
    
    def counting_ode(t, z, args):
        call_count['count'] += 1
        first_deriv, second_deriv = euclidean_path.derivative(t, order=2)
        combined = first_deriv[1:] + second_deriv[1:]
        return jnp.ones_like(z) * jnp.sum(combined) * 1e-6
    
    # Warmup
    solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(ode_with_euclidean_sg),
        solver=diffrax.Euler(),
        t0=t_span[0], t1=t_span[1], dt0=dt0,
        y0=z0,
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=50
    )
    
    # Reset and time integration
    call_count['count'] = 0
    
    jax.device_get(jnp.array(0.0))
    start = time.perf_counter()
    
    solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(counting_ode),
        solver=diffrax.Euler(),
        t0=t_span[0], t1=t_span[1], dt0=dt0,
        y0=z0,
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=50
    )
    
    jax.device_get(solution.ys)
    integration_time = time.perf_counter() - start
    
    total_calls = call_count['count']
    avg_integration = (integration_time * 1000) / total_calls if total_calls > 0 else 0
    
    print(f"  Integration time: {integration_time*1000:.1f} ms")
    print(f"  Total derivative calls: {total_calls}")
    print(f"  Average call during integration: {avg_integration:.3f} ms")
    print(f"  Integration vs isolated ratio: {avg_integration/avg_isolated:.1f}x")
    
    # Comparison with previous SO(3) results
    print(f"\n📊 COMPARISON WITH SO(3) SG PATH:")
    print(f"  SO(3) isolated calls: ~1.7 ms")
    print(f"  SO(3) integration calls: ~633 ms") 
    print(f"  SO(3) regression ratio: ~514x")
    print(f"")
    print(f"  Euclidean isolated calls: {avg_isolated:.3f} ms")
    print(f"  Euclidean integration calls: {avg_integration:.3f} ms")
    print(f"  Euclidean regression ratio: {avg_integration/avg_isolated:.1f}x")
    
    # Analysis
    print(f"\n🔍 ANALYSIS:")
    
    if avg_integration/avg_isolated < 10:
        print("✅ EXCELLENT: Euclidean SG has reasonable integration overhead")
        print("   → The 514x regression is caused by SO(3) operations!")
        print("   → rodrigues, ddexp_so3, compute_angular_velocity_from_coeffs are the bottlenecks")
    elif avg_integration/avg_isolated < 50:
        print("🟡 MODERATE: Euclidean SG has some integration overhead")
        print("   → Both polynomial math AND SO(3) operations contribute to regression")
    else:
        print("🚨 CRITICAL: Even Euclidean SG has major integration issues")
        print("   → The problem is in the polynomial mathematics or diffrax interaction")
    
    return {
        'isolated_ms': avg_isolated,
        'integration_ms': avg_integration,
        'regression_ratio': avg_integration/avg_isolated,
        'total_calls': total_calls
    }


if __name__ == "__main__":
    import time
    results = test_euclidean_sg_performance()