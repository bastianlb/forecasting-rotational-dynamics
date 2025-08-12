"""
Consolidated test for irregular sampling capability of JAX SG filter.

This test demonstrates the key advantage of the JAX implementation: 
native support for irregular time sampling.
"""

import pytest
import numpy as np
import jax.numpy as jnp
import jax
from pathlib import Path

from jax_training.utils.savitzky_golay_so3 import so3_savitzky_golay_filter
from jax_training.utils.so3 import log_map, rodrigues


class TestIrregularSampling:
    """Test suite for irregular sampling functionality."""

    def generate_rotation_trajectory(self, t):
        """Generate rotation trajectory at given time points."""
        # Complex trajectory with multiple frequency components
        angle = 0.5 * np.sin(2*t) + 0.2 * np.cos(5*t) + 0.1 * np.sin(0.5*t)
        
        # Create rotation matrices (rotation around z-axis)
        n_points = len(t)
        R = jnp.zeros((n_points, 3, 3))
        
        cos_angles = jnp.cos(angle)
        sin_angles = jnp.sin(angle)
        
        R = R.at[:, 0, 0].set(cos_angles)
        R = R.at[:, 0, 1].set(-sin_angles)
        R = R.at[:, 1, 0].set(sin_angles)
        R = R.at[:, 1, 1].set(cos_angles)
        R = R.at[:, 2, 2].set(1.0)
        
        return R, angle

    def add_measurement_noise(self, R, noise_level=0.1):
        """Add realistic measurement noise to rotation matrices."""
        # Convert to rotation vectors, add noise, convert back
        rotvec = jax.vmap(log_map)(R)
        noise = jnp.array(np.random.normal(0, noise_level, rotvec.shape))
        rotvec_noisy = rotvec + noise
        R_noisy = jax.vmap(rodrigues)(rotvec_noisy)
        
        return R_noisy

    def create_irregular_patterns(self, t_regular):
        """Create various irregular sampling patterns."""
        patterns = {}
        dt_nominal = (t_regular[-1] - t_regular[0]) / (len(t_regular) - 1)
        
        # 1. Random jitter (±20% of nominal dt)
        jitter = np.random.uniform(-0.2, 0.2, len(t_regular)) * dt_nominal
        patterns['jitter'] = t_regular + jitter
        
        # 2. Missing samples (simulate dropped measurements)
        drop_mask = np.random.random(len(t_regular)) > 0.15  # Drop 15% of samples
        patterns['dropped'] = t_regular[drop_mask]
        
        # 3. Burst sampling (periods of dense/sparse sampling)
        t_burst = []
        t_curr = t_regular[0]
        i = 0
        while t_curr < t_regular[-1] and i < 100:  # Safety limit
            if i % 20 < 5:  # Dense sampling for 5 steps
                dt = dt_nominal * 0.3
            else:  # Sparse sampling for 15 steps
                dt = dt_nominal * 1.5
            t_burst.append(t_curr)
            t_curr += dt
            i += 1
        patterns['burst'] = np.array(t_burst[:-1])
        
        return patterns

    def test_jitter_pattern(self):
        """Test filtering with jittered sampling."""
        np.random.seed(42)
        
        # Create regular timeline
        t_regular = np.linspace(0, 2*np.pi, 30)
        
        # Create jittered timeline
        dt_nominal = (t_regular[-1] - t_regular[0]) / (len(t_regular) - 1)
        jitter = np.random.uniform(-0.1, 0.1, len(t_regular)) * dt_nominal
        t_jitter = t_regular + jitter
        
        # Generate trajectory
        R_gt, angle_gt = self.generate_rotation_trajectory(t_jitter)
        R_noisy = self.add_measurement_noise(R_gt, noise_level=0.1)
        
        # Apply filter
        R_filtered, omega, coeffs = so3_savitzky_golay_filter(
            R_noisy, 
            jnp.array(t_jitter), 
            p=3,
            return_omega=True,
            return_coefficients=True
        )
        
        # Verify results
        assert R_filtered.shape == (3, 3), f"Expected (3, 3), got {R_filtered.shape}"
        assert omega.shape == (3,), f"Expected (3,), got {omega.shape}"
        assert coeffs.shape == (12,), f"Expected (12,), got {coeffs.shape}"  # 3*(3+1) = 12
        
        # Check that rotation matrix is valid
        assert jnp.allclose(jnp.linalg.det(R_filtered), 1.0, atol=1e-6)
        assert jnp.allclose(R_filtered @ R_filtered.T, jnp.eye(3), atol=1e-6)

    def test_dropped_samples(self):
        """Test filtering with dropped samples."""
        np.random.seed(42)
        
        # Create regular timeline
        t_regular = np.linspace(0, 2*np.pi, 30)
        
        # Drop random samples
        drop_mask = np.random.random(len(t_regular)) > 0.2  # Drop 20% of samples
        t_dropped = t_regular[drop_mask]
        
        # Generate trajectory
        R_gt, angle_gt = self.generate_rotation_trajectory(t_dropped)
        R_noisy = self.add_measurement_noise(R_gt, noise_level=0.1)
        
        # Apply filter
        R_filtered, omega, coeffs = so3_savitzky_golay_filter(
            R_noisy, 
            jnp.array(t_dropped), 
            p=3,
            return_omega=True,
            return_coefficients=True
        )
        
        # Verify results
        assert R_filtered.shape == (3, 3)
        assert omega.shape == (3,)
        assert coeffs.shape == (12,)
        
        # Check that rotation matrix is valid
        assert jnp.allclose(jnp.linalg.det(R_filtered), 1.0, atol=1e-6)
        assert jnp.allclose(R_filtered @ R_filtered.T, jnp.eye(3), atol=1e-6)

    def test_burst_sampling(self):
        """Test filtering with burst sampling pattern."""
        np.random.seed(42)
        
        # Create burst sampling pattern
        t_regular = np.linspace(0, 2*np.pi, 30)
        dt_nominal = (t_regular[-1] - t_regular[0]) / (len(t_regular) - 1)
        
        t_burst = []
        t_curr = t_regular[0]
        i = 0
        while t_curr < t_regular[-1] and i < 50:  # Safety limit
            if i % 10 < 3:  # Dense sampling for 3 steps
                dt = dt_nominal * 0.4
            else:  # Sparse sampling for 7 steps
                dt = dt_nominal * 1.2
            t_burst.append(t_curr)
            t_curr += dt
            i += 1
        
        t_burst = np.array(t_burst)
        
        # Generate trajectory
        R_gt, angle_gt = self.generate_rotation_trajectory(t_burst)
        R_noisy = self.add_measurement_noise(R_gt, noise_level=0.1)
        
        # Apply filter
        R_filtered, omega, coeffs = so3_savitzky_golay_filter(
            R_noisy, 
            jnp.array(t_burst), 
            p=3,
            return_omega=True,
            return_coefficients=True
        )
        
        # Verify results
        assert R_filtered.shape == (3, 3)
        assert omega.shape == (3,)
        assert coeffs.shape == (12,)
        
        # Check that rotation matrix is valid
        assert jnp.allclose(jnp.linalg.det(R_filtered), 1.0, atol=1e-6)
        assert jnp.allclose(R_filtered @ R_filtered.T, jnp.eye(3), atol=1e-6)

    def test_multiple_polynomial_orders(self):
        """Test filtering with different polynomial orders."""
        np.random.seed(42)
        
        # Create irregular timeline
        t_regular = np.linspace(0, 2*np.pi, 25)
        dt_nominal = (t_regular[-1] - t_regular[0]) / (len(t_regular) - 1)
        jitter = np.random.uniform(-0.1, 0.1, len(t_regular)) * dt_nominal
        t_irregular = t_regular + jitter
        
        # Generate trajectory
        R_gt, angle_gt = self.generate_rotation_trajectory(t_irregular)
        R_noisy = self.add_measurement_noise(R_gt, noise_level=0.1)
        
        # Test different polynomial orders
        for p in [1, 2, 3, 4]:
            R_filtered, omega, coeffs = so3_savitzky_golay_filter(
                R_noisy, 
                jnp.array(t_irregular), 
                p=p,
                return_omega=True,
                return_coefficients=True
            )
            
            # Verify results
            assert R_filtered.shape == (3, 3), f"p={p}: Expected (3, 3), got {R_filtered.shape}"
            assert omega.shape == (3,), f"p={p}: Expected (3,), got {omega.shape}"
            assert coeffs.shape == (3*(p+1),), f"p={p}: Expected ({3*(p+1)},), got {coeffs.shape}"
            
            # Check that rotation matrix is valid
            assert jnp.allclose(jnp.linalg.det(R_filtered), 1.0, atol=1e-6), f"p={p}: Invalid determinant"
            assert jnp.allclose(R_filtered @ R_filtered.T, jnp.eye(3), atol=1e-6), f"p={p}: Not orthogonal"

    def test_custom_center_points(self):
        """Test filtering with different center points."""
        np.random.seed(42)
        
        # Create irregular timeline
        t_regular = np.linspace(0, 2*np.pi, 21)
        dt_nominal = (t_regular[-1] - t_regular[0]) / (len(t_regular) - 1)
        jitter = np.random.uniform(-0.1, 0.1, len(t_regular)) * dt_nominal
        t_irregular = t_regular + jitter
        
        # Generate trajectory
        R_gt, angle_gt = self.generate_rotation_trajectory(t_irregular)
        R_noisy = self.add_measurement_noise(R_gt, noise_level=0.1)
        
        # Test different center points
        for center_idx in [len(t_irregular)//4, len(t_irregular)//2, 3*len(t_irregular)//4]:
            R_filtered, omega, coeffs = so3_savitzky_golay_filter(
                R_noisy, 
                jnp.array(t_irregular), 
                p=3,
                center_idx=center_idx,
                return_omega=True,
                return_coefficients=True
            )
            
            # Verify results
            assert R_filtered.shape == (3, 3), f"center_idx={center_idx}: Expected (3, 3), got {R_filtered.shape}"
            assert omega.shape == (3,), f"center_idx={center_idx}: Expected (3,), got {omega.shape}"
            assert coeffs.shape == (12,), f"center_idx={center_idx}: Expected (12,), got {coeffs.shape}"
            
            # Check that rotation matrix is valid
            assert jnp.allclose(jnp.linalg.det(R_filtered), 1.0, atol=1e-6), f"center_idx={center_idx}: Invalid determinant"
            assert jnp.allclose(R_filtered @ R_filtered.T, jnp.eye(3), atol=1e-6), f"center_idx={center_idx}: Not orthogonal"

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        np.random.seed(42)
        
        # Test with minimum number of points
        t_min = np.array([0.0, 0.1, 0.2, 0.3, 0.4])  # 5 points for p=3
        R_gt, _ = self.generate_rotation_trajectory(t_min)
        R_noisy = self.add_measurement_noise(R_gt, noise_level=0.05)
        
        # Should work with p=2 (needs 3 points minimum)
        R_filtered, _, _ = so3_savitzky_golay_filter(
            R_noisy, 
            jnp.array(t_min), 
            p=2
        )
        
        assert R_filtered.shape == (3, 3)
        assert jnp.allclose(jnp.linalg.det(R_filtered), 1.0, atol=1e-6)
        
        # Test with very small time intervals
        t_small = np.array([0.0, 0.001, 0.002, 0.003, 0.004])
        R_gt_small, _ = self.generate_rotation_trajectory(t_small)
        R_noisy_small = self.add_measurement_noise(R_gt_small, noise_level=0.01)
        
        R_filtered_small, _, _ = so3_savitzky_golay_filter(
            R_noisy_small, 
            jnp.array(t_small), 
            p=2
        )
        
        assert R_filtered_small.shape == (3, 3)
        assert jnp.allclose(jnp.linalg.det(R_filtered_small), 1.0, atol=1e-6)

    def test_weighted_filtering(self):
        """Test weighted filtering with irregular sampling."""
        np.random.seed(42)
        
        # Create irregular timeline
        t_regular = np.linspace(0, 2*np.pi, 20)
        dt_nominal = (t_regular[-1] - t_regular[0]) / (len(t_regular) - 1)
        jitter = np.random.uniform(-0.1, 0.1, len(t_regular)) * dt_nominal
        t_irregular = t_regular + jitter
        
        # Generate trajectory
        R_gt, angle_gt = self.generate_rotation_trajectory(t_irregular)
        R_noisy = self.add_measurement_noise(R_gt, noise_level=0.1)
        
        # Create weights (higher weight for central points)
        center_idx = len(t_irregular) // 2
        weights = np.exp(-0.1 * np.abs(np.arange(len(t_irregular)) - center_idx))
        weights = jnp.array(weights)
        
        # Apply weighted filter
        R_filtered, omega, coeffs = so3_savitzky_golay_filter(
            R_noisy, 
            jnp.array(t_irregular), 
            p=3,
            weight=weights,
            return_omega=True,
            return_coefficients=True
        )
        
        # Verify results
        assert R_filtered.shape == (3, 3)
        assert omega.shape == (3,)
        assert coeffs.shape == (12,)
        
        # Check that rotation matrix is valid
        assert jnp.allclose(jnp.linalg.det(R_filtered), 1.0, atol=1e-6)
        assert jnp.allclose(R_filtered @ R_filtered.T, jnp.eye(3), atol=1e-6)
        
        # Compare with unweighted version
        R_unweighted, _, _ = so3_savitzky_golay_filter(
            R_noisy, 
            jnp.array(t_irregular), 
            p=3
        )
        
        # Results should be different (but both valid)
        assert not jnp.allclose(R_filtered, R_unweighted, atol=1e-6), "Weighted and unweighted results should differ"


if __name__ == "__main__":
    pytest.main([__file__])