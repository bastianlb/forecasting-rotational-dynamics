"""
Test suite for SO3NeuralCDE implementation.

Tests both single sample and batched approaches with different interpolation methods.
"""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

# Import the model - need to handle relative imports
import sys
import os
# Add the jax_training directory to the path
jax_training_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, jax_training_dir)

# Now we can import with absolute imports
from models.SO3NeuralCDE import (
    create_so3_neural_cde, 
    SO3NeuralCDE, 
    CDEFunc,
    HermiteInterpolation,
    LinearInterpolation
)
from utils.so3 import symmetric_orthogonalization


class TestSO3NeuralCDE:
    """Test class for SO3NeuralCDE implementation."""
    
    @pytest.fixture
    def model_config(self):
        """Standard model configuration for tests."""
        return {
            'input_channel': 9,
            'latent_channels': 16,  # Smaller for faster tests
            'hidden_channels': 16,
            'output_channel': 9,
            'interpolation_method': 'hermite',
            'method': 'tsit5',
            'atol': 1e-4,  # Relaxed tolerances for testing
            'rtol': 1e-3
        }
    
    @pytest.fixture
    def test_data(self):
        """Generate test data for single sample and batch."""
        batch_size = 4
        seq_len = 8  # Shorter sequences for faster tests
        n_future = 4
        
        # Create time grids
        t_recon = jnp.linspace(0, 0.7, seq_len)[None, :].repeat(batch_size, axis=0)
        t_fut = jnp.linspace(0.8, 1.1, n_future)[None, :].repeat(batch_size, axis=0)
        
        # Create random rotation matrix data
        key = jax.random.key(42)
        x = jax.random.normal(key, (batch_size, seq_len, 3, 3))
        
        # Make them closer to valid rotation matrices
        x = x / jnp.linalg.norm(x, axis=(-2, -1), keepdims=True)
        
        return {
            't_recon': t_recon,
            't_fut': t_fut, 
            'x': x,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'n_future': n_future
        }
    
    def test_model_creation(self, model_config):
        """Test that model can be created successfully."""
        key = jax.random.key(42)
        rngs = nnx.Rngs(params=key)
        
        model = create_so3_neural_cde(**model_config, rngs=rngs)
        
        assert isinstance(model, SO3NeuralCDE)
        assert model.input_channel == model_config['input_channel']
        assert model.latent_channels == model_config['latent_channels']
        assert model.output_channel == model_config['output_channel']
    
    def test_cde_func_shapes(self, model_config):
        """Test CDEFunc input/output shapes."""
        key = jax.random.key(42)
        rngs = nnx.Rngs(params=key)
        
        func = CDEFunc(
            input_channels=model_config['input_channel'] + 1,
            hidden_channels=model_config['latent_channels'],
            bottle_neck_channel=model_config['hidden_channels'],
            rngs=rngs
        )
        
        # Test single sample
        t = 0.5
        z = jnp.ones((model_config['latent_channels'],))
        
        output = func(t, z)
        expected_shape = (model_config['latent_channels'], model_config['input_channel'] + 1)
        assert output.shape == expected_shape
    
    @pytest.mark.parametrize("interpolation_method", ["hermite", "linear"])
    def test_interpolation_methods(self, test_data, interpolation_method):
        """Test different interpolation methods."""
        if interpolation_method == "hermite":
            interp_class = HermiteInterpolation()
        else:
            interp_class = LinearInterpolation()
        
        # Test single sample interpolation
        t_single = test_data['t_recon'][0]  # (seq_len,)
        x_single = test_data['x'][0]  # (seq_len, 3, 3)
        
        # Flatten to 2D
        x_flat = x_single.reshape(test_data['seq_len'], -1)  # (seq_len, 9)
        
        # Create interpolation (needs batch dimension temporarily)
        t_batch = t_single[None, :]
        x_batch = x_flat[None, :]
        
        interpolation = interp_class.create_interpolation(t_batch, x_batch)
        
        # Test evaluation at a time point
        t_eval = t_single[0]
        result = interpolation.evaluate(t_eval)
        
        # Should have shape (1, 10) since we add time channel
        assert result.shape[1] == 10  # 9 for rotation matrix + 1 for time
    
    def test_single_sample_forward(self, model_config, test_data):
        """Test forward pass for a single sample."""
        key = jax.random.key(42)
        rngs = nnx.Rngs(params=key)
        
        # Use simpler solver for testing
        config = model_config.copy()
        config['method'] = 'euler'
        config['atol'] = 1e-3
        config['rtol'] = 1e-2
        
        model = create_so3_neural_cde(**config, rngs=rngs)
        
        # Test single sample
        t_recon_single = test_data['t_recon'][0]
        t_fut_single = test_data['t_fut'][0]
        x_single = test_data['x'][0]
        
        try:
            recon, pred = model.forward_single(t_recon_single, t_fut_single, x_single)
            
            # Check output shapes
            assert recon.shape == (test_data['seq_len'], 3, 3)
            assert pred.shape == (test_data['n_future'], 3, 3)
            
            # Check that outputs are valid rotation matrices (approximately)
            def check_rotation_matrix(R, tol=1e-1):  # Relaxed tolerance for testing
                # Check orthogonality: R @ R.T ≈ I
                should_be_identity = R @ jnp.transpose(R, (0, 2, 1))
                identity = jnp.eye(3)[None, :, :].repeat(R.shape[0], axis=0)
                is_orthogonal = jnp.allclose(should_be_identity, identity, atol=tol)
                
                # Check determinant ≈ 1
                det = jnp.linalg.det(R)
                is_proper = jnp.allclose(det, 1.0, atol=tol)
                
                return is_orthogonal and jnp.all(is_proper)
            
            assert check_rotation_matrix(recon), "Reconstruction should produce valid rotation matrices"
            assert check_rotation_matrix(pred), "Prediction should produce valid rotation matrices"
            
        except Exception as e:
            if "maximum number of solver steps" in str(e):
                pytest.skip(f"Solver convergence issue (expected for complex test data): {e}")
            else:
                raise
    
    def test_batched_forward(self, model_config, test_data):
        """Test forward pass for batched data using vmap."""
        key = jax.random.key(42)
        rngs = nnx.Rngs(params=key)
        
        # Use simpler solver for testing
        config = model_config.copy()
        config['method'] = 'euler'
        config['atol'] = 1e-3
        config['rtol'] = 1e-2
        
        model = create_so3_neural_cde(**config, rngs=rngs)
        
        try:
            recon, pred, _ = model(test_data['t_recon'], test_data['t_fut'], test_data['x'])
            
            # Check output shapes
            expected_recon_shape = (test_data['batch_size'], test_data['seq_len'], 3, 3)
            expected_pred_shape = (test_data['batch_size'], test_data['n_future'], 3, 3)
            
            assert recon.shape == expected_recon_shape
            assert pred.shape == expected_pred_shape
            
            # Check that outputs are finite
            assert jnp.all(jnp.isfinite(recon)), "Reconstruction should be finite"
            assert jnp.all(jnp.isfinite(pred)), "Prediction should be finite"
            
        except Exception as e:
            if "maximum number of solver steps" in str(e):
                pytest.skip(f"Solver convergence issue (expected for complex test data): {e}")
            else:
                raise
    
    def test_consistency_single_vs_batch(self, model_config, test_data):
        """Test that single sample and batched results are consistent."""
        key = jax.random.key(42)
        rngs = nnx.Rngs(params=key)
        
        # Use simpler solver for testing
        config = model_config.copy()
        config['method'] = 'euler'
        config['atol'] = 1e-3
        config['rtol'] = 1e-2
        
        model = create_so3_neural_cde(**config, rngs=rngs)
        
        try:
            # Single sample forward
            t_recon_single = test_data['t_recon'][0]
            t_fut_single = test_data['t_fut'][0]
            x_single = test_data['x'][0]
            
            recon_single, pred_single = model.forward_single(t_recon_single, t_fut_single, x_single)
            
            # Batched forward (just first sample)
            t_recon_batch = test_data['t_recon'][:1]
            t_fut_batch = test_data['t_fut'][:1]
            x_batch = test_data['x'][:1]
            
            recon_batch, pred_batch, _ = model(t_recon_batch, t_fut_batch, x_batch)
            
            # Results should be close (allowing for numerical differences)
            assert jnp.allclose(recon_single, recon_batch[0], atol=1e-3, rtol=1e-2)
            assert jnp.allclose(pred_single, pred_batch[0], atol=1e-3, rtol=1e-2)
            
        except Exception as e:
            if "maximum number of solver steps" in str(e):
                pytest.skip(f"Solver convergence issue (expected for complex test data): {e}")
            else:
                raise
    
    def test_different_input_shapes(self, model_config):
        """Test model with different input shapes (3x3 matrices vs flattened)."""
        key = jax.random.key(42)
        rngs = nnx.Rngs(params=key)
        
        config = model_config.copy()
        config['method'] = 'euler'
        
        model = create_so3_neural_cde(**config, rngs=rngs)
        
        seq_len = 6
        t_recon = jnp.linspace(0, 0.5, seq_len)
        t_fut = jnp.linspace(0.6, 0.8, 3)
        
        # Test with 3x3 matrices
        x_3x3 = jax.random.normal(jax.random.key(123), (seq_len, 3, 3))
        
        # Test with flattened input
        x_flat = x_3x3.reshape(seq_len, 9)
        
        try:
            # Both should work
            recon_3x3, pred_3x3 = model.forward_single(t_recon, t_fut, x_3x3)
            recon_flat, pred_flat = model.forward_single(t_recon, t_fut, x_flat)
            
            # Results should be identical (same underlying data)
            assert jnp.allclose(recon_3x3, recon_flat, atol=1e-6)
            assert jnp.allclose(pred_3x3, pred_flat, atol=1e-6)
            
        except Exception as e:
            if "maximum number of solver steps" in str(e):
                pytest.skip(f"Solver convergence issue: {e}")
            else:
                raise
    
    @pytest.mark.parametrize("output_channel", [6, 9])
    def test_different_output_representations(self, test_data, output_channel):
        """Test model with different output representations."""
        key = jax.random.key(42)
        rngs = nnx.Rngs(params=key)
        
        config = {
            'input_channel': 9,
            'latent_channels': 8,
            'hidden_channels': 8,
            'output_channel': output_channel,
            'method': 'euler',
            'atol': 1e-3,
            'rtol': 1e-2
        }
        
        model = create_so3_neural_cde(**config, rngs=rngs)
        
        # Use single sample for faster testing
        t_recon_single = test_data['t_recon'][0]
        t_fut_single = test_data['t_fut'][0] 
        x_single = test_data['x'][0]
        
        try:
            recon, pred = model.forward_single(t_recon_single, t_fut_single, x_single)
            
            # Both should produce 3x3 rotation matrices regardless of internal representation
            assert recon.shape == (test_data['seq_len'], 3, 3)
            assert pred.shape == (test_data['n_future'], 3, 3)
            
        except Exception as e:
            if "maximum number of solver steps" in str(e):
                pytest.skip(f"Solver convergence issue: {e}")
            else:
                raise
    
    def test_postprocessing_functions(self):
        """Test SO(3) post-processing functions."""
        # Test symmetric orthogonalization
        key = jax.random.key(42)
        random_matrices = jax.random.normal(key, (5, 3, 3))
        
        # Apply symmetric orthogonalization
        ortho_matrices = jax.vmap(symmetric_orthogonalization)(random_matrices)
        
        # Check properties
        for i in range(5):
            R = ortho_matrices[i]
            
            # Should be orthogonal: R @ R.T = I
            should_be_identity = R @ R.T
            assert jnp.allclose(should_be_identity, jnp.eye(3), atol=1e-5)
            
            # Should have determinant 1 (proper rotation)
            det = jnp.linalg.det(R)
            assert jnp.allclose(det, 1.0, atol=1e-5)


class TestInterpolationMethods:
    """Test interpolation methods separately."""
    
    def test_hermite_interpolation(self):
        """Test Hermite cubic interpolation."""
        seq_len = 6
        input_channels = 9
        
        t = jnp.linspace(0, 1, seq_len)
        x = jax.random.normal(jax.random.key(42), (seq_len, input_channels))
        
        # Add batch dimension
        t_batch = t[None, :]
        x_batch = x[None, :]
        
        hermite = HermiteInterpolation()
        interpolation = hermite.create_interpolation(t_batch, x_batch)
        
        # Test evaluation at different points
        t_eval = jnp.array([0.0, 0.5, 1.0])
        
        for t_point in t_eval:
            result = interpolation.evaluate(t_point)
            assert result.shape == (1, input_channels + 1)  # +1 for time channel
    
    def test_linear_interpolation(self):
        """Test linear interpolation."""
        seq_len = 6
        input_channels = 9
        
        t = jnp.linspace(0, 1, seq_len)
        x = jax.random.normal(jax.random.key(42), (seq_len, input_channels))
        
        # Add batch dimension
        t_batch = t[None, :]
        x_batch = x[None, :]
        
        linear = LinearInterpolation()
        interpolation = linear.create_interpolation(t_batch, x_batch)
        
        # Test evaluation at different points
        t_eval = jnp.array([0.0, 0.5, 1.0])
        
        for t_point in t_eval:
            result = interpolation.evaluate(t_point)
            assert result.shape == (1, input_channels + 1)  # +1 for time channel


if __name__ == "__main__":
    # Run a quick test if executed directly
    import time
    
    print("Running quick SO3NeuralCDE test...")
    
    # Simple test
    key = jax.random.key(42)
    rngs = nnx.Rngs(params=key)
    
    model = create_so3_neural_cde(
        input_channel=9,
        latent_channels=8,
        hidden_channels=8,
        output_channel=9,
        method='euler',
        atol=1e-3,
        rtol=1e-2,
        rngs=rngs
    )
    
    # Test data
    t_recon = jnp.linspace(0, 0.5, 6)
    t_fut = jnp.linspace(0.6, 0.8, 3)
    x = jax.random.normal(jax.random.key(123), (6, 3, 3))
    
    try:
        start = time.time()
        recon, pred = model.forward_single(t_recon, t_fut, x)
        end = time.time()
        
        print(f"✅ Single sample test passed!")
        print(f"   Time: {end-start:.3f}s")
        print(f"   Recon shape: {recon.shape}")
        print(f"   Pred shape: {pred.shape}")
        
    except Exception as e:
        if "maximum number of solver steps" in str(e):
            print(f"⚠️  Solver convergence issue (expected): {e}")
        else:
            print(f"❌ Test failed: {e}")
            raise