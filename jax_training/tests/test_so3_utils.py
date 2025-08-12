"""
Test suite for SO(3) utility functions.

Tests the core mathematical functions for working with rotation matrices,
including orthogonalization and Gram-Schmidt processes.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Import the utilities
import sys
import os
# Add the jax_training directory to the path
jax_training_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, jax_training_dir)

from utils.so3 import (
    symmetric_orthogonalization,
    gramschmidt_to_rotmat,
    special_gramschmidt_single,
    special_gramschmidt
)


class TestSO3Utils:
    """Test class for SO(3) utility functions."""
    
    def test_symmetric_orthogonalization_single(self):
        """Test symmetric orthogonalization for a single matrix."""
        # Create a random 3x3 matrix
        key = jax.random.key(42)
        A = jax.random.normal(key, (3, 3))
        
        # Apply symmetric orthogonalization
        R = symmetric_orthogonalization(A)
        
        # Check it's a valid rotation matrix
        assert R.shape == (3, 3)
        
        # Check orthogonality: R @ R.T = I
        should_be_identity = R @ R.T
        assert jnp.allclose(should_be_identity, jnp.eye(3), atol=1e-6)
        
        # Check determinant = 1 (proper rotation)
        det = jnp.linalg.det(R)
        assert jnp.allclose(det, 1.0, atol=1e-6)
    
    def test_symmetric_orthogonalization_batch(self):
        """Test symmetric orthogonalization for batch of matrices."""
        batch_size = 5
        key = jax.random.key(123)
        A_batch = jax.random.normal(key, (batch_size, 3, 3))
        
        # Apply symmetric orthogonalization using vmap
        R_batch = jax.vmap(symmetric_orthogonalization)(A_batch)
        
        assert R_batch.shape == (batch_size, 3, 3)
        
        # Check all are valid rotation matrices
        for i in range(batch_size):
            R = R_batch[i]
            
            # Check orthogonality
            should_be_identity = R @ R.T
            assert jnp.allclose(should_be_identity, jnp.eye(3), atol=1e-6)
            
            # Check determinant = 1
            det = jnp.linalg.det(R)
            assert jnp.allclose(det, 1.0, atol=1e-6)
    
    def test_gramschmidt_6d_input_single(self):
        """Test Gram-Schmidt with 6D input for single sample."""
        # Create a 6D vector
        key = jax.random.key(42)
        inp_6d = jax.random.normal(key, (6,))
        
        # Apply Gram-Schmidt
        R = gramschmidt_to_rotmat(inp_6d)
        
        # Check output shape and properties
        assert R.shape == (3, 3)
        
        # Check orthogonality
        should_be_identity = R @ R.T
        assert jnp.allclose(should_be_identity, jnp.eye(3), atol=1e-6)
        
        # Check determinant = 1
        det = jnp.linalg.det(R)
        assert jnp.allclose(det, 1.0, atol=1e-6)
    
    def test_gramschmidt_6d_input_batch(self):
        """Test Gram-Schmidt with 6D input for batch of samples."""
        batch_size = 4
        key = jax.random.key(123)
        inp_6d_batch = jax.random.normal(key, (batch_size, 6))
        
        # Apply Gram-Schmidt using vmap
        R_batch = jax.vmap(gramschmidt_to_rotmat)(inp_6d_batch)
        
        assert R_batch.shape == (batch_size, 3, 3)
        
        # Check all are valid rotation matrices
        for i in range(batch_size):
            R = R_batch[i]
            
            # Check orthogonality
            should_be_identity = R @ R.T
            assert jnp.allclose(should_be_identity, jnp.eye(3), atol=1e-6)
            
            # Check determinant = 1
            det = jnp.linalg.det(R)
            assert jnp.allclose(det, 1.0, atol=1e-6)
    
    def test_gramschmidt_matrix_input_single(self):
        """Test Gram-Schmidt with 3x3 matrix input for single sample."""
        # Create a 3x3 matrix
        key = jax.random.key(42)
        inp_matrix = jax.random.normal(key, (3, 3))
        
        # Apply Gram-Schmidt (should use first 2 columns)
        R = gramschmidt_to_rotmat(inp_matrix)
        
        # Check output shape and properties
        assert R.shape == (3, 3)
        
        # Check orthogonality
        should_be_identity = R @ R.T
        assert jnp.allclose(should_be_identity, jnp.eye(3), atol=1e-6)
        
        # Check determinant = 1
        det = jnp.linalg.det(R)
        assert jnp.allclose(det, 1.0, atol=1e-6)
    
    def test_gramschmidt_matrix_input_batch(self):
        """Test Gram-Schmidt with 3x3 matrix input for batch of samples."""
        batch_size = 4
        key = jax.random.key(123)
        inp_matrix_batch = jax.random.normal(key, (batch_size, 3, 3))
        
        # Apply Gram-Schmidt using vmap
        R_batch = jax.vmap(gramschmidt_to_rotmat)(inp_matrix_batch)
        
        assert R_batch.shape == (batch_size, 3, 3)
        
        # Check all are valid rotation matrices
        for i in range(batch_size):
            R = R_batch[i]
            
            # Check orthogonality
            should_be_identity = R @ R.T
            assert jnp.allclose(should_be_identity, jnp.eye(3), atol=1e-6)
            
            # Check determinant = 1
            det = jnp.linalg.det(R)
            assert jnp.allclose(det, 1.0, atol=1e-6)
    
    def test_gramschmidt_temporal_batch(self):
        """Test Gram-Schmidt with temporal batch (seq_len, 6) - common in models."""
        seq_len = 10
        key = jax.random.key(42)
        inp_temporal = jax.random.normal(key, (seq_len, 6))
        
        # Apply Gram-Schmidt using vmap over time dimension
        R_temporal = jax.vmap(gramschmidt_to_rotmat)(inp_temporal)
        
        assert R_temporal.shape == (seq_len, 3, 3)
        
        # Check all are valid rotation matrices
        for t in range(seq_len):
            R = R_temporal[t]
            
            # Check orthogonality
            should_be_identity = R @ R.T
            assert jnp.allclose(should_be_identity, jnp.eye(3), atol=1e-6)
            
            # Check determinant = 1
            det = jnp.linalg.det(R)
            assert jnp.allclose(det, 1.0, atol=1e-6)
    
    def test_gramschmidt_batch_temporal(self):
        """Test Gram-Schmidt with batch and temporal dimensions (batch_size, seq_len, 6)."""
        batch_size = 3
        seq_len = 8
        key = jax.random.key(123)
        inp_batch_temporal = jax.random.normal(key, (batch_size, seq_len, 6))
        
        # Apply Gram-Schmidt using double vmap
        R_batch_temporal = jax.vmap(jax.vmap(gramschmidt_to_rotmat))(inp_batch_temporal)
        
        assert R_batch_temporal.shape == (batch_size, seq_len, 3, 3)
        
        # Check all are valid rotation matrices
        for i in range(batch_size):
            for t in range(seq_len):
                R = R_batch_temporal[i, t]
                
                # Check orthogonality
                should_be_identity = R @ R.T
                assert jnp.allclose(should_be_identity, jnp.eye(3), atol=1e-6)
                
                # Check determinant = 1
                det = jnp.linalg.det(R)
                assert jnp.allclose(det, 1.0, atol=1e-6)
    
    def test_special_gramschmidt_single_function(self):
        """Test the core special_gramschmidt_single function directly."""
        # Create a 3x2 matrix (input format for special_gramschmidt_single)
        key = jax.random.key(42)
        M = jax.random.normal(key, (3, 2))
        
        # Apply special Gram-Schmidt
        R = special_gramschmidt_single(M)
        
        # Check output shape and properties
        assert R.shape == (3, 3)
        
        # Check orthogonality
        should_be_identity = R @ R.T
        assert jnp.allclose(should_be_identity, jnp.eye(3), atol=1e-6)
        
        # Check determinant = 1
        det = jnp.linalg.det(R)
        assert jnp.allclose(det, 1.0, atol=1e-6)
    
    def test_special_gramschmidt_batch_function(self):
        """Test the special_gramschmidt batch function directly."""
        batch_size = 5
        key = jax.random.key(123)
        M_batch = jax.random.normal(key, (batch_size, 3, 2))
        
        # Apply special Gram-Schmidt batch function
        R_batch = special_gramschmidt(M_batch)
        
        assert R_batch.shape == (batch_size, 3, 3)
        
        # Check all are valid rotation matrices
        for i in range(batch_size):
            R = R_batch[i]
            
            # Check orthogonality
            should_be_identity = R @ R.T
            assert jnp.allclose(should_be_identity, jnp.eye(3), atol=1e-6)
            
            # Check determinant = 1
            det = jnp.linalg.det(R)
            assert jnp.allclose(det, 1.0, atol=1e-6)
    
    def test_gramschmidt_edge_cases(self):
        """Test Gram-Schmidt with edge cases."""
        # Test with nearly parallel vectors
        M_parallel = jnp.array([[1.0, 1.001], 
                               [0.0, 0.001], 
                               [0.0, 0.001]])
        
        R = special_gramschmidt_single(M_parallel)
        
        # Should still produce valid rotation matrix
        assert R.shape == (3, 3)
        should_be_identity = R @ R.T
        assert jnp.allclose(should_be_identity, jnp.eye(3), atol=1e-5)
        det = jnp.linalg.det(R)
        assert jnp.allclose(det, 1.0, atol=1e-5)
    
    def test_input_shape_consistency(self):
        """Test that different input shapes to gramschmidt_to_rotmat work correctly."""
        key = jax.random.key(42)
        
        # Test 6D input
        inp_6d = jax.random.normal(key, (6,))
        R_6d = gramschmidt_to_rotmat(inp_6d)
        
        # Test equivalent 3x2 matrix input  
        inp_3x2 = inp_6d.reshape(3, 2)
        inp_3x3 = jnp.concatenate([inp_3x2, jnp.zeros((3, 1))], axis=1)
        R_3x3 = gramschmidt_to_rotmat(inp_3x3)
        
        # Results should be identical
        assert jnp.allclose(R_6d, R_3x3, atol=1e-6)
    
    @pytest.mark.parametrize("shape", [(6,), (3, 3), (4, 6), (4, 3, 3), (2, 5, 6)])
    def test_gramschmidt_different_shapes(self, shape):
        """Test gramschmidt_to_rotmat with various input shapes."""
        key = jax.random.key(42)
        
        if shape[-1] == 6:
            # 6D representation
            inp = jax.random.normal(key, shape)
            expected_output_shape = shape[:-1] + (3, 3)
        elif shape[-2:] == (3, 3):
            # 3x3 matrix representation
            inp = jax.random.normal(key, shape)
            expected_output_shape = shape
        else:
            pytest.skip(f"Shape {shape} not supported by gramschmidt_to_rotmat")
        
        # gramschmidt_to_rotmat should handle batching internally
        # We don't need to vmap it manually
        R = gramschmidt_to_rotmat(inp)
        
        assert R.shape == expected_output_shape
        
        # For single matrices, check rotation matrix properties
        if len(expected_output_shape) == 2:
            should_be_identity = R @ R.T
            assert jnp.allclose(should_be_identity, jnp.eye(3), atol=1e-6)
            det = jnp.linalg.det(R)
            assert jnp.allclose(det, 1.0, atol=1e-6)


if __name__ == "__main__":
    # Run a quick test if executed directly
    print("Running quick SO(3) utilities test...")
    
    # Test basic functionality
    key = jax.random.key(42)
    
    # Test symmetric orthogonalization
    A = jax.random.normal(key, (3, 3))
    R1 = symmetric_orthogonalization(A)
    print(f"Symmetric orthogonalization: {R1.shape}")
    
    # Test 6D Gram-Schmidt
    inp_6d = jax.random.normal(key, (6,))
    R2 = gramschmidt_to_rotmat(inp_6d)
    print(f"6D Gram-Schmidt: {R2.shape}")
    
    # Test 3x3 matrix Gram-Schmidt
    inp_3x3 = jax.random.normal(key, (3, 3))
    R3 = gramschmidt_to_rotmat(inp_3x3)
    print(f"3x3 Gram-Schmidt: {R3.shape}")
    
    # Test batch processing
    inp_batch = jax.random.normal(key, (5, 6))
    R_batch = jax.vmap(gramschmidt_to_rotmat)(inp_batch)
    print(f"Batch Gram-Schmidt: {R_batch.shape}")
    
    print("All quick tests passed!")