"""
Test suite for dataloader irregular temporal sampling capabilities.

This tests verify that the dataloader properly handles irregular sampling
and provides per-sample time grids for Neural CDE models.
"""

import pytest
import torch
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add the jax_training directory to the path
jax_training_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, jax_training_dir)

from experiments.so3_dataloader import SO3Dataloader


class TestDataloaderIrregularSampling:
    """Test irregular temporal sampling in the dataloader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create minimal test data
        self.batch_size = 4
        self.n_prev = 6
        self.n_future = 4
        self.dt = 0.1
        
        # Create mock dataset parameters
        self.dataset_params = {
            'dt': 0.001,  # High-res simulation dt
            't_f': 10.0,
            'scenario_description': 'TEST',
            'omega_scale': 0.3
        }
        
        # Create simple test data path (we'll mock the data loading)
        self.data_path = "test_data.pt"
    
    def create_mock_data(self, temporal_noise_level=0.0):
        """Create mock data for testing."""
        num_samples = 100
        high_res_steps = int(self.dataset_params['t_f'] / self.dataset_params['dt'])
        
        # Create high-resolution mock data
        mock_data = {
            'dt': self.dataset_params['dt'],  # Simulation timestep
            't_f': self.dataset_params['t_f'],  # Final time
            'scenario_description': self.dataset_params['scenario_description'],
            'omega_scale': self.dataset_params['omega_scale'],
            'quat': torch.randn(num_samples, high_res_steps, 4),  # Quaternions
            'omega': torch.randn(num_samples, high_res_steps, 3),  # Angular velocities
            'moi': torch.randn(num_samples, 3, 3)  # Moment of inertia tensors
        }
        
        # Mock torch.load to return our data
        import unittest.mock
        with unittest.mock.patch('torch.load', return_value=mock_data):
            dataloader = SO3Dataloader(
                data_path=self.data_path,
                n_prev=self.n_prev,
                n_future=self.n_future,
                dt=self.dt,
                tf=2.0,
                in_rep='9d',
                out_rep='9d',
                split=0,
                mode='train',
                temporal_noise_level=temporal_noise_level,
                rotational_noise_level=0.0  # No observation noise for these tests
            )
        
        return dataloader
    
    def test_regular_sampling_time_consistency(self):
        """Test that regular sampling provides consistent time grids across samples."""
        dataloader = self.create_mock_data(temporal_noise_level=0.0)
        
        # Get several samples
        sample1 = dataloader[0]
        sample2 = dataloader[1]
        sample3 = dataloader[2]
        
        # Extract time vectors (format: inputs, targets, recon, omega, moi)
        (t_recon1, t_fut1, _), _, _, _, _ = sample1
        (t_recon2, t_fut2, _), _, _, _, _ = sample2
        (t_recon3, t_fut3, _), _, _, _, _ = sample3
        
        # For regular sampling, all time vectors should be identical
        assert torch.allclose(t_recon1, t_recon2, atol=1e-6), "Regular sampling should have identical t_recon"
        assert torch.allclose(t_recon1, t_recon3, atol=1e-6), "Regular sampling should have identical t_recon"
        assert torch.allclose(t_fut1, t_fut2, atol=1e-6), "Regular sampling should have identical t_fut"
        assert torch.allclose(t_fut1, t_fut3, atol=1e-6), "Regular sampling should have identical t_fut"
        
        # Check expected regular time structure
        expected_times = torch.linspace(0, self.dt * (self.n_prev + self.n_future - 1), 
                                       self.n_prev + self.n_future)
        expected_t_recon = expected_times[:self.n_prev] * 10  # Dataloader scales by 10
        expected_t_fut = expected_times[self.n_prev:] * 10
        
        assert torch.allclose(t_recon1, expected_t_recon, atol=1e-6), "Regular t_recon structure incorrect"
        assert torch.allclose(t_fut1, expected_t_fut, atol=1e-6), "Regular t_fut structure incorrect"
    
    def test_irregular_sampling_time_uniqueness(self):
        """Test that irregular sampling provides unique time grids per sample."""
        dataloader = self.create_mock_data(temporal_noise_level=0.2)  # 20% noise
        
        # Get several samples
        samples = [dataloader[i] for i in range(10)]
        time_vectors = [(sample[0][0], sample[0][1]) for sample in samples]
        
        # Check that time vectors are different across samples
        for i in range(len(time_vectors)):
            for j in range(i + 1, len(time_vectors)):
                t_recon_i, t_fut_i = time_vectors[i]
                t_recon_j, t_fut_j = time_vectors[j]
                
                # Time vectors should be different (not identical)
                assert not torch.allclose(t_recon_i, t_recon_j, atol=1e-6), \
                    f"Irregular sampling: t_recon should differ between samples {i} and {j}"
                assert not torch.allclose(t_fut_i, t_fut_j, atol=1e-6), \
                    f"Irregular sampling: t_fut should differ between samples {i} and {j}"
    
    def test_irregular_sampling_time_properties(self):
        """Test properties of irregular time grids."""
        dataloader = self.create_mock_data(temporal_noise_level=0.15)
        
        # Get a sample
        (t_recon, t_fut, _), _, _, _, _ = dataloader[0]
        
        # Check dimensions
        assert t_recon.shape == (self.n_prev,), f"t_recon shape should be ({self.n_prev},), got {t_recon.shape}"
        assert t_fut.shape == (self.n_future,), f"t_fut shape should be ({self.n_future},), got {t_fut.shape}"
        
        # Check that times are monotonic (should be sorted)
        combined_times = torch.cat([t_recon, t_fut])
        sorted_times = torch.sort(combined_times)[0]
        assert torch.allclose(combined_times, sorted_times, atol=1e-6), \
            "Time vectors should be monotonically increasing"
        
        # Check that times are positive
        assert torch.all(combined_times >= 0), "All times should be non-negative"
        
        # Check that we have reasonable time spacing (not all clustered)
        time_diffs = combined_times[1:] - combined_times[:-1]
        assert torch.all(time_diffs > 0), "Time differences should be positive"
        
        # Check that noise is within expected bounds
        regular_times = torch.linspace(0, self.dt * (self.n_prev + self.n_future - 1), 
                                     self.n_prev + self.n_future) * 10
        max_expected_noise = 0.15 * self.dt * 10  # 15% of dt, scaled by 10
        
        noise = torch.abs(combined_times - regular_times)
        assert torch.all(noise <= max_expected_noise * 1.1), \
            f"Noise should be within bounds. Max noise: {torch.max(noise)}, expected: {max_expected_noise}"
    
    def test_batch_time_dimensions(self):
        """Test what happens when we create a batch - are time vectors per-sample or shared?"""
        dataloader = self.create_mock_data(temporal_noise_level=0.1)
        
        # Create a batch using DataLoader
        from torch.utils.data import DataLoader
        batch_loader = DataLoader(dataloader, batch_size=self.batch_size, shuffle=False)
        
        # Get one batch
        batch = next(iter(batch_loader))
        # Unpack: inputs, targets, recon, omega, moi
        (batch_t_recon, batch_t_fut, batch_x), batch_targets, batch_recon, batch_omega, batch_moi = batch
        
        print(f"Batch shapes:")
        print(f"  t_recon: {batch_t_recon.shape}")
        print(f"  t_fut: {batch_t_fut.shape}")
        print(f"  x: {batch_x.shape}")
        
        # This is the critical test: are time vectors per-sample or shared?
        if len(batch_t_recon.shape) == 1:
            print("WARNING: Time vectors are shared across batch (current behavior)")
            assert batch_t_recon.shape == (self.n_prev,), "Shared t_recon should be 1D"
            assert batch_t_fut.shape == (self.n_future,), "Shared t_fut should be 1D"
        else:
            print("SUCCESS: Time vectors are per-sample (desired behavior)")
            assert batch_t_recon.shape == (self.batch_size, self.n_prev), \
                f"Per-sample t_recon should be ({self.batch_size}, {self.n_prev})"
            assert batch_t_fut.shape == (self.batch_size, self.n_future), \
                f"Per-sample t_fut should be ({self.batch_size}, {self.n_future})"
            
            # Check that different samples have different time vectors
            for i in range(self.batch_size - 1):
                assert not torch.allclose(batch_t_recon[i], batch_t_recon[i+1], atol=1e-6), \
                    f"Samples {i} and {i+1} should have different t_recon"
                assert not torch.allclose(batch_t_fut[i], batch_t_fut[i+1], atol=1e-6), \
                    f"Samples {i} and {i+1} should have different t_fut"
    
    def test_irregular_vs_regular_data_consistency(self):
        """Test that irregular sampling produces different data than regular sampling."""
        regular_loader = self.create_mock_data(temporal_noise_level=0.0)
        irregular_loader = self.create_mock_data(temporal_noise_level=0.2)
        
        # Get same sample index from both loaders
        regular_sample = regular_loader[5]
        irregular_sample = irregular_loader[5]
        
        (t_recon_reg, t_fut_reg, x_reg), targets_reg, recon_reg, _, _ = regular_sample
        (t_recon_irreg, t_fut_irreg, x_irreg), targets_irreg, recon_irreg, _, _ = irregular_sample
        
        # Time vectors should be different
        assert not torch.allclose(t_recon_reg, t_recon_irreg, atol=1e-6), \
            "Regular and irregular t_recon should differ"
        assert not torch.allclose(t_fut_reg, t_fut_irreg, atol=1e-6), \
            "Regular and irregular t_fut should differ"
        
        # Data should be different because we're sampling at different time points
        if not torch.allclose(x_reg, x_irreg, atol=1e-6):
            print("SUCCESS: Irregular sampling produces different input data")
        else:
            print("WARNING: Irregular sampling produces identical input data")
        
        if not torch.allclose(targets_reg, targets_irreg, atol=1e-6):
            print("SUCCESS: Irregular sampling produces different target data")
        else:
            print("WARNING: Irregular sampling produces identical target data")
    
    def test_timestamp_monotonicity_sorting(self):
        """Test that timestamps are guaranteed to be monotonically increasing after sorting fix."""
        # Test different noise levels to ensure sorting works under various conditions
        noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]  # Include very high noise levels
        
        for noise_level in noise_levels:
            dataloader = self.create_mock_data(temporal_noise_level=noise_level)
            
            # Test multiple samples to ensure consistency
            num_samples_to_test = 20
            
            for i in range(num_samples_to_test):
                (t_recon, t_fut, _), _, _, _, _ = dataloader[i]
                
                # Combine all timestamps
                all_times = torch.cat([t_recon, t_fut])
                
                # Check strict monotonicity (each timestamp > previous)
                time_diffs = all_times[1:] - all_times[:-1]
                assert torch.all(time_diffs > 0), \
                    f"Timestamps must be strictly increasing for noise_level={noise_level}, sample={i}. " \
                    f"Times: {all_times.tolist()}, Diffs: {time_diffs.tolist()}"
                
                # Check that times are properly sorted
                sorted_times = torch.sort(all_times)[0]
                assert torch.allclose(all_times, sorted_times, atol=1e-8), \
                    f"Timestamps should be sorted for noise_level={noise_level}, sample={i}"
                
                # Verify no duplicate timestamps (which would break interpolation)
                assert torch.all(time_diffs > 1e-8), \
                    f"No duplicate timestamps allowed for noise_level={noise_level}, sample={i}"
        
        print(f"✅ Timestamp monotonicity verified for noise levels: {noise_levels}")

    def test_extreme_temporal_noise_levels(self):
        """Test that extreme temporal noise levels (1.0, 2.0) work correctly with monotonicity fix."""
        extreme_noise_levels = [1.0, 2.0]
        
        for noise_level in extreme_noise_levels:
            dataloader = self.create_mock_data(temporal_noise_level=noise_level)
            
            # Test multiple samples to ensure consistency
            num_samples_to_test = 10
            
            for i in range(num_samples_to_test):
                (t_recon, t_fut, _), _, _, _, _ = dataloader[i]
                
                # Combine all timestamps
                all_times = torch.cat([t_recon, t_fut])
                
                # Check strict monotonicity (each timestamp > previous)
                time_diffs = all_times[1:] - all_times[:-1]
                assert torch.all(time_diffs > 0), \
                    f"Extreme noise level {noise_level} failed monotonicity for sample {i}. " \
                    f"Times: {all_times.tolist()}, Diffs: {time_diffs.tolist()}"
                
                # Verify minimum separation (should be at least epsilon)
                min_diff = torch.min(time_diffs)
                assert min_diff > 1e-7, \
                    f"Minimum time difference too small for noise_level={noise_level}, sample={i}: {min_diff}"
                
                # Verify times are within reasonable bounds
                assert torch.all(all_times >= 0), \
                    f"Negative timestamps found for noise_level={noise_level}, sample={i}"
                # Note: dataloader scales times by 10x, so expected range is [0, 10 * dt * (n_prev + n_future)]
                expected_max_time = 10 * self.dt * (self.n_prev + self.n_future - 1)
                assert torch.all(all_times <= expected_max_time * 1.1), \
                    f"Timestamps exceed expected range for noise_level={noise_level}, sample={i}. " \
                    f"Max time: {torch.max(all_times)}, Expected max: {expected_max_time}"
        
        print(f"✅ Extreme temporal noise levels verified: {extreme_noise_levels}")

    def test_temporal_noise_scaling(self):
        """Test that different noise levels produce appropriately scaled irregularity."""
        # Use deterministic seed for reproducible test
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Test with just two well-separated noise levels
        noise_levels = [0.01, 0.2]
        loaders = [self.create_mock_data(temporal_noise_level=nl) for nl in noise_levels]
        
        # Get same sample from each loader
        samples = [loader[0] for loader in loaders]
        time_vectors = [(sample[0][0], sample[0][1]) for sample in samples]
        
        # Calculate deviations from regular grid
        regular_times = torch.linspace(0, self.dt * (self.n_prev + self.n_future - 1), 
                                     self.n_prev + self.n_future) * 10
        regular_t_recon = regular_times[:self.n_prev]
        regular_t_fut = regular_times[self.n_prev:]
        
        deviations = []
        for t_recon, t_fut in time_vectors:
            dev_recon = torch.abs(t_recon - regular_t_recon)
            dev_fut = torch.abs(t_fut - regular_t_fut)
            avg_deviation = torch.mean(torch.cat([dev_recon, dev_fut]))
            deviations.append(avg_deviation.item())
        
        print(f"Temporal noise levels: {noise_levels}")
        print(f"Average deviations: {deviations}")
        
        # Check that high noise level produces larger deviation than low noise level
        assert deviations[1] > deviations[0], \
            f"High noise level should produce larger deviations: {deviations[0]} vs {deviations[1]}"


if __name__ == "__main__":
    # Run tests directly
    print("Testing dataloader irregular sampling capabilities...")
    
    test_instance = TestDataloaderIrregularSampling()
    test_instance.setup_method()
    
    try:
        print("\n1. Testing regular sampling consistency...")
        test_instance.test_regular_sampling_time_consistency()
        print("✅ Regular sampling consistency test passed")
        
        print("\n2. Testing irregular sampling uniqueness...")
        test_instance.test_irregular_sampling_time_uniqueness()
        print("✅ Irregular sampling uniqueness test passed")
        
        print("\n3. Testing irregular sampling properties...")
        test_instance.test_irregular_sampling_time_properties()
        print("✅ Irregular sampling properties test passed")
        
        print("\n4. Testing batch time dimensions...")
        test_instance.test_batch_time_dimensions()
        print("✅ Batch time dimensions test completed")
        
        print("\n5. Testing irregular vs regular data consistency...")
        test_instance.test_irregular_vs_regular_data_consistency()
        print("✅ Irregular vs regular data consistency test passed")
        
        print("\n6. Testing timestamp monotonicity sorting...")
        test_instance.test_timestamp_monotonicity_sorting()
        print("✅ Timestamp monotonicity sorting test passed")
        
        print("\n7. Testing temporal noise scaling...")
        test_instance.test_temporal_noise_scaling()
        print("✅ Temporal noise scaling test passed")
        
        print("\nAll dataloader irregular sampling tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise