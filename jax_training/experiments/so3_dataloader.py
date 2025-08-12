import torch
from torch.utils.data import Dataset
import roma
from torch.utils.data import DataLoader
from utils.quat_utils import add_noise_to_quaternion


class SO3Dataloader(Dataset):

    def __init__(self, data_path, dt=0.1, mode='train', rotational_noise_level=0.0, 
                 tf=2.0, in_rep='quat', out_rep='quat', n_prev=12,
                 n_future=12, split=0, temporal_noise_level=0.0):
        """
        Initializes the pt_based_dataset object.
        Args:
            data_path (str): The path to the data file.
            dt (float, optional): The desired time step of the data. Defaults to 0.1.
            mode (str, optional): The mode of the dataset ('train' or 'val'). Defaults to 'train'.
            noise_level (float, optional): The noise level for validation mode. Defaults to 0.0.
            tf (float, optional): The final time. Defaults to 2.0.
            in_rep (str, optional): The input representation ('quat', 'euler', 'rotvec', '6d', or '9d'). Defaults to 'quat'.
            out_rep (str, optional): The output representation ('quat', 'euler', 'rotvec', '6d', or '9d'). Defaults to 'quat'.
            n_prev (int, optional): The number of previous time steps. Defaults to 12.
            n_future (int, optional): The number of future time steps. Defaults to 12.
            split (int, optional): The split number (0-3). Defaults to 0.
        """
        
        super(SO3Dataloader, self).__init__()  # Call the __init__() method of the parent class
        
        data = torch.load(data_path, weights_only=False)  # Load the data from the specified data file
        
        data_dt = data['dt']  # Simulation timestep from physics generation (high-resolution)
        data_tf = data['t_f']  # Simulation time span from physics generation
        self.moi = data["moi"]
        self.dt = dt  # Training timestep for computational efficiency (typically larger)

        # Skip factor downsamples from simulation resolution to training resolution
        # Example: simulation dt=0.001, training dt=0.1 → skip=100 (keep every 100th sample)
        # This dramatically reduces memory usage and training time while preserving dynamics
        skip = int(dt/data_dt)  # Calculate the skip factor based on the time step and data time step

        self.mode = mode  # Set the mode of the dataset (train or val)
        self.rotational_noise_level = rotational_noise_level  # Set the rotational noise level
        self.temporal_noise_level = temporal_noise_level  # Set the temporal sampling noise level

        self.in_rep = in_rep  # Set the input representation
        self.out_rep = out_rep  # Set the output representation

        self.n_prev = n_prev  # Set the number of previous time steps
        self.n_future = n_future  # Set the number of future time steps

        self.data = data['quat'][:, ::skip]  # Apply the skip factor to the data
        self.omega = data['omega'][:, ::skip]  # Apply the skip factor to the omega data

        # Split the data
        self.split_data(split, mode)

        if mode in ["val", "test"] and rotational_noise_level > 0.0:
            # the whole run is seeded so this should be reproducible
            input_data = add_noise_to_quaternion(self.data, rotational_noise_level)
        else:
            input_data = self.data  # Use the original data for input
        recon_data = self.data.clone()
        # Stack the input data by selecting a range of previous time steps
        self.inputs = torch.stack([input_data[:,i:i+n_prev]
                                   for i in range(0, self.data.shape[1]-n_future-n_prev+1)] ,1)
        self.recon = torch.stack([recon_data[:,i:i+n_prev]
                                  for i in range(0, self.data.shape[1]-n_future-n_prev+1)] ,1)
        # Stack the target data by selecting a range of future time steps
        self.targets = torch.stack([self.data[:,i+n_prev:i+n_prev+n_future]
                                    for i in range(0, self.data.shape[1]-n_future-n_prev+1)],1)

        self.num_scenes = self.data.shape[0]  # Get the number of scenes in the data
        self.num_samples = self.inputs.shape[1]  # Get the number of samples in the inputs
        self.w = torch.stack([self.omega[:,i:i+n_prev] for i in range(0, self.data.shape[1]-n_future-n_prev+1)] ,1)

        # Define transforms for input and output representations
        transforms = {
            "6d": lambda x: roma.unitquat_to_rotmat(x)[..., :2],
            "9d": roma.unitquat_to_rotmat,
            "quat": lambda x: x,
            "euler": roma.unitquat_to_euler,
            "rotvec": roma.unitquat_to_rotvec,
        }
        self.in_transform = transforms[in_rep]
        self.out_transform = transforms[out_rep]
        # Apply the corresponding transform to the inputs and targets based on the input and output representations
        self.targets = self.out_transform(self.targets)
        self.recon = self.out_transform(self.recon)
        self.ts = torch.linspace(0,dt*(n_prev+n_future-1), n_prev+n_future)
        self.data_dt = data_dt  # Store data timestep for irregular sampling
        
        # Pre-compute irregular sampling if enabled (using high-res data)
        if temporal_noise_level > 0.0:
            data_highres = data['quat']  # Keep high-res data temporarily
            omega_highres = data['omega']  # Keep high-res data temporarily
            self.irregular_inputs, self.irregular_targets, self.irregular_recon, \
            self.irregular_omega, self.irregular_times = self._precompute_irregular_sampling(data_highres, omega_highres, data_dt)
            # High-res data is automatically discarded after this point
        else:
            self.irregular_inputs = self.irregular_targets = self.irregular_recon = None
            self.irregular_omega = self.irregular_times = None

    def split_data(self, split, mode):
        """
        Splits the data into train, validation, and test sets.
        Args:
            split (int): The split number (0-3).
            mode (str): The mode of the dataset ('train', 'val', or 'test').
        """
        assert split in [0, 1, 2, 3], "Split must be 0, 1, 2, or 3"
        assert mode in ['train', 'val', 'test'], "Mode must be 'train', 'val', or 'test'"
        
        num_scenes = self.data.shape[0]
        split_size = num_scenes // 4
        
        # Calculate indices for each of the 4 splits
        splits = []
        for i in range(4):
            start = i * split_size
            end = (i + 1) * split_size if i < 3 else num_scenes
            splits.append(list(range(start, end)))
        
        # Determine which splits to use based on the fold number and mode
        indices = []
        if mode == 'train':
            # Train on 2 splits
            train_split_indices = [(split) % 4, (split + 1) % 4]
            for idx in train_split_indices:
                indices.extend(splits[idx])
        elif mode == 'val':
            # Validate on 1 split
            val_split_index = (split + 2) % 4
            indices = splits[val_split_index]
        elif mode == 'test':
            # Test on 1 split
            test_split_index = (split + 3) % 4
            indices = splits[test_split_index]
        
        # Filter data based on selected indices
        self.data = self.data[indices]
        self.omega = self.omega[indices]
        self.moi = self.moi[indices]
        
        print(f"Splitting data for split {split} mode {mode}:", self.data.shape)

    def _precompute_irregular_sampling(self, data_highres, omega_highres, data_dt):
        """Pre-compute all irregular sampling patterns using high-resolution data and vectorized operations."""
        print("Pre-computing irregular sampling patterns with high-resolution data (vectorized)...")
        
        num_samples = self.data.shape[1] - self.n_future - self.n_prev + 1
        total_steps = self.n_prev + self.n_future
        
        # Generate all irregular time patterns at once (vectorized)
        regular_times_base = torch.linspace(0, self.dt * (total_steps - 1), total_steps)
        max_noise = self.temporal_noise_level * self.dt
        
        # Generate noise for all samples at once: (num_samples, total_steps)
        all_noise = (torch.rand(num_samples, total_steps) - 0.5) * 2 * max_noise
        all_irregular_times = regular_times_base.unsqueeze(0) + all_noise  # Broadcasting
        
        # Clamp all times
        all_irregular_times = torch.clamp(all_irregular_times, 0, self.dt * (total_steps - 1))
        
        # Sort timestamps and get sorting indices for each sample
        sorted_times, sort_indices = torch.sort(all_irregular_times, dim=1)
        all_irregular_times = sorted_times
        
        # Ensure strict monotonicity by adding small increments to duplicate timestamps
        # This is necessary when high noise levels cause clamping to create duplicates
        epsilon = 1e-6
        
        # Vectorized duplicate detection
        time_diffs = all_irregular_times[:, 1:] - all_irregular_times[:, :-1]  # (num_samples, total_steps-1)
        duplicate_mask = time_diffs <= epsilon  # (num_samples, total_steps-1)
        
        # Find samples that have duplicates
        samples_with_duplicates = torch.any(duplicate_mask, dim=1)  # (num_samples,)
        
        if torch.any(samples_with_duplicates):
            # Only process samples that have duplicates
            problem_indices = torch.where(samples_with_duplicates)[0]
            
            for sample_idx in problem_indices:
                # For each problematic sample, apply sequential fixes
                times = all_irregular_times[sample_idx].clone()
                
                # Simple sequential fix: ensure each timestamp is at least epsilon larger than previous
                for i in range(1, len(times)):
                    if times[i] <= times[i-1]:
                        times[i] = times[i-1] + epsilon
                
                all_irregular_times[sample_idx] = times
        
        # Convert to high-resolution indices (vectorized): (num_samples, total_steps)
        # Use data_dt for high-resolution indexing instead of self.dt
        highres_data_indices = torch.round(all_irregular_times / data_dt).long()
        # Base indices need to account for the downsampling factor
        skip = int(self.dt / data_dt)
        base_indices_highres = (torch.arange(num_samples) * skip).unsqueeze(1)  # (num_samples, 1)
        all_actual_indices = base_indices_highres + highres_data_indices  # Broadcasting: (num_samples, total_steps)
        all_actual_indices = torch.clamp(all_actual_indices, 0, data_highres.shape[1] - 1)
        
        # Extract data using advanced indexing from high-resolution data
        # Create index arrays for all scenes and samples
        scene_indices = torch.arange(data_highres.shape[0]).unsqueeze(1).unsqueeze(2)  # (num_scenes, 1, 1)
        sample_indices = all_actual_indices.unsqueeze(0)  # (1, num_samples, total_steps)
        
        # Extract all data at once from high-resolution data: (num_scenes, num_samples, total_steps, data_dim)
        all_irregular_data = data_highres[scene_indices, sample_indices]
        all_irregular_omega = omega_highres[scene_indices, sample_indices]
        
        # Sort the data according to the sorted timestamps using the sort_indices
        # Expand sort_indices to match data dimensions
        sort_indices_expanded = sort_indices.unsqueeze(0).unsqueeze(-1)  # (1, num_samples, total_steps, 1)
        sort_indices_expanded = sort_indices_expanded.expand(data_highres.shape[0], -1, -1, all_irregular_data.shape[-1])
        all_irregular_data = torch.gather(all_irregular_data, dim=2, index=sort_indices_expanded)
        
        # Sort omega data similarly
        sort_indices_omega = sort_indices.unsqueeze(0).unsqueeze(-1)  # (1, num_samples, total_steps, 1)
        sort_indices_omega = sort_indices_omega.expand(omega_highres.shape[0], -1, -1, all_irregular_omega.shape[-1])
        all_irregular_omega = torch.gather(all_irregular_omega, dim=2, index=sort_indices_omega)
        
        # Split into inputs/targets: (num_scenes, num_samples, n_prev/n_future, data_dim)
        irregular_inputs = all_irregular_data[:, :, :self.n_prev]
        irregular_targets = all_irregular_data[:, :, self.n_prev:]
        irregular_recon = irregular_inputs.clone()
        irregular_omega = all_irregular_omega[:, :, :self.n_prev]
        
        # Times for all samples: (num_samples, total_steps) -> (num_scenes, num_samples, total_steps)
        irregular_times = all_irregular_times.unsqueeze(0).expand(data_highres.shape[0], -1, -1)
        
        # Apply output transforms (handle dimension changes from quaternions to rotation matrices)
        original_target_shape = irregular_targets.shape  # (num_scenes, num_samples, n_future, 4)
        original_recon_shape = irregular_recon.shape     # (num_scenes, num_samples, n_prev, 4)
        
        # Reshape for transform: (batch_size, 4) for quaternions
        targets_flat = irregular_targets.reshape(-1, original_target_shape[-1])
        recon_flat = irregular_recon.reshape(-1, original_recon_shape[-1])
        
        # Apply transforms (quaternions → rotation representation)
        targets_transformed = self.out_transform(targets_flat)
        recon_transformed = self.out_transform(recon_flat)
        
        # Determine output dimensions based on representation
        if self.out_rep == "6d":
            out_dims = (3, 2)  # 6D continuous representation  
        elif self.out_rep == "9d":
            out_dims = (3, 3)  # Full rotation matrices
        elif self.out_rep == "quat":
            out_dims = (4,)    # Quaternions
        else:
            # For other representations, infer from transformed shape
            sample_shape = targets_transformed.shape[1:]
            out_dims = sample_shape
        
        # Reshape back to proper dimensions with correct output shape
        if len(out_dims) == 2:
            irregular_targets = targets_transformed.reshape(original_target_shape[0], original_target_shape[1], 
                                                           original_target_shape[2], out_dims[0], out_dims[1])
            irregular_recon = recon_transformed.reshape(original_recon_shape[0], original_recon_shape[1],
                                                       original_recon_shape[2], out_dims[0], out_dims[1])
        else:
            irregular_targets = targets_transformed.reshape(original_target_shape[0], original_target_shape[1], 
                                                           original_target_shape[2], -1)
            irregular_recon = recon_transformed.reshape(original_recon_shape[0], original_recon_shape[1],
                                                       original_recon_shape[2], -1)
        
        print(f"Pre-computed irregular sampling: {irregular_inputs.shape}")
        return irregular_inputs, irregular_targets, irregular_recon, irregular_omega, irregular_times

    def generate_irregular_indices(self, base_sample_idx):
        """
        Generate irregular time indices for a given sample.
        
        Args:
            base_sample_idx: Base sample index for regular sampling
            
        Returns:
            irregular_indices: List of actual data indices to use
            actual_times: Actual time values corresponding to these indices
        """
        if self.temporal_noise_level == 0.0:
            # Regular sampling - return original indices and times (avoid expensive operations)
            total_steps = self.n_prev + self.n_future
            indices = [base_sample_idx + i for i in range(total_steps)]  # Faster than range()
            return indices, self.ts  # No need to clone, it's read-only
        
        # Generate irregular sampling
        total_steps = self.n_prev + self.n_future
        
        # Start with regular time grid
        regular_times = torch.linspace(0, self.dt * (total_steps - 1), total_steps)
        
        # Add noise to each time point
        max_noise = self.temporal_noise_level * self.dt
        noise = (torch.rand(total_steps) - 0.5) * 2 * max_noise  # Uniform in [-max_noise, max_noise]
        irregular_times = regular_times + noise
        
        # Clamp to valid time range
        irregular_times = torch.clamp(irregular_times, 0, self.dt * (total_steps - 1))
        
        # Sort timestamps and get sorting indices
        sorted_times, sort_indices = torch.sort(irregular_times)
        irregular_times = sorted_times
        
        # Convert to data indices and round to grid (vectorized)
        data_indices = torch.round(irregular_times / self.dt).long()
        actual_indices = base_sample_idx + data_indices
        actual_indices = torch.clamp(actual_indices, 0, self.data.shape[1] - 1)
        
        return actual_indices.tolist(), irregular_times

    def __len__(self):
        return (self.data.shape[1]-self.n_future-self.n_prev+1) * self.data.shape[0] 
    
    def __getitem__(self, index):
        scene_idx = index // self.num_samples
        sample_idx = index % self.num_samples

        if self.temporal_noise_level > 0.0:
            # Use pre-computed irregular sampling (much faster!)
            input_data = self.irregular_inputs[scene_idx, sample_idx]
            target_data = self.irregular_targets[scene_idx, sample_idx]  # Already transformed
            recon_data = self.irregular_recon[scene_idx, sample_idx]     # Already transformed
            omega_data = self.irregular_omega[scene_idx, sample_idx]
            
            # Split pre-computed times
            all_times = self.irregular_times[scene_idx, sample_idx]
            
            # The times are already sorted, but we need to ensure that when split,
            # the combined sequence [t_recon, t_fut] is monotonically increasing
            # for the Neural CDE. Since the split is at n_prev, we need to ensure
            # the boundary condition is respected.
            
            # Split at the n_prev boundary
            t_recon = all_times[:self.n_prev]
            t_fut = all_times[self.n_prev:]
            
            # Ensure strict monotonicity across the boundary
            # If t_fut[0] <= t_recon[-1], shift t_fut to maintain monotonicity
            if len(t_fut) > 0 and len(t_recon) > 0 and t_fut[0] <= t_recon[-1]:
                # Calculate the minimum shift needed
                epsilon = 1e-6
                shift = t_recon[-1] - t_fut[0] + epsilon
                t_fut = t_fut + shift
            
            # Apply rotational noise to input quaternions if needed
            if self.mode == "train" and self.rotational_noise_level > 0.0:
                input_data = add_noise_to_quaternion(input_data, self.rotational_noise_level)
            
            # Apply input transform only (targets already transformed during pre-computation)
            input_transformed = self.in_transform(input_data)
            target_transformed = target_data
            recon_transformed = recon_data
        else:
            # Use regular sampling (original behavior - data already transformed)
            input_data = self.inputs[scene_idx, sample_idx]
            target_data = self.targets[scene_idx, sample_idx]
            recon_data = self.recon[scene_idx, sample_idx]
            omega_data = self.w[scene_idx, sample_idx]
            
            # Use regular times
            t_recon = self.ts[:self.n_prev]
            t_fut = self.ts[self.n_prev:]

            # Apply rotational noise to input quaternions if needed
            if self.mode == "train" and self.rotational_noise_level > 0.0:
                input_data = add_noise_to_quaternion(input_data, self.rotational_noise_level)
            
            # For regular sampling, apply transform only to input (targets already transformed)
            input_transformed = self.in_transform(input_data)
            target_transformed = target_data
            recon_transformed = recon_data
        
        return (
            (10*t_recon, 10*t_fut, input_transformed),
            target_transformed,
            recon_transformed,
            omega_data,
            self.moi[scene_idx]
        )

    def __del__(self):
        """Automatically clean up arrays when dataset is deleted to prevent memory leaks."""
        try:
            from utils.data import cleanup_dataset_arrays
            cleanup_dataset_arrays(self)
        except Exception:
            # Ignore errors during cleanup to avoid issues in destructor
            pass


if __name__ == "__main__":

    import time

    # Test all 4 splits for both train and val modes
    for split in range(4):
        for mode in ['train', 'val']:
            print(f"Testing split {split}, mode {mode}")
            data = SO3Dataloader(data_path='../data/damped_magnetic/rigid_body_DAMPED.pt',
                                    dt=0.1, mode=mode, noise_level=0.0, tf=2.0, in_rep='6d', out_rep='quat',
                                    n_prev=12, n_future=12, split=split)

            # Create a DataLoader
            dataloader = DataLoader(data, batch_size=1, shuffle=True)

            # Iterate over the dataset
            num_iters = 0
            start_time = time.time()
            for inputs, targets, recon in dataloader:
                num_iters += 1

            # Compute the average time per iteration
            average_time = (time.time() - start_time) / num_iters
            print(f"Number of samples: {len(data)}")
            print(f"Average time per iteration: {average_time*1000:.6f} ms")
            print()
