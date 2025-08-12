"""
Simulation data utilities.
"""
import torch


def merge_scenarios(saved_scenarios, target_samples=None):
    """
    Merge multiple scenario datasets while preserving the 4-way MOI distribution structure.
    Subsamples each distribution to achieve target size.
    
    Args:
        saved_scenarios: List of scenario dictionaries to merge
        target_samples: Total number of samples in final merged dataset
    
    Returns:
        Merged dataset dictionary with preserved 4-way split structure
    """
    if not saved_scenarios:
        return None
    
    # Get target sample count
    if target_samples is None:
        if 'scenario_params' in saved_scenarios[0]:
            target_samples = saved_scenarios[0]['scenario_params']['SIM']['N_SAMPLES']
        else:
            target_samples = saved_scenarios[0]['quat'].shape[0]
    
    # Ensure target_samples is divisible by 4 for clean splits
    target_samples = (target_samples // 4) * 4
    samples_per_distribution = target_samples // 4
    
    print(f"Merging scenarios to create dataset with {target_samples} samples")
    print(f"Each distribution will have {samples_per_distribution} samples")
    
    # Initialize merged dataset with non-tensor values from first scenario
    first_scenario = saved_scenarios[0]
    merged = {}
    for key, value in first_scenario.items():
        if not isinstance(value, torch.Tensor):
            merged[key] = value
    
    # Update scenario description
    merged['scenario'] = 'merged'
    merged['scenario_description'] = 'Merged scenarios with 4 MOI distributions'
    
    # Get common keys across all scenarios
    common_tensor_keys = set()
    for key, value in first_scenario.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            common_tensor_keys.add(key)
    
    for scenario in saved_scenarios[1:]:
        scenario_keys = set()
        for key, value in scenario.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                scenario_keys.add(key)
        common_tensor_keys = common_tensor_keys.intersection(scenario_keys)
    
    print(f"Common tensor keys across all scenarios: {common_tensor_keys}")
    
    # Collect tensor data
    batch_dim_tensors = {}  # Tensors where batch is first dimension
    other_tensors = {}      # Tensors with other dimensions

    # First pass - identify tensor types and dimensions
    for key in common_tensor_keys:
        is_batch_dim_first = True
        batch_size = first_scenario['quat'].shape[0]  # Reference batch size
        
        for scenario in saved_scenarios:
            tensor = scenario[key]
            if tensor.dim() > 0 and tensor.shape[0] != batch_size:
                is_batch_dim_first = False
                break
        
        if is_batch_dim_first:
            batch_dim_tensors[key] = [[] for _ in range(4)]
        else:
            # Skip tensors where batch dimension isn't first for simplicity
            print(f"Skipping tensor {key} - batch dimension not first")
            continue
    
    # Process each scenario
    for scenario in saved_scenarios:
        scenario_samples = scenario['quat'].shape[0]
        samples_per_dist = scenario_samples // 4
        
        # Process each distribution segment
        for dist_idx in range(4):
            start_idx = dist_idx * samples_per_dist
            end_idx = (dist_idx + 1) * samples_per_dist if dist_idx < 3 else scenario_samples
            
            # Collect batch-dim-first tensor data
            for key in batch_dim_tensors:
                batch_dim_tensors[key][dist_idx].append(scenario[key][start_idx:end_idx])
    
    # Create consistent indices for each distribution
    distribution_indices = {}
    for dist_idx in range(4):
        # Get total samples for this distribution
        total_samples = 0
        for key in batch_dim_tensors:
            if batch_dim_tensors[key][dist_idx]:
                total_samples = batch_dim_tensors[key][dist_idx][0].shape[0]
                for scenario_tensor in batch_dim_tensors[key][dist_idx][1:]:
                    total_samples += scenario_tensor.shape[0]
                break
        
        if total_samples >= samples_per_distribution:
            # Generate consistent random indices for this distribution
            distribution_indices[dist_idx] = torch.randperm(total_samples)[:samples_per_distribution]
        else:
            # If not enough samples, we'll use all available and fill the rest later
            distribution_indices[dist_idx] = torch.arange(total_samples)
    
    # Process batch-dim-first tensors using consistent indices
    for key in batch_dim_tensors:
        final_dist_tensors = []
        
        for dist_idx in range(4):
            if not batch_dim_tensors[key][dist_idx]:
                print(f"Warning: No data for distribution {dist_idx}, tensor {key}")
                # Create empty tensor with correct shape
                if key in first_scenario:
                    shape = list(first_scenario[key].shape)
                    shape[0] = samples_per_distribution
                    empty_tensor = torch.zeros(shape, dtype=first_scenario[key].dtype)
                    final_dist_tensors.append(empty_tensor)
                continue
                
            try:
                # Concatenate all scenarios for this distribution
                dist_tensor = torch.cat(batch_dim_tensors[key][dist_idx], dim=0)
                total_samples = dist_tensor.shape[0]
                
                if total_samples >= samples_per_distribution:
                    # Use consistent indices for subsampling
                    subsampled = dist_tensor[distribution_indices[dist_idx]]
                    final_dist_tensors.append(subsampled)
                else:
                    print(f"Warning: Not enough samples for {key}, dist {dist_idx}. Need {samples_per_distribution}, have {total_samples}")
                    # Fill with zeros or repeat existing data
                    shape = list(dist_tensor.shape)
                    shape[0] = samples_per_distribution
                    empty_tensor = torch.zeros(shape, dtype=dist_tensor.dtype)
                    # Use all available indices
                    empty_tensor[:total_samples] = dist_tensor[distribution_indices[dist_idx]]
                    final_dist_tensors.append(empty_tensor)
            except Exception as e:
                print(f"Error processing {key}, dist {dist_idx}: {e}")
                # Skip this key if we can't process it
                break
        
        # Only concat and add to merged if we have data for all distributions
        if len(final_dist_tensors) == 4:
            merged[key] = torch.cat(final_dist_tensors, dim=0)
    
    # Add distribution indices for reference
    merged['distribution_indices'] = torch.arange(4).repeat_interleave(samples_per_distribution)
    
    # Verify data consistency
    for key in merged:
        if isinstance(merged[key], torch.Tensor) and merged[key].shape[0] == target_samples:
            print(f"Merged tensor {key}: shape {merged[key].shape}")
    
    print(f"Final merged dataset has {merged['quat'].shape[0]} samples")
    return merged
