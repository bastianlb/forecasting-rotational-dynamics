import torch

def load_dataset_parameters(data_path):
    """
    Loads parameters from the dataset file for reference/logging purposes only.
    
    These are the original simulation parameters used to generate the physics data.
    They are NOT used for training configuration - training uses separate, optimized
    temporal parameters for memory efficiency.
    
    Args:
        data_path: Path to the dataset file
        
    Returns:
        Dictionary of extracted simulation parameters (for logging only)
    """
    # Load dataset
    data = torch.load(data_path, weights_only=False)
    
    # Extract parameters
    params = {}
    
    # Basic dataset info
    params['dt'] = float(data.get('dt', 0.1))
    params['t_f'] = float(data.get('t_f', 2.0))
    params['moi'] = data.get('moi', None)  # Moments of inertia
    
    # Additional information if available
    if 'scenario_params' in data:
        params['scenario_params'] = data['scenario_params']
    
    if 'scenario' in data:
        params['scenario_description'] = data['scenario']
    
    if 'omega_scale' in data:
        params['omega_scale'] = data['omega_scale']
        
    # Calculate model dimensions based on data
    if 'quat' in data:
        # Get shapes from the data
        params['data_shape'] = data['quat'].shape
        
    if 'omega' in data:
        params['omega_shape'] = data['omega'].shape
    
    # Return the extracted parameters
    return params

def update_cfg_from_dataset(cfg, data_path):
    """
    Extract parameters from the dataset for logging/reference purposes only.
    
    IMPORTANT: This function does NOT update the training configuration with dataset values.
    
    The dataset contains high-resolution simulation parameters (e.g., dt=0.001, tf=10.0) 
    that were used for physics simulation accuracy. However, training uses different 
    temporal parameters (dt=0.1, tf=2.0) for:
    - Memory efficiency (100x fewer training samples)
    - Computational efficiency during training
    - Standard training efficiency practices
    
    The skip factor (skip = training_dt / simulation_dt) automatically handles
    the downsampling from simulation resolution to training resolution.
    
    Args:
        cfg: YACS configuration node (unchanged)
        data_path: Path to the dataset file
        
    Returns:
        Tuple of (unchanged cfg, extracted dataset parameters for logging)
    """
    # Load parameters from dataset for logging/reference only
    params = load_dataset_parameters(data_path)
    
    # Log extracted parameters for reference (but don't use them for training config)
    print(f"Dataset parameters extracted from {data_path} (for reference only):")
    for k, v in params.items():
        if k not in ['data_shape', 'omega_shape', 'moi']:
            print(f"  {k}: {v}")
    
    print(f"Training will use config values: dt={cfg.DATA.DT}, tf={cfg.DATA.TF}")
    print(f"Skip factor will be: {int(cfg.DATA.DT / params.get('dt', cfg.DATA.DT))}")
    
    return cfg, params


