#!/usr/bin/env python
"""
Configuration management for JAX/Flax training pipeline.
"""
import argparse
import os
from pathlib import Path
from yacs.config import CfgNode as CN


def get_cfg_defaults():
    """Get default training configuration"""
    _C = CN()
    
    # Data configuration
    _C.DATA = CN()
    _C.DATA.PATH = "./data/rigid_body_FREE_ROTATION.pt"
    _C.DATA.DATASET_NAME = ""  # Automatically extracted from PATH
    _C.DATA.N_PREV = 12  # Number of previous timesteps for reconstruction
    _C.DATA.N_FUTURE = 8  # Number of future timesteps to predict
    _C.DATA.BATCH_SIZE = 500  # Batch size for training
    _C.DATA.VAL_BATCH_SIZE = 5000  # Batch size for validation (10x training)
    _C.DATA.NUM_WORKERS = 4
    # Training temporal parameters (NOT simulation parameters)
    # These control sparsity of training measurements, not physics simulation temporal resolution 
    # The dataset may contain higher-resolution simulation data (e.g., dt=0.001, tf=10.0)
    # but training downsamples via skip factor for computational efficiency
    _C.DATA.DT = 0.1  # Training timestep for computational efficiency
    _C.DATA.TF = 2.0  # Training time horizon
    _C.DATA.IN_REP = '9d'  # Input representation (9D rotation matrices)
    _C.DATA.OUT_REP = '6d'  # Output representation (9D rotation matrices)
    _C.DATA.SPLIT = 0  # Split index (0-3) for 4-way MOI distribution splits
    _C.DATA.TRAIN_SPLIT = 0.8
    _C.DATA.VAL_SPLIT = 0.1
    _C.DATA.TEST_SPLIT = 0.1
    
    # Noise configuration
    _C.DATA.TRAIN_NOISE_LEVELS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]  # Multiple noise levels for training (concatenated datasets)
    _C.DATA.VAL_NOISE_LEVELS = [0.02, 0.05]  # Multiple noise levels for validation (concatenated datasets)
    _C.DATA.TEST_NOISE_LEVELS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]  # Noise levels for testing
    _C.DATA.TEST_HORIZONS = [1, 4, 6, 8, 10, 12]  # Prediction horizons for testing
    
    # Temporal sampling configuration
    _C.DATA.TEMPORAL_NOISE_LEVEL = 0.3  # Max temporal noise as fraction of dt (0.0 = regular sampling, >0 = irregular)
    _C.DATA.TRAIN_TEMPORAL_NOISE_LEVEL = 0.3  # Training temporal noise (overrides TEMPORAL_NOISE_LEVEL if set)
    _C.DATA.TEST_TEMPORAL_NOISE_LEVELS = []  # Test-time temporal noise levels for evaluation
    
    # Model configuration
    _C.MODEL = CN()
    _C.MODEL.TYPE = "GRU"
    _C.MODEL.INPUT_CHANNEL = 9  # 9D rotation matrices
    _C.MODEL.LATENT_DIM = 30        # GRU hidden size / Neural CDE latent state dimension
    _C.MODEL.MLP_WIDTH = 50         # Neural CDE vector field MLP bottleneck width
    _C.MODEL.OUTPUT_CHANNEL = 6  # 6D output (used with GSO to recover 9D)
    _C.MODEL.NUM_LAYERS = 3  # Number of GRU layers
    _C.MODEL.DROPOUT = 0.1
    _C.MODEL.USE_RECONSTRUCTION = True
    _C.MODEL.USE_TIME_CONDITIONING = False  # Enable time conditioning in GRU (adds +1 input dimension)
    
    # SO3NeuralCDE-specific parameters
    _C.MODEL.INTERPOLATION_METHOD = "hermite"  # Options: "hermite", "linear", "savitzky_golay"/"sg"
    _C.MODEL.ODE_METHOD = "dopri5"  # Options: "tsit5", "dopri5", "euler"
    _C.MODEL.ATOL = 1e-4  # Absolute tolerance for ODE solver
    _C.MODEL.RTOL = 1e-4  # Relative tolerance for ODE solver
    _C.MODEL.MAX_STEPS = 4096  # Maximum number of solver steps
    
    # Tolerance scheduling (adaptive precision during training)
    _C.MODEL.TOLERANCE_SCHEDULING = CN()
    _C.MODEL.TOLERANCE_SCHEDULING.ENABLE = False  # Enable tolerance scheduling
    _C.MODEL.TOLERANCE_SCHEDULING.INITIAL_ATOL = 1e-3  # Start with relaxed tolerances
    _C.MODEL.TOLERANCE_SCHEDULING.INITIAL_RTOL = 1e-3  # Start with relaxed tolerances
    _C.MODEL.TOLERANCE_SCHEDULING.FINAL_ATOL = 1e-5    # End with strict tolerances
    _C.MODEL.TOLERANCE_SCHEDULING.FINAL_RTOL = 1e-5    # End with strict tolerances
    _C.MODEL.TOLERANCE_SCHEDULING.SCHEDULER_TYPE = "exponential"  # Options: "linear", "exponential", "cosine"
    _C.MODEL.TOLERANCE_SCHEDULING.WARMUP_EPOCHS = 15    # Number of epochs to reach final tolerance
    
    # Savitzky-Golay specific parameters
    _C.MODEL.SG_LEARNABLE_WEIGHTS = True  # Whether to use learnable weights in SG filter
    _C.MODEL.SG_POLYNOMIAL_ORDER = 2  # Polynomial order for SG filter
    _C.MODEL.SECOND_ORDER = False  # Enable second-order Neural CDE with dual neural networks
    _C.MODEL.USE_REFIT = False  # Enable refitting during autoregressive prediction
    
    # Training configuration
    _C.TRAIN = CN()
    _C.TRAIN.NUM_EPOCHS = 20  # Reduced from 100; early-stopping will usually finish sooner
    _C.TRAIN.PATIENCE = 5          # Number of epochs to wait for improvement
    _C.TRAIN.MIN_DELTA = 1e-4      # Minimum change in monitored metric to qualify as improvement
    _C.TRAIN.LEARNING_RATE = 0.001
    _C.TRAIN.WEIGHT_DECAY = 1e-4
    _C.TRAIN.GRADIENT_CLIP_VAL = 1.0
    _C.TRAIN.SEED = 42
    _C.TRAIN.LOG_INTERVAL = 10
    
    # Learning rate scheduler
    _C.TRAIN.STEP_SIZE = 30
    _C.TRAIN.GAMMA = 0.1
    
    # System configuration
    _C.SYSTEM = CN()
    _C.SYSTEM.NUM_GPUS = 0  # Will be auto-detected
    
    # Output configuration
    _C.OUTPUT = CN()
    _C.OUTPUT.LOG_DIR = "./logs"
    _C.OUTPUT.MODEL_DIR = "./models"
    _C.OUTPUT.WANDB_PROJECT = "camera_ready_final"
    _C.OUTPUT.WANDB_ENTITY = None
    _C.OUTPUT.TOP_K_CHECKPOINTS = 5  # Keep only top K best checkpoints + final model
    
    # Debug mode
    _C.DEBUG = False
    
    return _C.clone()


def process_opts(opts):
    """
    Process command-line configuration overrides.
    Handles both 'KEY VALUE' and 'key=value' formats for wandb sweep compatibility.
    Converts dash-separated hierarchical keys to dot-separated for YACS compatibility.
    
    Args:
        opts: List of command line arguments for config overrides
        
    Returns:
        List of processed arguments suitable for update_config
    """
    if not opts:
        return []
    
    processed_opts = []
    
    i = 0
    while i < len(opts):
        arg = opts[i]
        
        # Handle key=value format (wandb sweep compatible)
        if '=' in arg:
            key, value = arg.split('=', 1)
            
            # Remove leading dashes from key (e.g., --DATA-BATCH_SIZE -> DATA-BATCH_SIZE)
            key = key.lstrip('-')
            
            # Convert dash-separated hierarchy to dot-separated
            if '-' in key:
                key = key.replace('-', '.')
            
            # Convert to uppercase for YACS compatibility
            key = key.upper()
            
            # Only process hierarchical keys
            if '.' in key:
                processed_opts.append(key)
                processed_opts.append(value)
            i += 1
        else:
            # Handle traditional 'KEY VALUE' format
            # Check if this looks like a config key (contains dots but not a file path)
            if '.' in arg and not (arg.startswith('/') or arg.startswith('./')):
                processed_opts.append(arg.upper())
                # If this is followed by a value, include that too
                if i + 1 < len(opts) and not ('.' in opts[i+1] and not (opts[i+1].startswith('/') or opts[i+1].startswith('./'))) and not ('=' in opts[i+1]):
                    value = opts[i+1]
                    processed_opts.append(value)  # Don't convert value to uppercase
                    i += 2
                else:
                    i += 1
            else:
                i += 1
    
    return processed_opts


def update_config(config, config_file=None, args=None):
    """
    Update config with values from file and command line args.
    
    Args:
        config: Configuration to update
        config_file: Path to YAML configuration file
        args: Command line arguments (list of key-value pairs)
        
    Returns:
        Updated configuration
    """
    # Update from file
    if config_file:
        if os.path.isfile(config_file):
            config.merge_from_file(config_file)
        else:
            raise Exception(f"Config file {config_file} does not exist")
    
    # Update from command line args
    if args:
        config.merge_from_list(args)
    
    # Auto-detect number of GPUs if not manually set
    import torch
    if not args or 'SYSTEM.NUM_GPUS' not in ' '.join(args):
        available_gpus = torch.cuda.device_count()
        if available_gpus > 0:
            config.SYSTEM.NUM_GPUS = available_gpus
            print(f"Auto-detected {available_gpus} available GPUs")
    
    return config


def to_dict(cfg_node, key_list=None):
    """
    Convert a config node to dictionary.
    
    Args:
        cfg_node (CN): Config node.
        key_list (list): Used for recursion.
        
    Returns:
        dict: A dict from the config node.
    """
    if not isinstance(cfg_node, CN):
        return cfg_node
    
    if key_list is None:
        key_list = []
        
    result = {}
    for k, v in cfg_node.items():
        if isinstance(v, CN):
            # Create a flattened representation for nested configs
            result[k] = to_dict(v)
        else:
            result[k] = v
            
    return result


def parse_args():
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(description="JAX/Flax Training for Rotational Dynamics")
    parser.add_argument("--config-file", default="", metavar="FILE", 
                       help="path to config file")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER,
                       help="Modify config options using the command-line 'KEY VALUE' pairs")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode (faster training with limited data)")
    
    args = parser.parse_args()
    
    # Get default config
    cfg = get_cfg_defaults()
    
    # Process and filter opts
    processed_opts = process_opts(args.opts) if args.opts else None
    
    # Update config with file and command line arguments
    cfg = update_config(cfg, args.config_file, processed_opts)
    
    # Set debug mode
    if args.debug:
        cfg.DEBUG = True
    
    # Make paths absolute
    cfg.DATA.PATH = os.path.abspath(cfg.DATA.PATH)
    cfg.OUTPUT.LOG_DIR = os.path.abspath(cfg.OUTPUT.LOG_DIR)
    cfg.OUTPUT.MODEL_DIR = os.path.abspath(cfg.OUTPUT.MODEL_DIR)
    
    # Extract dataset name from path
    cfg.DATA.DATASET_NAME = os.path.basename(cfg.DATA.PATH).split('.')[0]
    
    # Create output directories
    os.makedirs(cfg.OUTPUT.LOG_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT.MODEL_DIR, exist_ok=True)
    
    # Freeze config
    cfg.freeze()
    
    return cfg


def save_cfg(cfg, path):
    """Save configuration to file."""
    with open(path, 'w') as f:
        f.write(cfg.dump())
