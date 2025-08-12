#!/usr/bin/env python
"""
Standalone testing script for SO3 models with JAX/Flax.
Uses wandb API to retrieve model configuration and checkpoint.
"""
import os
import argparse
import pickle
import jax
import numpy as np
from pathlib import Path
from flax import nnx

from config import get_cfg_defaults, process_opts, update_config
from models import create_model_from_cfg
from experiments import update_cfg_from_dataset
from utils.test_utils import get_best_checkpoint_jax, run_full_evaluation


def create_cfg_from_wandb_run(wandb_run):
    """
    Create a proper configuration object from wandb run.
    
    Args:
        wandb_run: Wandb run object
        
    Returns:
        Configuration object compatible with training pipeline
    """
    # Start with default configuration
    cfg = get_cfg_defaults()
    
    # Helper function to recursively update config from wandb config
    def update_config_from_wandb(cfg_section, wandb_section, section_name=""):
        """Recursively update config section from wandb config section."""
        if not isinstance(wandb_section, dict):
            return
        
        for key, value in wandb_section.items():
            if hasattr(cfg_section, key):
                current_value = getattr(cfg_section, key)
                # If current value is a config node, recurse
                if hasattr(current_value, '__dict__') and isinstance(value, dict):
                    update_config_from_wandb(current_value, value, f"{section_name}.{key}")
                else:
                    # Update the value
                    setattr(cfg_section, key, value)
                    if section_name:
                        print(f"  Updated {section_name}.{key}: {current_value} -> {value}")
    
    # Update config sections from wandb config
    print("Updating config from wandb run...")
    for section_name in ['MODEL', 'DATA', 'TRAIN']:
        if section_name in wandb_run.config:
            wandb_section = wandb_run.config[section_name]
            cfg_section = getattr(cfg, section_name)
            update_config_from_wandb(cfg_section, wandb_section, section_name)
    
    # Handle special cases and fallbacks
    model_name = wandb_run.config.get('MODEL_NAME', '')
    
    # Auto-detect model type from name if not set
    if 'SG_filter' in model_name or 'neural_cde' in model_name.lower():
        cfg.MODEL.TYPE = "SO3NeuralCDE"
        
        # Fill in missing parameters from model name
        if not getattr(cfg.MODEL, 'INTERPOLATION_METHOD', None) and 'SG_filter' in model_name:
            cfg.MODEL.INTERPOLATION_METHOD = 'savitzky_golay'
        if not getattr(cfg.MODEL, 'SG_LEARNABLE_WEIGHTS', None) and '_W_' in model_name:
            cfg.MODEL.SG_LEARNABLE_WEIGHTS = True
        if not getattr(cfg.MODEL, 'SECOND_ORDER', None) and '2nd_order' in model_name:
            cfg.MODEL.SECOND_ORDER = True
    
    # Handle missing dataset name
    if not getattr(cfg.DATA, 'DATASET_NAME', None):
        scenario = wandb_run.config.get('scenario_description')
        if scenario:
            cfg.DATA.DATASET_NAME = f"rigid_body_{scenario}"
    
    # Output and training configuration with simple defaults  
    cfg.OUTPUT.WANDB_ENTITY = None
    cfg.TRAIN.SEED = 42
    
    return cfg


def load_model_from_checkpoint(checkpoint_path, cfg):
    """
    Load a trained model from checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        cfg: Configuration object
        
    Returns:
        Loaded model with trained parameters
    """
    # Create model architecture
    model = create_model_from_cfg(cfg)
    
    # Load saved parameters
    with open(checkpoint_path, 'rb') as f:
        saved_params = pickle.load(f)
    
    # Split model to get graph definition and merge with saved parameters
    graphdef, _ = nnx.split(model, nnx.Param)
    model = nnx.merge(graphdef, saved_params)
    
    print(f"Model loaded from {checkpoint_path}")
    return model


def initialize_environment(cfg):
    """Initialize JAX environment for optimal GPU usage."""
    # Configure JAX to use GPU if available
    if jax.devices()[0].platform == 'gpu':
        print(f"JAX using: {jax.devices()[0].platform}")
    else:
        print(f"JAX using: {jax.devices()[0].platform} (GPU not available)")
    
    # Set a consistent random seed
    key = jax.random.key(cfg.TRAIN.SEED)
    return key


def main():
    """Main entry point for standalone testing."""
    parser = argparse.ArgumentParser(description="Test SO3 models from wandb run")
    parser.add_argument("--run_id", type=str, required=False, 
                       help="WandB run ID to load model from")
    parser.add_argument("--project", type=str, required=False,
                       help="WandB project name")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Direct path to model checkpoint file (.pkl)")
    parser.add_argument("--model_type", type=str, default=None,
                       help="Model type for non-parametric models (e.g., 'hamiltonian')")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Path to dataset file (.pt) for non-parametric models")
    parser.add_argument("--output_dir", type=str, default="./test_outputs",
                       help="Directory to save test results")
    parser.add_argument("--debug", action='store_true',
                       help="Run in debug mode (reduced test scope)")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER,
                       help="Modify config options using command line (e.g., --opts MODEL.USE_REFIT True)")
    
    args = parser.parse_args()
    
    # Handle different model types
    if args.model_type:
        print(f"Testing non-parametric model: {args.model_type}")
        if args.model_type.lower() == "hamiltonian":
            if not args.dataset:
                print("Error: --dataset is required for Hamiltonian model")
                return 1
        else:
            print(f"Error: Unknown non-parametric model type: {args.model_type}")
            return 1
    else:
        if not args.run_id:
            print("Error: Either --run_id or --model_type must be specified")
            return 1
        print(f"Testing model from wandb run: {args.run_id}")
    
    # Get checkpoint path and wandb configuration
    try:
        if args.model_type:
            # Non-parametric model - no checkpoint or wandb run needed
            checkpoint_path = None
            wandb_run = None
        elif args.checkpoint_path:
            # Use provided checkpoint path
            checkpoint_path = Path(args.checkpoint_path)
            if not checkpoint_path.exists():
                print(f"Error: Checkpoint file not found: {checkpoint_path}")
                return 1
            
            # Still get wandb config for model parameters
            import wandb
            api = wandb.Api()
            run = api.run(f"{args.project}/{args.run_id}")
            
            wandb_run = run
        else:
            # Use the wandb-based checkpoint finding
            checkpoint_path, wandb_config = get_best_checkpoint_jax(args.run_id, args.project)
            wandb_run = None  # Will need to get run again for proper config access
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 1
    
    # Create configuration
    if args.model_type:
        # Non-parametric model - create config from defaults
        cfg = get_cfg_defaults()
        cfg.MODEL.TYPE = args.model_type
        
        # Override representations for Hamiltonian inference
        if args.model_type.lower() == "hamiltonian":
            cfg.DATA.IN_REP = 'quat'   # Input as quaternions
            cfg.DATA.OUT_REP = '9d'    # Output as rotation matrices (matches metrics expectations)
            cfg.DATA.PATH = os.path.abspath(args.dataset)
            cfg.DATA.DATASET_NAME = os.path.basename(cfg.DATA.PATH).split('.')[0]
            # Keep standard batch sizes - vectorized ODE integration can handle them
    else:
        # Get wandb run for proper config access
        if wandb_run is None:
            import wandb
            api = wandb.Api()
            wandb_run = api.run(f"{args.project}/{args.run_id}")
        
        # Create configuration from wandb run
        cfg = create_cfg_from_wandb_run(wandb_run)
    
    # Apply command line config overrides
    if args.opts:
        processed_opts = process_opts(args.opts)
        if processed_opts:
            print(f"Applying config overrides: {processed_opts}")
            cfg = update_config(cfg, config_file=None, args=processed_opts)
    
    # Override with command line arguments
    if args.debug:
        cfg.DEBUG = True
    else:
        cfg.DEBUG = False
    
    # Set dataset path with fallback logic
    if not args.model_type:  # Only for parametric models with wandb config
        if cfg.DATA.PATH:
            print(f"Using dataset path from wandb config: {cfg.DATA.PATH}")
        elif cfg.DATA.DATASET_NAME:
            cfg.DATA.PATH = f"./data/{cfg.DATA.DATASET_NAME}.pt"
            print(f"Constructed dataset path from dataset name: {cfg.DATA.PATH}")
        else:
            scenario = wandb_run.config.get('scenario_description') if wandb_run else None
            if scenario:
                cfg.DATA.PATH = f"./data/rigid_body_{scenario}.pt"
                print(f"Constructed dataset path from scenario: {cfg.DATA.PATH}")
            else:
                print("Warning: No dataset path available in wandb config")
    
    # Update output directory
    cfg.OUTPUT.LOG_DIR = args.output_dir
    
    print(f"Configuration created:")
    print(f"  Model type: {cfg.MODEL.TYPE}")
    print(f"  Dataset: {cfg.DATA.PATH}")
    print(f"  Debug mode: {cfg.DEBUG}")
    if args.model_type and args.model_type.lower() == "hamiltonian":
        print(f"  Non-parametric model: {args.model_type}")
        print(f"  Data representations: IN_REP={cfg.DATA.IN_REP}, OUT_REP={cfg.DATA.OUT_REP}")
    else:
        print(f"  Key model params: LATENT_DIM={cfg.MODEL.LATENT_DIM}, MLP_WIDTH={cfg.MODEL.MLP_WIDTH}, INPUT_CHANNEL={cfg.MODEL.INPUT_CHANNEL}, OUTPUT_CHANNEL={cfg.MODEL.OUTPUT_CHANNEL}")
    
    # Load dataset parameters and update config
    try:
        cfg, dataset_params = update_cfg_from_dataset(cfg, cfg.DATA.PATH)
        print(f"  Dataset parameters loaded from {cfg.DATA.PATH}")
    except Exception as e:
        print(f"Warning: Could not load dataset parameters: {e}")
        dataset_params = None
    
    # Initialize environment
    key = initialize_environment(cfg)
    
    # Load model (either from checkpoint or create non-parametric model)
    try:
        if args.model_type:
            # Non-parametric model - create directly without checkpoint
            from models import create_model_from_cfg
            model = create_model_from_cfg(cfg)
            print(f"Created non-parametric model: {cfg.MODEL.TYPE}")
        else:
            # Parametric model - load from checkpoint
            model = load_model_from_checkpoint(checkpoint_path, cfg)
            print(f"Loaded model from checkpoint: {cfg.MODEL.TYPE}")
    except Exception as e:
        print(f"Error loading/creating model: {e}")
        return 1
    
    # Run comprehensive evaluation
    print("\n=== Starting Model Evaluation ===")
    
    try:
        results = run_full_evaluation(
            model=model,
            cfg=cfg,
            dataset_params=dataset_params,
            wandb_project=args.project if args.project else None
        )
        
        print("=== Evaluation Complete ===")
        print(f"Results saved to: {cfg.OUTPUT.LOG_DIR}")
        
        # Print summary
        if results is not None:
            mean_re = results.groupby('horizon')['re_pred'].mean()
            print("\nTest Results Summary (Rotation Error by Horizon):")
            for horizon, re in mean_re.items():
                print(f"  Horizon {horizon}: {re:.2f}°")
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
