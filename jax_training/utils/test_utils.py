#!/usr/bin/env python
"""
Consolidated testing utilities for SO3 models with JAX/Flax.
"""
import os
import gc
import jax
import wandb
import numpy as np
import pandas as pd
from pathlib import Path
from utils.data import create_test_dataloader
from utils.train import evaluate_model, get_model_name_from_cfg


class ModelNotFoundException(Exception):
    """Exception raised when a model checkpoint cannot be found."""
    pass


def get_best_checkpoint_jax(run_id: str, project: str = None) -> tuple[Path, dict]:
    """
    Find the best checkpoint for a given wandb run ID and get its configuration.
    Adapted for JAX/Flax implementation.
    
    Args:
        run_id: WandB run ID
        project: WandB project name (if None, attempts to infer from config)
        
    Returns:
        tuple: (checkpoint_path, config_dict)
        
    Raises:
        ModelNotFoundException: If no checkpoint is found or directory doesn't exist
    """
    # Get run configuration from wandb
    api = wandb.Api()
    
    # If no project specified, try to infer from run
    if project is None:
        try:
            run = api.run(run_id)
        except Exception:
            # Try common project names
            for proj in ["rotdyn_jax", "variable_final", "rot_dyn"]:
                try:
                    run = api.run(f"{proj}/{run_id}")
                    project = proj
                    break
                except Exception:
                    continue
            if project is None:
                raise ModelNotFoundException(f"Could not find run {run_id} in any common project")
    else:
        run = api.run(f"{project}/{run_id}")
    
    # Look for checkpoint in typical locations using wandb project name
    checkpoint_dirs = [
        Path("checkpoints") / run_id,
        Path("outputs") / project / run_id / "checkpoints",
        Path("outputs") / "logs" / run.name / "checkpoints", 
        Path("outputs") / run.name / "checkpoints",
        Path("jax_training") / "outputs" / "logs" / run.name / "checkpoints",
        # Check for actual run name subdirectory structure
        Path("outputs") / run.name / project / run_id / "checkpoints",
        # Check for legacy structures (specific project names)
        Path("outputs") / "so3_2nd_order_test" / project / run_id / "checkpoints",
        Path("outputs") / "rot_dyn_jax" / run_id / "checkpoints",
    ]
    
    checkpoint_path = None
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.exists():
            # Find best checkpoint file (prioritize best over final)
            checkpoint_files = list(checkpoint_dir.glob("best-checkpoint*.pkl"))
            if not checkpoint_files:
                checkpoint_files = list(checkpoint_dir.glob("final-model.pkl"))
            if checkpoint_files:
                # Use the latest best checkpoint if multiple exist
                checkpoint_path = sorted(checkpoint_files)[-1]
                break
    
    if checkpoint_path is None:
        raise ModelNotFoundException(f"No checkpoint file found for run {run_id} in any of: {[str(d) for d in checkpoint_dirs]}")
    
    # Extract configuration relevant for JAX models
    config = {
        'model_name': run.config.get('MODEL_NAME', run.config.get('model_name')),
        'model_type': run.config.get('MODEL.TYPE', run.config.get('MODEL_TYPE')),
        'interpolation_method': run.config.get('MODEL.INTERPOLATION_METHOD', run.config.get('interpolation_method')),
        'dataset_name': run.config.get('DATA.DATASET_NAME', run.config.get('dataset_name')),
        'n_future': run.config.get('DATA.N_FUTURE', run.config.get('n_future')),
        'scenario_params': run.config.get('scenario_params', {}),
        'scenario_description': run.config.get('scenario_description', ''),
        'omega_scale': run.config.get('omega_scale', None),
        'training_run_id': run_id,
        'project': project,
        # Add other relevant config parameters
        'hidden_size': run.config.get('MODEL.HIDDEN_SIZE', run.config.get('hidden_size')),
        'num_layers': run.config.get('MODEL.NUM_LAYERS', run.config.get('num_layers')),
        'use_refit': run.config.get('MODEL.USE_REFIT', run.config.get('use_refit')),
        'second_order': run.config.get('MODEL.SECOND_ORDER', run.config.get('second_order')),
        'sg_learnable_weights': run.config.get('MODEL.SG_LEARNABLE_WEIGHTS', run.config.get('sg_learnable_weights')),
    }
    
    print(f"\nFound configuration for run {run_id}:")
    for key, value in config.items():
        if key not in ['scenario_params'] and value is not None:
            print(f"  {key}: {value}")
    print(f"\nUsing checkpoint: {checkpoint_path}")
    
    return checkpoint_path, config


def test_model_comprehensive(model, cfg, dataset_params=None, wandb_project=None):
    """
    Test model on different horizons and noise levels.
    Extracted from main train.py for reusability.
    
    Args:
        model: Model to test
        cfg: Configuration object
        dataset_params: Additional dataset parameters
        wandb_project: Wandb project name
        
    Returns:
        DataFrame with test results
    """
    # Configure noise levels and horizons for testing from config
    noise_levels = cfg.DATA.TEST_NOISE_LEVELS
    horizons = cfg.DATA.TEST_HORIZONS
    
    if cfg.DEBUG:
        noise_levels = noise_levels[:2]
        horizons = horizons[:2]
    
    # Initialize results
    results = []
    
    # Initialize wandb if no active run exists
    model_name = get_model_name_from_cfg(cfg)
    run_name = f"test_{model_name}_{cfg.DATA.DATASET_NAME}"
    if wandb.run is None:
        # Convert config to dict and add model name for consistency with baseline
        wandb_config = dict(cfg)
        wandb_config['MODEL_NAME'] = model_name
        
        wandb.init(
            project=wandb_project if wandb_project else cfg.OUTPUT.WANDB_PROJECT,
            entity=cfg.OUTPUT.WANDB_ENTITY,
            config=wandb_config,
            name=run_name
        )
    
    # Add dataset parameters to wandb config if available
    if dataset_params:
        for key in ['scenario_params', 'scenario_description', 'omega_scale']:
            if key in dataset_params and dataset_params[key] is not None:
                wandb.config.update({key: dataset_params[key]})
    
    # Loop through each horizon
    for horizon in horizons:
        print(f"Testing horizon {horizon}...")
        
        # Loop through each noise level
        for noise in noise_levels:
            print(f"  Noise level: {noise:.2f}")
            
            # Create test dataloader
            test_dataloader = create_test_dataloader(cfg, horizon, noise)
            
            # Create model kwargs for this noise level
            model_kwargs = {'omega_noise': noise, 'key': jax.random.PRNGKey(cfg.TRAIN.SEED)}
            
            # Evaluate model
            test_metrics = evaluate_model(model, test_dataloader, split="test", cfg=cfg, model_kwargs=model_kwargs)
            
            # Explicit cleanup to prevent memory leaks
            from utils.data import cleanup_dataloader
            cleanup_dataloader(test_dataloader)
            del test_dataloader
            
            # Add horizon and noise level to metrics
            test_metrics["horizon"] = horizon
            test_metrics["noise"] = noise
            
            # Log to wandb with specific horizon/noise tags
            wandb_test_metrics = {
                f"test/h{horizon}_n{noise:.2f}_{key}": value 
                for key, value in test_metrics.items() 
                if key not in ["horizon", "noise"]
            }
            # Also log with general test prefix for easy filtering
            wandb_test_metrics.update({
                f"test_summary/horizon": horizon,
                f"test_summary/noise": noise,
                f"test_summary/re_pred": test_metrics["re_pred"],
                f"test_summary/loss_pred": test_metrics["loss_pred"]
            })
            wandb.log(wandb_test_metrics)
            
            # Add to results
            results.append(test_metrics)
            
            print(f"    Loss: {test_metrics['loss_pred']:.4f}, "
                  f"RE: {test_metrics['re_pred']:.2f}° ± {test_metrics['re_pred_std']:.2f}°")
            
            # Force garbage collection and clear both JAX and Diffrax caches after each test
            gc.collect()
            jax.clear_caches()
            
            # Clear Diffrax's internal compilation cache (the real culprit!)
            try:
                import diffrax
                if hasattr(diffrax.diffeqsolve, '_cached'):
                    diffrax.diffeqsolve._cached.clear_cache()
            except (ImportError, AttributeError):
                pass  # Diffrax not available or cache structure changed
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Create output directory
    output_dir = os.path.join(cfg.OUTPUT.LOG_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    csv_path = os.path.join(output_dir, "test_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Create and save visualization
    try:
        from utils.plot import plot_prediction_horizons
        fig = plot_prediction_horizons(df, save_path=os.path.join(output_dir, "prediction_comparison.png"))
        
        # Log summary to wandb
        wandb.log({
            "prediction_comparison": wandb.Image(fig),
            "results_summary": df.groupby('horizon')['re_pred']
                                  .agg(['mean', 'std']).to_dict()
        })
    except ImportError:
        print("Warning: Could not import plotting utilities. Skipping visualization.")
    
    return df


def run_full_evaluation(model, cfg, dataset_params=None, wandb_project=None):
    """
    Run comprehensive evaluation.
    
    Args:
        model: Model to test
        cfg: Configuration object  
        dataset_params: Additional dataset parameters
        wandb_project: Explicit wandb project name (overrides config)
        
    Returns:
        DataFrame with test results
    """
    print("Running standard testing...")
    test_results = test_model_comprehensive(model, cfg, dataset_params, wandb_project)
    return test_results