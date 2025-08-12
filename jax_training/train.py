#!/usr/bin/env python
"""
Streamlined training script for SO3 GRU models with JAX/Flax.
"""
import os
import gc

# Fix PyTorch CUDA allocator issue
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ''
import time
import jax
import jax.numpy as jnp
import optax
import wandb
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Update import path for flax.nnx (no longer experimental)
from flax import nnx

from config import parse_args, save_cfg
from models import create_model_from_cfg, get_model_save_path
from experiments import update_cfg_from_dataset
from utils.data import create_dataloaders, create_test_dataloader
from utils.train import convert_batch_to_jax, compute_metrics, log_metrics, evaluate_model, get_model_name_from_cfg
from utils.test_utils import test_model_comprehensive


def initialize_environment(cfg):
    """Initialize JAX and PyTorch environment for optimal GPU usage."""
    # Set PyTorch to use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch using: {device}")
    
    # Configure JAX to use GPU if available
    if torch.cuda.is_available():
        try:
            jax.config.update('jax_platform_name', 'gpu')
        except Exception as e:
            print(f"Could not set JAX platform to GPU: {e}")
    
    print(f"JAX using: {jax.devices()[0].platform}")
    
    # Set a consistent random seed
    torch.manual_seed(cfg.TRAIN.SEED)
    key = jax.random.key(cfg.TRAIN.SEED)
    
    return device, key


def train_model(model, train_dataloader, val_dataloader, cfg, dataset_params=None):
    """
    Train a model using the training and validation dataloaders.
    
    Args:
        model: Model to train
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        cfg: Configuration object
        dataset_params: Additional dataset parameters
        
    Returns:
        Tuple of (trained model, best parameters)
    """
    # Initialize wandb
    model_name = get_model_name_from_cfg(cfg)
    run_name = f"{model_name}_{cfg.DATA.DATASET_NAME}_n{cfg.DATA.N_FUTURE}"
    
    # Convert config to dict and add model name for consistency with baseline
    wandb_config = dict(cfg)
    wandb_config['MODEL_NAME'] = model_name
    
    wandb.init(
        project=cfg.OUTPUT.WANDB_PROJECT,
        entity=cfg.OUTPUT.WANDB_ENTITY,
        config=wandb_config,
        name=run_name
    )
    
    # DEBUG: Add temporal noise levels to wandb config for evaluation
    # TODO: Remove this after temporal noise evaluation is fixed
    # Store temporal noise levels as individual config parameters for later access
    for i, noise_level in enumerate(cfg.DATA.TEST_TEMPORAL_NOISE_LEVELS):
        wandb.config.update({f'temporal_noise_level_{i}': noise_level})
    
    # Add dataset parameters to wandb config if available
    if dataset_params:
        for key in ['scenario_params', 'scenario_description', 'omega_scale']:
            if key in dataset_params and dataset_params[key] is not None:
                wandb.config.update({key: dataset_params[key]})
    
    # Create LR schedule (StepLR equivalent)
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=cfg.TRAIN.LEARNING_RATE,
        boundaries_and_scales={
            cfg.TRAIN.STEP_SIZE * len(train_dataloader): cfg.TRAIN.GAMMA,
            2 * cfg.TRAIN.STEP_SIZE * len(train_dataloader): cfg.TRAIN.GAMMA,
            3 * cfg.TRAIN.STEP_SIZE * len(train_dataloader): cfg.TRAIN.GAMMA
        }
    )
    
    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.TRAIN.GRADIENT_CLIP_VAL),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY
        )
    )
    
    # Initialize tolerance scheduler if enabled
    from utils.tolerance_scheduler import create_tolerance_scheduler_from_config
    tolerance_scheduler = create_tolerance_scheduler_from_config(cfg)
    if tolerance_scheduler:
        print(f"Using tolerance scheduler: {tolerance_scheduler}")
        # Set initial tolerances for models that support it
        if hasattr(model, 'update_tolerances'):
            initial_atol, initial_rtol = tolerance_scheduler.get_tolerances(0)
            model.update_tolerances(initial_atol, initial_rtol)
            print(f"Set initial tolerances: ATOL={initial_atol:.2e}, RTOL={initial_rtol:.2e}")
    
    # Extract model parameters and initialize optimizer state
    graphdef, params = nnx.split(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    # Define update step with JIT compilation
    @jax.jit
    def update_step(params, opt_state, t_recon, t_fut, x, targets, recon, omega, moi):
        """Perform a single training update step."""
        def loss_fn(params):
            model_with_params = nnx.merge(graphdef, params)
            # All models now have consistent interface - always returns 3 values
            recon_hat, pred_hat, solver_stats = model_with_params(
                t_recon, t_fut, x, omega, moi, 
                return_solver_stats=(cfg.MODEL.TYPE.upper() == "SO3NEURALCDE")
            )
            
            # Convert targets to 9D rotation matrices for loss computation
            if cfg.DATA.OUT_REP == "6d":
                # Flatten to (batch*seq, 6), convert to rotation matrices
                from utils.so3 import gramschmidt_to_rotmat
                targets_flat = targets.reshape(-1, 6)  # (batch*seq, 6)
                targets_reshaped = jax.vmap(gramschmidt_to_rotmat)(targets_flat)  # (batch*seq, 3, 3)
            else:
                # Already in 9D rotation matrix form, just flatten batch*seq
                targets_reshaped = targets.reshape(-1, 3, 3)
            
            # Model outputs are always 9D rotation matrices (post-processed)
            pred_hat_reshaped = pred_hat.reshape(-1, 3, 3)
            loss_pred = jnp.mean(jnp.linalg.norm(targets_reshaped - pred_hat_reshaped, axis=(1, 2)))
            
            # Calculate reconstruction loss if applicable
            if recon_hat is not None:
                if cfg.DATA.OUT_REP == "6d":
                    # Flatten to (batch*seq, 6), convert to rotation matrices
                    recon_flat = recon.reshape(-1, 6)  # (batch*seq, 6)
                    recon_reshaped = jax.vmap(gramschmidt_to_rotmat)(recon_flat)  # (batch*seq, 3, 3)
                else:
                    recon_reshaped = recon.reshape(-1, 3, 3)
                
                # Model reconstruction outputs are always 9D rotation matrices (post-processed)
                recon_hat_reshaped = recon_hat.reshape(-1, 3, 3)
                loss_recon = jnp.mean(jnp.linalg.norm(recon_reshaped - recon_hat_reshaped, axis=(1, 2)))
                total_loss = loss_pred + loss_recon
            else:
                loss_recon = jnp.array(0.0)
                total_loss = loss_pred
                
            return total_loss, (pred_hat, recon_hat, loss_pred, loss_recon, solver_stats)
        
        # Calculate loss and gradients
        (loss, (pred_hat, recon_hat, loss_pred, loss_recon, solver_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss, pred_hat, recon_hat, loss_pred, loss_recon, solver_stats
    
    # Training loop
    start_time = time.time()
    step_count = 0
    best_val_loss = float('inf')
    best_params = None
    best_checkpoints = []  # Track (loss, path) for top-K cleanup
    
    # Create log directory
    log_dir = os.path.join(cfg.OUTPUT.LOG_DIR, run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(log_dir, "config.yaml")
    save_cfg(cfg, config_path)
    
    # Set training/validation limits based on debug mode
    max_epochs = 1 if cfg.DEBUG else cfg.TRAIN.NUM_EPOCHS

    # Early stopping parameters
    patience = cfg.TRAIN.PATIENCE
    min_delta = cfg.TRAIN.MIN_DELTA
    epochs_since_improve = 0
    limit_train_batches = 5 if cfg.DEBUG else len(train_dataloader)
    limit_val_batches = 3 if cfg.DEBUG else None  # None means evaluate on all batches
    
    # Training loop
    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        
        # Update tolerances if scheduler is enabled
        if tolerance_scheduler and hasattr(model, 'update_tolerances'):
            atol, rtol = tolerance_scheduler.get_tolerances(epoch)
            model.update_tolerances(atol, rtol)
            # Log tolerance updates to wandb
            wandb.log({
                'solver_atol': atol,
                'solver_rtol': rtol,
                'epoch': epoch
            })
            if epoch == 0 or (epoch + 1) % 5 == 0:  # Log every 5 epochs or at start
                print(f"Epoch {epoch+1}: Updated tolerances to ATOL={atol:.2e}, RTOL={rtol:.2e}")
        
        train_metrics_list = []
        solver_steps_list = []  # Track solver steps per batch
        
        # Training phase
        train_pbar = tqdm(enumerate(train_dataloader), total=limit_train_batches, 
                         desc=f"Epoch {epoch+1}/{max_epochs}", leave=False)
        
        for i, batch in train_pbar:
            if i >= limit_train_batches:
                break
                
            # Convert batch to JAX
            jax_batch = convert_batch_to_jax(batch)
            t_recon, t_fut, x, targets, recon, omega, moi = jax_batch
            
            # Update parameters
            params, opt_state, loss, pred_hat, recon_hat, loss_pred, loss_recon, solver_stats = update_step(
                params, opt_state, t_recon, t_fut, x, targets, recon, omega, moi
            )
            
            # Compute metrics
            metrics = compute_metrics(pred_hat, targets, recon_hat, recon, cfg)
            train_metrics_list.append(metrics)
            
            # Track solver steps only for Neural CDE models that actually have solvers
            if cfg.MODEL.TYPE.upper() == "SO3NEURALCDE":
                if solver_stats is None:
                    raise RuntimeError(f"SO3NeuralCDE model returned None for solver_stats - this should never happen")
                
                num_steps = solver_stats.get('num_steps', 0)
                solver_steps_list.append(num_steps)
            
            step_count += 1
            
            # Update progress bar
            current_lr = float(lr_schedule(step_count))
            train_pbar.set_postfix({
                'Loss': f"{metrics['total_loss']:.4f}",
                'RE': f"{metrics['re_pred']:.2f}°",
                'LR': f"{current_lr:.2e}"
            })
            
            # Log to wandb periodically
            if i % cfg.TRAIN.LOG_INTERVAL == 0:
                log_metrics(metrics, split="train", step=step_count, epoch=epoch, lr=current_lr)
        
        # Update model with latest parameters
        model = nnx.merge(graphdef, params)
        
        # Compute average training metrics
        avg_train_metrics = {}
        for key in train_metrics_list[0].keys():
            avg_train_metrics[key] = np.mean([m[key] for m in train_metrics_list])
        
        # Evaluate on validation set
        val_metrics = evaluate_model(model, val_dataloader, limit_batches=limit_val_batches, split="val", cfg=cfg)
        
        # Check improvement for early stopping
        improved = val_metrics["loss_pred"] < (best_val_loss - min_delta)

        # Save best model if improved and manage top-K checkpoints
        if improved:
            best_val_loss = val_metrics["loss_pred"]
            best_params = params
            best_model_path = get_model_save_path(cfg, epoch=epoch, metric_value=val_metrics["re_pred"], is_best=True)
            with open(best_model_path, 'wb') as f:
                import pickle
                pickle.dump(params, f)
            print(f"New best model saved to {best_model_path}")
            
            # Track this checkpoint for top-K management
            best_checkpoints.append((val_metrics["loss_pred"], best_model_path))
            
            # Keep only top-K checkpoints
            if len(best_checkpoints) > cfg.OUTPUT.TOP_K_CHECKPOINTS:
                # Sort by loss (ascending - lower is better)
                best_checkpoints.sort(key=lambda x: x[0])
                # Remove the worst checkpoint (highest loss)
                worst_loss, worst_path = best_checkpoints.pop()
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                    print(f"Removed old checkpoint: {worst_path} (loss: {worst_loss:.4f})")
            
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        
        # Compute solver step statistics (only for Neural CDE models)
        solver_stats_str = ""
        if cfg.MODEL.TYPE.upper() == "SO3NEURALCDE":
            if not solver_steps_list:
                raise RuntimeError("SO3NeuralCDE model was used but no solver steps were tracked - this indicates a bug")
            
            solver_steps_array = np.array(solver_steps_list)
            if len(solver_steps_array) == 0:
                raise RuntimeError("Empty solver_steps_array for SO3NeuralCDE - this should never happen")
            
            avg_steps = np.mean(solver_steps_array)
            std_steps = np.std(solver_steps_array)
            max_steps = np.max(solver_steps_array)
            solver_stats_str = f" | Solver Steps: {avg_steps:.1f}±{std_steps:.1f} (max: {max_steps})"
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{max_epochs} completed in {epoch_time:.2f}s | "
              f"Train Loss: {avg_train_metrics['loss_pred']:.4f} | "
              f"Train RE: {avg_train_metrics['re_pred']:.2f}° | "
              f"Val Loss: {val_metrics['loss_pred']:.4f} | "
              f"Val RE: {val_metrics['re_pred']:.2f}°{solver_stats_str}")

        # Clear JAX and Diffrax caches after each epoch to prevent memory leaks
        gc.collect()
        jax.clear_caches()
        try:
            import diffrax
            if hasattr(diffrax.diffeqsolve, '_cached'):
                diffrax.diffeqsolve._cached.clear_cache()
        except (ImportError, AttributeError):
            pass

        # Early stopping condition
        if epochs_since_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs).")
            break
    
    # Save final model
    final_model_path = get_model_save_path(cfg)
    with open(final_model_path, 'wb') as f:
        import pickle
        pickle.dump(params, f)
    print(f"Final model saved to {final_model_path}")
    
    # Log total training time
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f}s")
    
    # Keep wandb run active for testing
    
    return model, best_params


def test_model(model, cfg, dataset_params=None):
    """
    Test model on different horizons and noise levels.
    Uses consolidated testing utilities for consistency.
    
    Args:
        model: Model to test
        cfg: Configuration object
        dataset_params: Additional dataset parameters
        
    Returns:
        DataFrame with test results
    """
    # Use consolidated testing function
    df = test_model_comprehensive(model, cfg, dataset_params)
    
    # Finish wandb logging (consolidated function doesn't call finish to allow additional logging)
    wandb.finish()
    
    return df


def main():
    """Main entry point for training and testing."""
    # Parse command line arguments and get configuration
    cfg = parse_args()
    
    # Configuration is now handled by method-specific config files
    print(f"Using interpolation method: {cfg.MODEL.INTERPOLATION_METHOD}")
    print(f"Using wandb project: {cfg.OUTPUT.WANDB_PROJECT}")
    
    # Load parameters from dataset and update config
    cfg, dataset_params = update_cfg_from_dataset(cfg, cfg.DATA.PATH)
    
    # Initialize environment (JAX, PyTorch, GPU settings)
    device, key = initialize_environment(cfg)
    
    # Create model
    model = create_model_from_cfg(cfg)
    print(f"Initialized {cfg.MODEL.TYPE} model")
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(cfg)
    
    # Train model
    model, best_params = train_model(model, train_dataloader, val_dataloader, cfg, dataset_params)
    
    # Replace model parameters with the best-found parameters for evaluation
    if best_params is not None:
        graphdef, _ = nnx.split(model, nnx.Param)
        model = nnx.merge(graphdef, best_params)
        print("Loaded best validation parameters for testing.")
    else:
        print("Warning: best_params is None; using last epoch parameters for testing.")
    
    # Test model (allowing debug mode for testing wandb fix)
    test_results = test_model(model, cfg, dataset_params)
    print("Testing completed successfully.")
    
    
    return model, best_params


if __name__ == "__main__":
    main()
