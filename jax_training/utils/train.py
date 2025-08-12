"""
Streamlined utilities for SO3 models.
"""
import jax
import jax.numpy as jnp
import numpy as np
import torch
import wandb
from tqdm import tqdm

# Update import path for flax.nnx (no longer experimental)
from flax import nnx

from utils.so3 import rotmat_geodesic_distance


def get_model_name_from_cfg(cfg):
    """
    Generate model name based on configuration parameters for Weights & Biases.
    
    Convention:
    - so3_spline_model: regular hermite spline neural CDE
    - so3_SG_filter_model: SO3 model with SG filtering  
    - so3_SG_filter_W_model: SO3 model with weighted SG filtering
    - so3_SG_filter_W_model_2nd_order: weighted SG filtering with 2nd order neural CDE
    
    Args:
        cfg: Configuration object
        
    Returns:
        str: Model name for logging
    """
    if cfg.MODEL.TYPE.upper() == "SO3NEURALCDE":
        # Check interpolation method
        if cfg.MODEL.INTERPOLATION_METHOD in ["savitzky_golay", "sg"]:
            # Savitzky-Golay based models
            if cfg.MODEL.SG_LEARNABLE_WEIGHTS:
                # Weighted SG filtering
                base_name = "so3_SG_filter_W"
                if cfg.MODEL.USE_REFIT:
                    base_name += "_refit"
                base_name += "_model"
                if cfg.MODEL.SECOND_ORDER:
                    base_name += "_2nd_order"
                return base_name
            else:
                # Regular SG filtering (non-weighted)
                base_name = "so3_SG_filter"
                if cfg.MODEL.USE_REFIT:
                    base_name += "_refit"
                base_name += "_model"
                if cfg.MODEL.SECOND_ORDER:
                    base_name += "_2nd_order"
                return base_name
        else:
            # Hermite spline or other interpolation methods
            base_name = f"so3_{cfg.MODEL.INTERPOLATION_METHOD}_model"
            if cfg.MODEL.USE_REFIT:
                base_name = base_name.replace("_model", "_refit_model")
            if cfg.MODEL.SECOND_ORDER:
                base_name += "_2nd_order"
            return base_name
    else:
        # For GRU or other models, use the original naming
        return cfg.MODEL.TYPE


def convert_batch_to_jax(batch):
    """
    Convert a batch of PyTorch tensors to JAX arrays with efficient GPU transfer.
    
    Args:
        batch: Tuple of PyTorch tensors from dataloader
        
    Returns:
        Tuple of JAX arrays
    """
    (t_recon, t_fut, x), targets, recon, omega, moi = batch
    
    # Use DLPack for efficient zero-copy GPU transfer when possible
    def pytorch_to_jax(torch_tensor):
        if torch_tensor.is_cuda:
            try:
                # Use DLPack for zero-copy GPU transfer
                dlpack = torch.utils.dlpack.to_dlpack(torch_tensor)
                return jax.dlpack.from_dlpack(dlpack)
            except (RuntimeError, ValueError, TypeError) as e:
                # Fall back to numpy bridge
                return jnp.array(torch_tensor.detach().cpu().numpy())
        else:
            # CPU tensor uses numpy bridge
            return jnp.array(torch_tensor.detach().numpy())
    
    # Convert all tensors to JAX
    t_recon_jax = pytorch_to_jax(t_recon)
    t_fut_jax = pytorch_to_jax(t_fut)
    x_jax = pytorch_to_jax(x)
    targets_jax = pytorch_to_jax(targets)
    recon_jax = pytorch_to_jax(recon)
    omega_jax = pytorch_to_jax(omega)
    moi_jax = pytorch_to_jax(moi)
    
    return t_recon_jax, t_fut_jax, x_jax, targets_jax, recon_jax, omega_jax, moi_jax


def compute_metrics(pred_hat, targets, recon_hat=None, recon=None, cfg=None):
    """
    Compute metrics for model outputs.
    
    Args:
        pred_hat: Predicted values (should be 9D rotation matrices from model post-processing)
        targets: Target values (may be 6D or 9D depending on cfg.DATA.OUT_REP)
        recon_hat: Reconstructed values (should be 9D rotation matrices from model post-processing)
        recon: Original values to reconstruct (may be 6D or 9D depending on cfg.DATA.OUT_REP)
        cfg: Configuration object to determine representation
        
    Returns:
        Dictionary of metrics
    """
    # Convert targets to 9D rotation matrices if needed
    if cfg is not None and cfg.DATA.OUT_REP == "6d":
        # Handle 6D representation: targets should be (batch, seq, 3, 2) or (batch*seq, 3, 2)
        from utils.so3 import gramschmidt_to_rotmat
        if len(targets.shape) == 4:
            # Shape is (batch, seq, 3, 2), reshape to (batch*seq, 3, 2) then flatten to (batch*seq, 6)
            batch_size, seq_len, _, _ = targets.shape
            targets_flat = targets.reshape(batch_size * seq_len, 6)  # (batch*seq, 6)
        elif len(targets.shape) == 3:
            # Shape is (batch*seq, 3, 2), flatten to (batch*seq, 6)
            targets_flat = targets.reshape(-1, 6)  # (batch*seq, 6)
        else:
            # Shape is (batch*seq, 6), already flattened
            targets_flat = targets
        targets_rotmat = jax.vmap(gramschmidt_to_rotmat)(targets_flat)  # (batch*seq, 3, 3)
        targets_reshaped = targets_rotmat
    else:
        # Already in 9D rotation matrix form, just flatten batch*seq
        targets_reshaped = targets.reshape(-1, 3, 3)
    
    # Model outputs should already be 9D rotation matrices (post-processed)
    pred_hat_reshaped = pred_hat.reshape(-1, 3, 3)
    loss_pred = jnp.mean(jnp.linalg.norm(targets_reshaped - pred_hat_reshaped, axis=(1, 2)))
    
    # Calculate rotation error in SO(3) space
    re_pred = jnp.mean(rotmat_geodesic_distance(targets_reshaped, pred_hat_reshaped))
    re_pred_degrees = float(re_pred) * 180 / np.pi
    
    # Initialize results
    metrics = {
        "loss_pred": float(loss_pred),
        "re_pred": re_pred_degrees
    }
    
    # Add reconstruction metrics if available
    if recon_hat is not None and recon is not None:
        # Convert reconstruction targets to 9D if needed
        if cfg is not None and cfg.DATA.OUT_REP == "6d":
            # Handle 6D representation: recon should be (batch, seq, 3, 2) or (batch*seq, 3, 2)
            if len(recon.shape) == 4:
                # Shape is (batch, seq, 3, 2), reshape to (batch*seq, 3, 2) then flatten to (batch*seq, 6)
                batch_size, seq_len, _, _ = recon.shape
                recon_flat = recon.reshape(batch_size * seq_len, 6)  # (batch*seq, 6)
            elif len(recon.shape) == 3:
                # Shape is (batch*seq, 3, 2), flatten to (batch*seq, 6)
                recon_flat = recon.reshape(-1, 6)  # (batch*seq, 6)
            else:
                # Shape is (batch*seq, 6), already flattened
                recon_flat = recon
            recon_rotmat = jax.vmap(gramschmidt_to_rotmat)(recon_flat)  # (batch*seq, 3, 3)
            recon_reshaped = recon_rotmat
        else:
            recon_reshaped = recon.reshape(-1, 3, 3)
        
        # Model reconstruction should already be 9D rotation matrices (post-processed)
        recon_hat_reshaped = recon_hat.reshape(-1, 3, 3)
        loss_recon = jnp.mean(jnp.linalg.norm(recon_reshaped - recon_hat_reshaped, axis=(1, 2)))
        metrics["loss_recon"] = float(loss_recon)
        metrics["total_loss"] = float(loss_pred) + float(loss_recon)
    else:
        metrics["loss_recon"] = 0.0
        metrics["total_loss"] = float(loss_pred)
    
    return metrics


def log_metrics(metrics, split="train", step=None, epoch=None, lr=None):
    """
    Log metrics to wandb.
    
    Args:
        metrics: Dictionary of metrics
        split: Data split (train, val, test)
        step: Training step
        epoch: Training epoch
        lr: Learning rate
    """
    # Add prefix to metrics
    wandb_metrics = {f"{split}/{key}": value for key, value in metrics.items()}
    
    # Add step, epoch, and learning rate
    if step is not None:
        wandb_metrics["step"] = step
    if epoch is not None:
        wandb_metrics["epoch"] = epoch
    if lr is not None:
        wandb_metrics["lr"] = lr
    
    # Log to wandb
    wandb.log(wandb_metrics)


def evaluate_model(model, dataloader, limit_batches=None, split="val", cfg=None, model_kwargs=None):
    """
    Evaluate model on dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: Dataloader for evaluation
        limit_batches: Maximum number of batches (None for all)
        split: Data split name for logging
        cfg: Configuration object for representation handling
        model_kwargs: Additional kwargs to pass to model forward pass
        
    Returns:
        Dictionary of aggregated metrics
    """
    # Initialize metrics collection
    all_metrics = []
    
    # Set batch limit
    batch_limit = len(dataloader) if limit_batches is None else min(limit_batches, len(dataloader))
    
    # Evaluate all batches with progress bar
    pbar = tqdm(enumerate(dataloader), total=batch_limit, desc=f"{split.capitalize()} Eval")
    for i, batch in pbar:
        if i >= batch_limit:
            break
        
        # Convert to JAX
        t_recon, t_fut, x, targets, recon, omega, moi = convert_batch_to_jax(batch)
        
        # Forward pass - all models now return 3 values consistently
        if model_kwargs:
            recon_hat, pred_hat, _ = model(t_recon, t_fut, x, omega, moi, **model_kwargs)
        else:
            recon_hat, pred_hat, _ = model(t_recon, t_fut, x, omega, moi)
        
        # Compute metrics
        batch_metrics = compute_metrics(pred_hat, targets, recon_hat, recon, cfg)
        all_metrics.append(batch_metrics)
    
    # Aggregate metrics across batches
    agg_metrics = {}
    metric_keys = all_metrics[0].keys()
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        agg_metrics[key] = np.mean(values)
        agg_metrics[f"{key}_std"] = np.std(values)
    
    # Log metrics
    log_metrics(agg_metrics, split=split)
    
    return agg_metrics


def evaluate_temporal_noise_robustness(model, cfg, mode='test', limit_batches=None):
    """
    Evaluate model robustness across different temporal noise levels.
    Tests each temporal noise level with horizons 8 and 10, rotational noise 0.0 and 0.05.
    
    Args:
        model: Model to evaluate
        cfg: Configuration object
        mode: Data mode ('test', 'val')
        limit_batches: Maximum number of batches per noise level
        
    Returns:
        Dictionary of metrics for each temporal noise level
    """
    from .data import create_dataset
    import torch
    
    temporal_noise_levels = cfg.DATA.TEST_TEMPORAL_NOISE_LEVELS
    # Focus on key horizons and rotational noise levels
    horizons = [8, 10]
    rotational_noise_levels = [0.0, 0.05]
    
    if cfg.DEBUG:
        temporal_noise_levels = temporal_noise_levels[:2]
        horizons = horizons[:1]
        rotational_noise_levels = rotational_noise_levels[:1]
    
    results = {}
    
    print(f"Evaluating temporal noise robustness with {len(temporal_noise_levels)} temporal noise levels...")
    print(f"Testing horizons {horizons} and rotational noise levels {rotational_noise_levels}")
    
    for temp_noise in temporal_noise_levels:
        print(f"\n=== Temporal Noise Level: {temp_noise} ===")
        results[temp_noise] = {}
        
        # Test each horizon
        for horizon in horizons:
            print(f"  Testing horizon {horizon}...")
            results[temp_noise][horizon] = {}
            
            # Test each rotational noise level
            for rot_noise in rotational_noise_levels:
                print(f"    Rotational noise: {rot_noise:.2f}")
                
                # Create a temporary copy of the config
                from yacs.config import CfgNode as CN
                temp_cfg = CN(cfg)
                temp_cfg.defrost()
                temp_cfg.DATA.TEMPORAL_NOISE_LEVEL = temp_noise
                temp_cfg.DATA.N_FUTURE = horizon
                temp_cfg.freeze()
                
                # Create dataset for this configuration
                dataset = create_dataset(temp_cfg, mode=mode, noise_level=rot_noise)
                
                # Create dataloader
                batch_size = cfg.DATA.VAL_BATCH_SIZE if mode == 'val' else 1000
                dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=False,  # No shuffling for consistent evaluation
                    num_workers=0 if cfg.DEBUG else cfg.DATA.NUM_WORKERS,
                    persistent_workers=(0 if cfg.DEBUG else cfg.DATA.NUM_WORKERS) > 0  # Enable persistent workers for spawn efficiency
                )
                
                # Evaluate model with limited batches for efficiency
                eval_limit = limit_batches if limit_batches is not None else (5 if cfg.DEBUG else 10)
                metrics = evaluate_model(model, dataloader, eval_limit, 
                                       split=f"{mode}_temporal_{temp_noise}_h{horizon}_n{rot_noise:.2f}")
                
                # Store results
                results[temp_noise][horizon][rot_noise] = metrics
                
                # Log detailed metrics to wandb (matching standard test format)
                import wandb
                wandb.log({
                    f"{mode}_temporal_{temp_noise}/h{horizon}_n{rot_noise:.2f}_loss_pred": metrics['loss_pred'],
                    f"{mode}_temporal_{temp_noise}/h{horizon}_n{rot_noise:.2f}_re_pred": metrics['re_pred'],
                    f"{mode}_temporal_{temp_noise}/h{horizon}_n{rot_noise:.2f}_loss_recon": metrics['loss_recon'],
                    f"{mode}_temporal_{temp_noise}/h{horizon}_n{rot_noise:.2f}_total_loss": metrics['total_loss'],
                    f"{mode}_temporal_noise_level": temp_noise,
                    f"{mode}_rotational_noise_level": rot_noise,
                    f"{mode}_horizon": horizon
                })
                
                print(f"      Loss: {metrics['loss_pred']:.4f}, "
                      f"RE: {metrics['re_pred']:.2f}°")
    
    # Log summary statistics
    print("\n=== Temporal Noise Robustness Summary ===")
    for temp_noise in temporal_noise_levels:
        print(f"Temporal Noise {temp_noise}:")
        for horizon in horizons:
            avg_re = np.mean([results[temp_noise][horizon][rot_noise]['re_pred'] 
                             for rot_noise in rotational_noise_levels])
            avg_loss = np.mean([results[temp_noise][horizon][rot_noise]['loss_pred'] 
                               for rot_noise in rotational_noise_levels])
            print(f"  h{horizon}: Loss={avg_loss:.4f}, RE={avg_re:.2f}°")
    
    return results
