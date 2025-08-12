
"""Model factory and utilities for JAX training pipeline."""

import jax
from flax import nnx

# Import model creators for easier access
from .GRU import create_gru_model, GRUBaseline
from .SO3NeuralCDE import create_so3_neural_cde, SO3NeuralCDE

__all__ = [
    'create_model_from_cfg',
    'create_gru_model', 'GRUBaseline',
    'create_so3_neural_cde', 'SO3NeuralCDE',
    'get_model_save_path', 'get_best_checkpoint_path'
]


def create_model_from_cfg(cfg):
    """Create a model from the configuration."""
    model_type = cfg.MODEL.TYPE.lower()
    
    if model_type == "gru":
        from models.GRU import create_gru_model
        
        key = jax.random.key(cfg.TRAIN.SEED)
        rngs = nnx.Rngs(params=key)
        
        model_args = {
            'input_channel': cfg.MODEL.INPUT_CHANNEL,
            'latent_channels': cfg.MODEL.LATENT_DIM,
            'hidden_channels': cfg.MODEL.MLP_WIDTH,
            'output_channel': cfg.MODEL.OUTPUT_CHANNEL,
            'num_gru_layers': cfg.MODEL.NUM_LAYERS,
            'use_time_conditioning': cfg.MODEL.USE_TIME_CONDITIONING
        }
        
        print(f"Initializing GRU model with arguments: {model_args}")
        
        model = create_gru_model(
            input_channel=cfg.MODEL.INPUT_CHANNEL,
            latent_channels=cfg.MODEL.LATENT_DIM,
            hidden_channels=cfg.MODEL.MLP_WIDTH,
            output_channel=cfg.MODEL.OUTPUT_CHANNEL,
            num_gru_layers=cfg.MODEL.NUM_LAYERS,
            use_time_conditioning=cfg.MODEL.USE_TIME_CONDITIONING,
            rngs=rngs
        )
        return model
    
    elif model_type == "so3neuralcde":
        from models.SO3NeuralCDE import create_so3_neural_cde
        
        key = jax.random.key(cfg.TRAIN.SEED)
        rngs = nnx.Rngs(params=key)
        
        model_args = {
            'input_channel': cfg.MODEL.INPUT_CHANNEL,
            'latent_channels': cfg.MODEL.LATENT_DIM,
            'hidden_channels': cfg.MODEL.MLP_WIDTH,
            'output_channel': cfg.MODEL.OUTPUT_CHANNEL,
            'interpolation_method': cfg.MODEL.INTERPOLATION_METHOD,
            'method': cfg.MODEL.ODE_METHOD,
            'atol': cfg.MODEL.ATOL,
            'rtol': cfg.MODEL.RTOL,
            'max_steps': cfg.MODEL.MAX_STEPS,
            'sg_learnable_weights': cfg.MODEL.SG_LEARNABLE_WEIGHTS,
            'sg_polynomial_order': cfg.MODEL.SG_POLYNOMIAL_ORDER,
            'second_order': cfg.MODEL.SECOND_ORDER,
            'use_refit': cfg.MODEL.USE_REFIT
        }
        
        print(f"Initializing SO3NeuralCDE model with arguments: {model_args}")
        
        model = create_so3_neural_cde(
            input_channel=cfg.MODEL.INPUT_CHANNEL,
            latent_channels=cfg.MODEL.LATENT_DIM,
            hidden_channels=cfg.MODEL.MLP_WIDTH,
            output_channel=cfg.MODEL.OUTPUT_CHANNEL,
            interpolation_method=cfg.MODEL.INTERPOLATION_METHOD,
            method=cfg.MODEL.ODE_METHOD,
            atol=cfg.MODEL.ATOL,
            rtol=cfg.MODEL.RTOL,
            max_steps=cfg.MODEL.MAX_STEPS,
            sg_learnable_weights=cfg.MODEL.SG_LEARNABLE_WEIGHTS,
            sg_polynomial_order=cfg.MODEL.SG_POLYNOMIAL_ORDER,
            second_order=cfg.MODEL.SECOND_ORDER,
            use_refit=cfg.MODEL.USE_REFIT,
            rngs=rngs
        )
        return model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_save_path(cfg, epoch=None, metric_value=None, is_best=False):
    """Get the path to save the model using wandb run ID for consistent checkpointing."""
    import os
    import wandb
    
    if wandb.run is None:
        raise RuntimeError("No active wandb run. Model saving requires wandb to be initialized.")
    
    # Create directory structure: ./outputs/{project}/{run_id}/checkpoints/
    # Use wandb project name as the base directory
    project_dir = os.path.join("./outputs", cfg.OUTPUT.WANDB_PROJECT)
    run_dir = os.path.join(project_dir, wandb.run.id, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create filename following standard checkpoint pattern
    if is_best and epoch is not None and metric_value is not None:
        filename = f"best-checkpoint-epoch={epoch:02d}-re_pred={metric_value:.2f}.pkl"
    elif epoch is not None:
        filename = f"last-checkpoint-epoch={epoch:02d}.pkl"
    else:
        # Fallback for final model
        filename = "final-model.pkl"
    
    return os.path.join(run_dir, filename)


def get_best_checkpoint_path(run_id, project_name="rot_dyn_jax", outputs_dir="./outputs"):
    """
    Get the path to the best checkpoint for a given wandb run ID.
    
    Args:
        run_id: WandB run ID (the unique identifier)
        project_name: Project name (default: "rot_dyn_jax")
        outputs_dir: Base outputs directory
        
    Returns:
        Path to the best checkpoint file
    """
    import os
    from pathlib import Path
    
    # Construct expected directory path
    run_dir = Path(outputs_dir) / project_name / run_id / "checkpoints"
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Find best checkpoint file
    checkpoint_files = list(run_dir.glob("best-checkpoint*.pkl"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No best checkpoint file found in {run_dir}")
    
    return str(checkpoint_files[0])
