import argparse
import os
from pathlib import Path

# Fix PyTorch CUDA allocator issue
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ''

# simulation libs
import roma
import torch
from torchdiffeq import odeint
import numpy as np
import jax.random
import random

# custom imports
from simulation.config import get_cfg_defaults
from simulation.data import merge_scenarios

from simulation.so3_dynamics import RigidBody, dynamics
from simulation.so3_scenarios import create_scenarios


def seed_everything(seed: int):
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # JAX uses explicit PRNG keys, so we just create one for consistency
    jax.random.PRNGKey(seed)


def generate_dataset(cfg, scenario_name: str, output_dir: Path):
    """
    Generate dataset with 4 different MOI distributions that align with splits
    
    Args:
        cfg: YACS configuration node
        scenario_name: Name of the scenario to generate
    """
    # Set device for hardware acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Calculate samples per distribution - ensure divisible by 4
    batch_size = cfg.SIM.N_SAMPLES
    samples_per_distribution = batch_size // 4
    total_samples = samples_per_distribution * 4  # Ensure total is exactly divisible by 4
    
    # Initialize storage for combined results
    all_q0 = []
    all_omega0 = []
    all_moi_matrices = []
    
    # Generate samples for each distribution
    for dist_idx, dist in enumerate(cfg.BODY.MOI_DISTRIBUTIONS):
        # Set seed for reproducibility while having different distributions
        torch.manual_seed(cfg.SIM.SEED + dist_idx)
        
        # Generate random quaternions for this distribution
        q0_dist = roma.random_unitquat((samples_per_distribution,)).to(device)
        
        # Generate angular velocities with minimum norm constraint
        min_norm = cfg.SIM.MIN_OMEGA_NORM
        omega0_dist = torch.randn(samples_per_distribution, 3, device=device) * cfg.SIM.OMEGA_SCALE
        norms = torch.norm(omega0_dist, dim=1)
        mask = norms < min_norm
        
        # Resample small vectors
        while mask.any():
            print(f"Resampling omega0 for distribution {dist_idx+1}...")
            omega0_dist[mask] = torch.randn(mask.sum(), 3, device=device) * cfg.SIM.OMEGA_SCALE
            norms = torch.norm(omega0_dist, dim=1)
            mask = norms < min_norm
        
        # Create moment of inertia for this distribution
        base_moi = torch.tensor(dist["BASE"], device=device)
        moment_of_inertia = base_moi.unsqueeze(0).expand(samples_per_distribution, -1)
        
        # Apply permutation to create variety while maintaining distribution characteristics
        perm_indices = torch.stack([torch.randperm(3, device=device) for _ in range(samples_per_distribution)])
        moment_of_inertia = torch.gather(moment_of_inertia, 1, perm_indices)
        
        # Add distribution-specific noise
        noise = torch.randn_like(moment_of_inertia) * dist["NOISE"] * torch.abs(moment_of_inertia)
        moment_of_inertia = moment_of_inertia + noise
        
        # Ensure positive values (physically valid)
        moment_of_inertia = torch.abs(moment_of_inertia)
        
        # Convert to diagonal matrices
        moi_matrices = torch.diag_embed(moment_of_inertia)
        
        # Store results from this distribution
        all_q0.append(q0_dist)
        all_omega0.append(omega0_dist)
        all_moi_matrices.append(moi_matrices)
        
        print(f"Generated distribution {dist_idx+1}: base={dist['BASE']}, noise={dist['NOISE']}")
    
    # Combine all distributions in order (crucial for alignment with splits)
    q0 = torch.cat(all_q0, dim=0)
    omega0 = torch.cat(all_omega0, dim=0)
    moment_of_inertia = torch.cat(all_moi_matrices, dim=0)
    
    # Create rigid body with combined data
    rigid_body = RigidBody(moment_of_inertia, q0, omega0)
    
    # Time span for simulation
    t_span = torch.linspace(0, cfg.SIM.T_FINAL, int(cfg.SIM.T_FINAL / cfg.SIM.DT), device=device)
    
    # Add scenario parameters
    scenario_params = {
        'initial_state': torch.cat([q0, omega0], dim=-1),
        'moi': moment_of_inertia
    }
    
    # Add all scenario configurations
    for name in cfg.SCENARIOS:
        scenario_params[name] = dict(getattr(cfg.SCENARIOS, name))
    
    scenarios = create_scenarios(scenario_params)
    scenario = scenarios[scenario_name]
    
    # Simulate dynamics for all samples
    print(f"Integrating dynamics for {scenario_name} scenario with 4 MOI distributions...")
    solution = odeint(
        lambda t, y: dynamics(t, y, rigid_body, scenario.torque_fn),
        rigid_body.state,
        t_span,
        method='dopri5'
    )
    
    # Rearrange to [B, T, 7]
    solution = solution.transpose(0, 1)
    
    # Save dataset with distribution metadata
    save_dict = {
        'quat': solution[..., :4],
        'omega': solution[..., 4:],
        'dt': cfg.SIM.DT,
        't_f': cfg.SIM.T_FINAL,
        'moi': torch.cat(all_moi_matrices),
        'distribution_indices': torch.arange(4, device=device).repeat_interleave(samples_per_distribution),
        'body_params': cfg.BODY,
        'scenario': scenario_name,
        'scenario_description': scenario.description,
        'scenario_params': scenario_params,
        'omega_scale': cfg.SIM.OMEGA_SCALE,
    }
    
    if cfg.SAVE_DERIVATIVES:
        # Calculate derivatives at timepoints
        derivatives = torch.stack([
            dynamics(t, y, rigid_body, scenario.torque_fn)
            for t, y in zip(t_span, solution.transpose(0,1))
        ])  # [T, B, 7]
        
        # Split derivatives into q_dot and w_dot
        q_dots = derivatives[..., :4].transpose(0, 1)  # [B, T, 4]
        w_dots = derivatives[..., 4:].transpose(0, 1)  # [B, T, 3]
        
        save_dict.update(**{
            'q_dot': q_dots,
            'w_dot': w_dots,
        })
    
    # Move tensors to CPU before saving
    for key, val in save_dict.items():
        if isinstance(val, torch.Tensor):
            save_dict[key] = val.cpu()
    
    path = output_dir / f"rigid_body_{scenario_name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to {path}")
    torch.save(save_dict, path)
    return save_dict


def main():
    args = parse_args()
    
    # Load default config
    cfg = get_cfg_defaults()
    
    # Load config file if specified
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    
    # Override config with command-line options
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    # Create output directory
    output_dir = Path(cfg.DATA_ROOT) / cfg.SIM.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate datasets for enabled scenarios
    scenario_count = 0
    all_saved = []
    for scenario_name in cfg.SCENARIOS:
        if getattr(cfg.SCENARIOS, scenario_name).ENABLED:
            scenario_count += 1
            print(f"Generating dataset for {scenario_name} scenario...")
            # seed before generating each time
            seed_everything(cfg.SIM.SEED)
            saved = generate_dataset(cfg, scenario_name, output_dir)
            all_saved.append(saved)

    print(f"simulated {scenario_count} datasets")
    if len(all_saved) > 1:
        print("Merging all scenarios...")
        merged = merge_scenarios(all_saved, cfg.SIM.N_SAMPLES)
        merge_path = output_dir / "rigid_body_merged.pt"
        torch.save(merged, merge_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Rigid Body Trajectory Generation")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                       help="Modify config options using the command-line")
    return parser.parse_args()

if __name__ == "__main__":
    main()
