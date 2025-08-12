"""
Data handling for JAX/Flax training pipeline.
"""
import torch
import torch.multiprocessing as mp
import numpy as np
import gc
import weakref
from torch.utils.data import DataLoader, ConcatDataset
from experiments.so3_dataloader import SO3Dataloader

# Keep fork for normal training, but tests should use num_workers=0
# Fork multiprocessing can cause memory leaks in repeated test scenarios
try:
    mp.set_start_method('fork', force=True)
except RuntimeError:
    # Context already set, which is fine
    pass


def cleanup_dataset_arrays(dataset):
    """
    Helper function to explicitly deallocate arrays from dataset to prevent memory leaks.
    
    Args:
        dataset: SO3Dataloader or ConcatDataset instance
    """
    try:
        # Clean up SO3Dataloader arrays
        if hasattr(dataset, 'data') and dataset.data is not None:
            del dataset.data
            dataset.data = None
        if hasattr(dataset, 'omega') and dataset.omega is not None:
            del dataset.omega
            dataset.omega = None
        if hasattr(dataset, 'inputs') and dataset.inputs is not None:
            del dataset.inputs
            dataset.inputs = None
        if hasattr(dataset, 'recon') and dataset.recon is not None:
            del dataset.recon
            dataset.recon = None
        if hasattr(dataset, 'targets') and dataset.targets is not None:
            del dataset.targets
            dataset.targets = None
        if hasattr(dataset, 'moi') and dataset.moi is not None:
            del dataset.moi
            dataset.moi = None
        # Clean up irregular sampling arrays
        if hasattr(dataset, 'irregular_inputs') and dataset.irregular_inputs is not None:
            del dataset.irregular_inputs
            dataset.irregular_inputs = None
        if hasattr(dataset, 'irregular_targets') and dataset.irregular_targets is not None:
            del dataset.irregular_targets
            dataset.irregular_targets = None
        if hasattr(dataset, 'irregular_recon') and dataset.irregular_recon is not None:
            del dataset.irregular_recon
            dataset.irregular_recon = None
        if hasattr(dataset, 'irregular_omega') and dataset.irregular_omega is not None:
            del dataset.irregular_omega
            dataset.irregular_omega = None
        if hasattr(dataset, 'irregular_times') and dataset.irregular_times is not None:
            del dataset.irregular_times
            dataset.irregular_times = None
            
        # Handle ConcatDataset case
        if hasattr(dataset, 'datasets'):
            for subdataset in dataset.datasets:
                cleanup_dataset_arrays(subdataset)  # Recursive cleanup
                
    except Exception:
        # Ignore errors during cleanup
        pass


def cleanup_dataloader(dataloader):
    """
    Explicitly clean up dataloader and its dataset to prevent memory leaks.
    
    Args:
        dataloader: PyTorch DataLoader instance
    """
    try:
        if hasattr(dataloader, 'dataset'):
            cleanup_dataset_arrays(dataloader.dataset)
        # Force garbage collection
        gc.collect()
    except Exception:
        pass


def create_dataset(cfg, mode='train', noise_levels=None, horizon=None):
    """
    Create dataset with specified parameters.
    
    Args:
        cfg: Configuration node
        mode: Dataset mode ('train', 'val', or 'test')
        noise_levels: List of noise levels (if None, uses config defaults)
        horizon: Future horizon (if None, uses cfg.DATA.N_FUTURE)
        
    Returns:
        PyTorch dataset or ConcatDataset of multiple datasets
    """
    # Set defaults for noise levels from config if not provided
    if noise_levels is None:
        if mode == 'train':
            noise_levels = cfg.DATA.TRAIN_NOISE_LEVELS
        elif mode == 'val':
            noise_levels = cfg.DATA.VAL_NOISE_LEVELS
        else:  # test
            noise_levels = cfg.DATA.TEST_NOISE_LEVELS
    
    # Limit noise levels in debug mode
    if cfg.DEBUG:
        noise_levels = noise_levels[:2]
        
    print(f"Creating {mode} dataset with noise levels: {noise_levels}")
    
    # Use provided horizon or default from config
    n_future = horizon if horizon is not None else cfg.DATA.N_FUTURE
    
    # Common parameters for dataset creation
    dataset_kwargs = {
        'data_path': cfg.DATA.PATH,
        'mode': mode,
        'in_rep': cfg.DATA.IN_REP,
        'out_rep': cfg.DATA.OUT_REP,
        'dt': cfg.DATA.DT,
        'tf': cfg.DATA.TF,
        'n_prev': cfg.DATA.N_PREV,
        'n_future': n_future,
        'split': cfg.DATA.SPLIT,
        'temporal_noise_level': cfg.DATA.TRAIN_TEMPORAL_NOISE_LEVEL if cfg.DATA.TRAIN_TEMPORAL_NOISE_LEVEL > 0 else cfg.DATA.TEMPORAL_NOISE_LEVEL,
    }
    
    # Create datasets for each noise level
    print(f"Creating {len(noise_levels)} datasets for {mode} mode...")
    datasets = []
    for i, noise in enumerate(noise_levels):
        print(f"  Creating dataset {i+1}/{len(noise_levels)} with noise={noise}")
        dataset = SO3Dataloader(**dataset_kwargs, rotational_noise_level=noise)
        datasets.append(dataset)
        print(f"  Dataset {i+1} created with {len(dataset)} samples")
    
    # Return a single dataset or concatenated datasets
    if len(datasets) == 1:
        print(f"Returning single {mode} dataset")
        return datasets[0]
    else:
        print(f"Concatenating {len(datasets)} {mode} datasets...")
        concat_dataset = ConcatDataset(datasets)
        print(f"Final {mode} dataset has {len(concat_dataset)} total samples")
        return concat_dataset


def create_dataloaders(cfg):
    """
    Create train and validation dataloaders.
    
    Args:
        cfg: Configuration node
        
    Returns:
        train_dataloader, val_dataloader
    """
    # Create datasets
    print("=== Creating Training Dataset ===")
    train_dataset = create_dataset(cfg, mode='train')
    print("=== Creating Validation Dataset ===")
    val_dataset = create_dataset(cfg, mode='val')
    
    # Determine number of workers based on debug mode
    num_workers_train = 0 if cfg.DEBUG else cfg.DATA.NUM_WORKERS
    num_workers_val = 0 if cfg.DEBUG else 4  # Fixed for validation efficiency
    
    # Create train dataloader
    print(f"=== Creating Training DataLoader ===")
    print(f"Batch size: {cfg.DATA.BATCH_SIZE}, Workers: {num_workers_train}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.DATA.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,  # Enable memory pinning for GPU efficiency
        num_workers=num_workers_train,
        persistent_workers=num_workers_train > 0  # Enable persistent workers for spawn efficiency
    )
    
    # Create validation dataloader
    print(f"=== Creating Validation DataLoader ===")
    print(f"Batch size: {cfg.DATA.VAL_BATCH_SIZE}, Workers: {num_workers_val}")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.DATA.VAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=False,  # Original implementation didn't specify pin_memory for validation
        num_workers=num_workers_val,
        persistent_workers=num_workers_val > 0  # Enable persistent workers for spawn efficiency
    )
    
    print(f"Created dataloaders. Train: {len(train_dataloader)} batches, Val: {len(val_dataloader)} batches")
    
    return train_dataloader, val_dataloader


def create_test_dataloader(cfg, horizon, noise_level):
    """
    Create test dataloader for a specific horizon and noise level.
    
    Args:
        cfg: Configuration node
        horizon: Future horizon
        noise_level: Noise level
        
    Returns:
        PyTorch dataloader
    """
    # Create dataset for single noise level
    test_dataset = create_dataset(
        cfg, 
        mode='test', 
        noise_levels=[noise_level], 
        horizon=horizon
    )
    
    # Create dataloader with no multiprocessing to prevent memory leaks in testing
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.DATA.VAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=False,  # Disable for memory efficiency
        num_workers=0,     # Force single-threaded to prevent multiprocessing memory leaks
        persistent_workers=False  # Disable persistent workers
    )
    
    return test_dataloader