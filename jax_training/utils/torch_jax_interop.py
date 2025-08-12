#!/usr/bin/env python
"""
Efficient PyTorch <-> JAX interoperability utilities that preserve GPU memory.

These utilities avoid unnecessary CPU transfers when converting between
PyTorch tensors and JAX arrays on GPU.
"""
import torch
import jax
import jax.numpy as jnp
from jax.dlpack import from_dlpack as jax_from_dlpack
from jax.dlpack import to_dlpack as jax_to_dlpack


def torch_to_jax(torch_tensor: torch.Tensor, copy: bool = False) -> jnp.ndarray:
    """
    Convert PyTorch tensor to JAX array while preserving GPU memory.
    
    Args:
        torch_tensor: PyTorch tensor (CPU or GPU)
        copy: If True, force a copy. If False, use zero-copy when possible.
        
    Returns:
        JAX array on the same device
    """
    if copy:
        # Force a copy through CPU (slower but safer)
        return jnp.array(torch_tensor.detach().cpu().numpy())
    else:
        # Zero-copy conversion using DLPack (preserves GPU memory)
        # Ensure tensor is contiguous for DLPack compatibility
        tensor = torch_tensor.detach()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return jax_from_dlpack(tensor)


def jax_to_torch(jax_array: jnp.ndarray, copy: bool = False) -> torch.Tensor:
    """
    Convert JAX array to PyTorch tensor while preserving GPU memory.
    
    Args:
        jax_array: JAX array (CPU or GPU)
        copy: If True, force a copy. If False, use zero-copy when possible.
        
    Returns:
        PyTorch tensor on the same device
    """
    if copy:
        # Force a copy through CPU (slower but safer)
        return torch.from_numpy(jax_array.__array__()).float()
    else:
        # Zero-copy conversion using DLPack (preserves GPU memory)
        return torch.from_dlpack(jax_to_dlpack(jax_array))


def torch_batch_to_jax(torch_tensors: list[torch.Tensor], copy: bool = False) -> list[jnp.ndarray]:
    """
    Convert a batch of PyTorch tensors to JAX arrays efficiently.
    
    Args:
        torch_tensors: List of PyTorch tensors
        copy: If True, force copies. If False, use zero-copy when possible.
        
    Returns:
        List of JAX arrays
    """
    return [torch_to_jax(tensor, copy=copy) for tensor in torch_tensors]


def jax_batch_to_torch(jax_arrays: list[jnp.ndarray], copy: bool = False) -> list[torch.Tensor]:
    """
    Convert a batch of JAX arrays to PyTorch tensors efficiently.
    
    Args:
        jax_arrays: List of JAX arrays
        copy: If True, force copies. If False, use zero-copy when possible.
        
    Returns:
        List of PyTorch tensors
    """
    return [jax_to_torch(array, copy=copy) for array in jax_arrays]


# Convenience aliases
torch2jax = torch_to_jax
jax2torch = jax_to_torch