"""
Helper functions to enable efficient GPU transfers between PyTorch and JAX.
"""
import jax
import jax.numpy as jnp
import torch
import numpy as np


def check_gpu_compatibility():
    """
    Check if both PyTorch and JAX can access GPUs and are compatible.
    
    Returns:
        dict: Information about GPU compatibility between PyTorch and JAX
    """
    info = {
        "pytorch_cuda_available": torch.cuda.is_available(),
        "pytorch_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "jax_devices": str(jax.devices()),
        "jax_gpu_devices": len([d for d in jax.devices() if d.platform == 'gpu']),
        "dlpack_support": hasattr(torch.utils, 'dlpack') and hasattr(jax, 'dlpack'),
        "compatible": False
    }
    
    # Check if both frameworks can access GPUs
    if info["pytorch_cuda_available"] and info["jax_gpu_devices"] > 0:
        info["compatible"] = True
    
    return info


def pytorch_to_jax_dlpack(torch_tensor):
    """
    Convert a PyTorch tensor to a JAX array using DLPack for zero-copy GPU transfer.
    
    Args:
        torch_tensor: PyTorch tensor on GPU
        
    Returns:
        JAX array on the same GPU device
    """
    if not torch_tensor.is_cuda:
        # If tensor is on CPU, move it to GPU first
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch_tensor = torch_tensor.to(device)
    
    # Convert using DLPack
    dlpack = torch.utils.dlpack.to_dlpack(torch_tensor)
    return jax.dlpack.from_dlpack(dlpack)


def jax_to_pytorch_dlpack(jax_array):
    """
    Convert a JAX array to a PyTorch tensor using DLPack for zero-copy GPU transfer.
    
    Args:
        jax_array: JAX array on GPU
        
    Returns:
        PyTorch tensor on the same GPU device
    """
    # Check if the JAX array is on GPU
    jax_devices = jax.devices()
    gpu_devices = [d for d in jax_devices if d.platform == 'gpu']
    
    if not gpu_devices:
        # If no GPU devices, convert through numpy
        return torch.from_numpy(np.asarray(jax_array))
    
    # Convert using DLPack
    dlpack = jax.dlpack.to_dlpack(jax_array)
    return torch.utils.dlpack.from_dlpack(dlpack)


def safe_pytorch_to_jax(torch_tensor):
    """
    Convert PyTorch tensor to JAX array, falling back to numpy if DLPack fails.
    
    Args:
        torch_tensor: PyTorch tensor
        
    Returns:
        JAX array
    """
    if torch_tensor.is_cuda and hasattr(torch.utils, 'dlpack') and hasattr(jax, 'dlpack'):
        try:
            # Try DLPack first for zero-copy transfer
            return pytorch_to_jax_dlpack(torch_tensor)
        except (RuntimeError, ValueError, TypeError) as e:
            print(f"DLPack conversion failed: {e}. Falling back to numpy bridge.")
    
    # Fall back to numpy bridge
    return jnp.array(torch_tensor.detach().cpu().numpy())


def verify_memory_sharing(torch_tensor, jax_array):
    """
    Verify if PyTorch tensor and JAX array share the same memory.
    
    Args:
        torch_tensor: PyTorch tensor
        jax_array: JAX array
        
    Returns:
        bool: True if memory is shared
    """
    # Only works on GPU tensors
    if not torch_tensor.is_cuda:
        return False
    
    try:
        # Get memory address for JAX array
        jax_ptr = jax_array.unsafe_buffer_pointer()
        
        # For PyTorch, we can use the data_ptr() method
        torch_ptr = torch_tensor.data_ptr()
        
        # Compare the addresses
        return jax_ptr == torch_ptr
    except (AttributeError, RuntimeError):
        # If unsafe_buffer_pointer is not available or fails
        return False


def convert_batch_to_jax_gpu(batch):
    """
    Efficiently convert a batch of PyTorch tensors to JAX arrays using DLPack when possible.
    
    Args:
        batch: Tuple of PyTorch tensors from dataloader
        
    Returns:
        Tuple of JAX arrays
    """
    (t_recon, t_fut, x), targets, recon, omega, moi = batch
    
    # Convert all tensors to JAX with GPU optimization
    t_recon_jax = safe_pytorch_to_jax(t_recon)
    t_fut_jax = safe_pytorch_to_jax(t_fut)
    x_jax = safe_pytorch_to_jax(x)
    targets_jax = safe_pytorch_to_jax(targets)
    recon_jax = safe_pytorch_to_jax(recon)
    omega_jax = safe_pytorch_to_jax(omega)
    moi_jax = safe_pytorch_to_jax(moi)
    
    return t_recon_jax, t_fut_jax, x_jax, targets_jax, recon_jax, omega_jax, moi_jax


# Example usage
if __name__ == "__main__":
    # Check compatibility
    compatibility = check_gpu_compatibility()
    print("GPU Compatibility:", compatibility)
    
    if compatibility["compatible"]:
        # Create sample PyTorch tensor on GPU
        torch_tensor = torch.randn(1000, 1000, device="cuda")
        
        # Time the conversion using DLPack
        import time
        start = time.time()
        jax_array = pytorch_to_jax_dlpack(torch_tensor)
        dlpack_time = time.time() - start
        
        # Time the conversion using numpy bridge
        start = time.time()
        jax_array_np = jnp.array(torch_tensor.detach().cpu().numpy())
        numpy_time = time.time() - start
        
        print(f"DLPack conversion time: {dlpack_time:.6f} seconds")
        print(f"NumPy bridge conversion time: {numpy_time:.6f} seconds")
        print(f"Speedup: {numpy_time/dlpack_time:.2f}x")
        
        # Verify memory sharing
        is_shared = verify_memory_sharing(torch_tensor, jax_array)
        print(f"Memory is shared: {is_shared}")
