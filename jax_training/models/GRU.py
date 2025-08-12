import jax
import jax.numpy as jnp
from flax import nnx
from typing import Callable, Optional, Tuple, Any, List

from jax_training.utils.so3 import symmetric_orthogonalization, gramschmidt_to_rotmat


class CustomGRUCell(nnx.Module):
    """Custom implementation of a GRU cell."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        rngs: nnx.Rngs
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        
        # Input-to-hidden weights
        self.W_ir = nnx.Linear(in_features, hidden_features, rngs=rngs)
        self.W_iz = nnx.Linear(in_features, hidden_features, rngs=rngs)
        self.W_in = nnx.Linear(in_features, hidden_features, rngs=rngs)
        
        # Hidden-to-hidden weights
        self.W_hr = nnx.Linear(hidden_features, hidden_features, rngs=rngs)
        self.W_hz = nnx.Linear(hidden_features, hidden_features, rngs=rngs)
        self.W_hn = nnx.Linear(hidden_features, hidden_features, rngs=rngs)
    
    def __call__(self, h, x):
        """Process a single step through the GRU cell.
        
        Args:
            h: Hidden state of shape (batch_size, hidden_features)
            x: Input of shape (batch_size, in_features)
            
        Returns:
            Updated hidden state
        """
        # Reset gate
        r = jax.nn.sigmoid(self.W_ir(x) + self.W_hr(h))
        
        # Update gate
        z = jax.nn.sigmoid(self.W_iz(x) + self.W_hz(h))
        
        # Candidate
        n = jax.nn.tanh(self.W_in(x) + r * self.W_hn(h))
        
        # New state
        h_new = (1.0 - z) * n + z * h
        
        return h_new


class MultiLayerGRU(nnx.Module):
    """A multi-layer GRU implementation."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_layers: int = 3,
        use_time_conditioning: bool = True,
        *,
        rngs: nnx.Rngs
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.use_time_conditioning = use_time_conditioning
        
        # Create GRU cells for each layer
        self.gru_cells = []
        for i in range(num_layers):
            # First layer accounts for time conditioning if enabled
            if i == 0 and use_time_conditioning:
                layer_in_features = in_features + 1  # +1 for timestamp
            else:
                layer_in_features = in_features if i == 0 else hidden_features
            cell = CustomGRUCell(
                in_features=layer_in_features,
                hidden_features=hidden_features,
                rngs=rngs
            )
            self.gru_cells.append(cell)
    
    def init_hidden(self, batch_size):
        """Initialize hidden states for all layers."""
        return [jnp.zeros((batch_size, self.hidden_features)) 
                for _ in range(self.num_layers)]
    
    def __call__(self, x, hidden_states=None, timestamps=None):
        """Process a sequence through all GRU layers.
        
        Args:
            x: Input sequence of shape (batch_size, seq_len, in_features)
            hidden_states: Optional list of initial hidden states
            timestamps: Optional timestamps for each timestep (batch_size, seq_len)
            
        Returns:
            hidden_states: Updated hidden states for each layer
            outputs: Output from the last layer (all time steps)
        """
        # Handle 4D input (batch_size, seq_len, 3, 3) -> flatten to (batch_size, seq_len, 9)
        # Flatten 3x3 rotation matrices to 9D vectors
        if len(x.shape) == 4:
            batch_size, seq_len, _, _ = x.shape
            x = x.reshape(batch_size, seq_len, -1)  # Flatten 3x3 matrices to 9D vectors
        else:
            batch_size, seq_len, _ = x.shape
        
        # Add time conditioning to input if enabled and timestamps provided
        if self.use_time_conditioning and timestamps is not None:
            # Concatenate timestamps to input: (batch_size, seq_len, in_features + 1)
            x = jnp.concatenate([jnp.expand_dims(timestamps, -1), x], axis=-1)
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = self.init_hidden(batch_size)
        
        # Sanity check on hidden states
        assert len(hidden_states) == self.num_layers, \
            f"Expected {self.num_layers} hidden states, got {len(hidden_states)}"
        
        # If sequence length is 1, process single time step more efficiently
        if seq_len == 1:
            x_t = x[:, 0]
            
            # Process through each layer
            new_hidden_states = []
            for i, (cell, h) in enumerate(zip(self.gru_cells, hidden_states)):
                layer_input = x_t if i == 0 else layer_input
                h_new = cell(h, layer_input)
                new_hidden_states.append(h_new)
                layer_input = h_new
            
            # Return updated hidden states and last layer output
            return new_hidden_states, jnp.expand_dims(new_hidden_states[-1], axis=1)
        
        # Full sequence processing
        all_outputs = []
        
        # For each time step
        for t in range(seq_len):
            x_t = x[:, t]
            
            # For each layer
            new_hidden_states = []
            for i, (cell, h) in enumerate(zip(self.gru_cells, hidden_states)):
                layer_input = x_t if i == 0 else layer_input
                h_new = cell(h, layer_input)
                new_hidden_states.append(h_new)
                layer_input = h_new
            
            # Update hidden states
            hidden_states = new_hidden_states
            
            # Collect output from the last layer
            all_outputs.append(hidden_states[-1])
        
        # Stack outputs along time dimension
        outputs = jnp.stack(all_outputs, axis=1)
        
        return hidden_states, outputs


class GRUBaseline(nnx.Module):
    """GRU baseline implementation using JAX and Flax NNX."""
    
    def __init__(
        self, 
        input_channel: int,
        latent_channels: int,
        hidden_channels: int,
        output_channel: int,
        num_gru_layers: int = 3,
        method: str = 'euler',
        use_time_conditioning: bool = True,
        *, 
        rngs: nnx.Rngs
    ):
        self.input_channel = input_channel
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.output_channel = output_channel
        self.method = method
        self.num_gru_layers = num_gru_layers
        
        # Multi-layer GRU with configurable time conditioning
        self.gru = MultiLayerGRU(
            in_features=input_channel,
            hidden_features=latent_channels,
            num_layers=num_gru_layers,
            use_time_conditioning=use_time_conditioning,
            rngs=rngs
        )
        
        # Output network
        self.lin1 = nnx.Linear(latent_channels, latent_channels, rngs=rngs)
        self.lin2 = nnx.Linear(latent_channels, output_channel, rngs=rngs)
        
        # Projection for autoregression
        if use_time_conditioning:
            # When time conditioning is enabled, project 9D to 10D for prediction steps
            self.projection = nnx.Linear(input_channel, input_channel + 1, rngs=rngs)
        else:
            # No time conditioning, keep same dimensions
            self.projection = nnx.Linear(input_channel, input_channel, rngs=rngs)
        
        # Set the appropriate post-processing function
        if output_channel == 9:
            # Use vmap for batched symmetric orthogonalization
            self.postprocess_fn = jax.vmap(symmetric_orthogonalization)
        elif output_channel == 6:
            self.postprocess_fn = gramschmidt_to_rotmat
        elif output_channel == 4 or output_channel == 3:
            self.postprocess_fn = lambda x: x
        else:
            self.postprocess_fn = lambda x: x
    
    def output_net(self, x):
        """Forward pass through the output network."""
        x = self.lin1(x)
        x = jax.nn.elu(x)
        x = self.lin2(x)
        return x
    
    def postprocess(self, x):
        """Apply post-processing to the output and flatten for autoregression.
        
        For 6D internal representation: applies GSO to convert to rotation matrix, then flattens to 9D.
        For 9D internal representation: applies SO(3) projection, then flattens to 9D.
        Always returns 9D flattened vectors for consistent autoregression.
        """
        processed = self.postprocess_fn(x)
        
        # The post-processing functions should always return (3, 3) rotation matrices
        # Flatten rotation matrices to 9D vectors for autoregression 
        # Flatten rotation matrices to 9D vectors for autoregression
        return processed.reshape(processed.shape[:-2] + (-1,))
    
    def __call__(
        self, 
        t_recon: jnp.ndarray, 
        t_fut: jnp.ndarray, 
        x: jnp.ndarray, 
        *args,
        return_solver_stats: bool = False,
        **kwargs
    ) -> Tuple[None, jnp.ndarray, None]:
        """Forward pass for the Neural CDE model.
        
        Args:
            t_recon: Time points for reconstruction
            t_fut: Future time points
            x: Input sequence
            *args: Additional arguments (e.g., omega, moi)
            return_solver_stats: Whether to return solver statistics (ignored for GRU)
            **kwargs: Additional keyword arguments (ignored)
            
        Returns:
            Tuple of (None, predictions, None) - consistent interface with SO3NeuralCDE
        """
        # Process input sequence through GRU with timestamp conditioning
        # Pass reconstruction timestamps (t_recon) to condition the GRU during reconstruction
        # t_recon already has shape (batch_size, seq_len), no need to expand dims
        timestamps_normalized = t_recon / 10.0  # Undo the *10 scaling from dataloader
        hidden_states, z = self.gru(x, timestamps=timestamps_normalized)
        
        # Get the final output (last time step)
        z_final = z[:, -1]
        
        # Generate first prediction
        first_output = self.output_net(z_final)
        first_pred = self.postprocess(first_output)
        
        # Store all predictions
        preds_list = [first_pred]
        
        # Auto-regressive prediction loop
        for _ in range(t_fut.shape[1] - 1):
            # Create input for next step from previous prediction
            prev_pred = preds_list[-1]
            input_pred = self.projection(prev_pred)
            
            # Add batch dimension for sequence processing (batch_size, 1, input_dim)
            input_pred = jnp.expand_dims(input_pred, axis=1)
            
            # Process single step through GRU (no timestamps provided, so no time conditioning)
            hidden_states, z_new = self.gru(input_pred, hidden_states)
            
            # Generate next prediction
            next_output = self.output_net(z_new[:, 0])
            next_pred = self.postprocess(next_output)
            
            # Add to predictions
            preds_list.append(next_pred)
        
        # Stack all predictions along time dimension
        pred = jnp.stack(preds_list, axis=1)  # (batch, time, 9)
        
        # Always output 9D rotation matrices regardless of internal representation
        # pred is (batch, time, 9) from flattened rotation matrices
        # Reshape to rotation matrices for output format
        pred = pred.reshape(pred.shape[0], pred.shape[1], 3, 3)  # (batch, time, 3, 3)
        
        return None, pred, None


def create_gru_model(
    input_channel: int,
    latent_channels: int,
    hidden_channels: int,
    output_channel: int,
    num_gru_layers: int = 3,
    method: str = 'euler',
    use_time_conditioning: bool = True,
    *,
    rngs: nnx.Rngs
):
    """Create a GRU baseline model."""
    return GRUBaseline(
        input_channel=input_channel,
        latent_channels=latent_channels,
        hidden_channels=hidden_channels,
        output_channel=output_channel,
        num_gru_layers=num_gru_layers,
        method=method,
        use_time_conditioning=use_time_conditioning,
        rngs=rngs
    )


# Example usage:
if __name__ == "__main__":
    import time
    
    # Model parameters
    input_channel = 10
    latent_channels = 32
    hidden_channels = 32
    output_channel = 9
    num_gru_layers = 3  # Configurable number of GRU layers
    
    # Initialize random number generator for reproducibility
    key = jax.random.key(0)
    rngs = nnx.Rngs(params=key)
    
    # Create model
    model = create_gru_model(
        input_channel=input_channel,
        latent_channels=latent_channels,
        hidden_channels=hidden_channels,
        output_channel=output_channel,
        num_gru_layers=num_gru_layers,
        rngs=rngs
    )
    
    print("Model initialized successfully!")
    
    # Create a JIT-compiled forward function
    @nnx.jit
    def forward(model, t_recon, t_fut, x):
        """JIT-compiled forward function using nnx.jit."""
        return model(t_recon, t_fut, x)
    
    # Create test data
    batch_size = 10000
    seq_len = 12
    t_recon = jnp.zeros((batch_size, seq_len))
    t_fut = jnp.zeros((batch_size, seq_len))
    x = jnp.ones((batch_size, seq_len, input_channel))
    
    # Benchmark
    try:
        print("Running first forward pass (compiling)...")
        # Run once to compile
        start_time = time.time()
        _, pred = forward(model, t_recon, t_fut, x)
        compilation_time = time.time() - start_time
        print(f"Compilation time: {compilation_time:.6f} seconds")
        print(f"First prediction shape: {pred.shape}")
        
        # Benchmark
        num_iters = 100
        print(f"Running benchmark with {num_iters} iterations...")
        start_time = time.time()
        
        for _ in range(num_iters):
            _, _ = forward(model, t_recon, t_fut, x)
        
        # Compute average time
        average_time = (time.time() - start_time) / num_iters
        print(f"Average time per iteration: {average_time:.6f} seconds")
        
    except Exception as e:
        print(f"Error in benchmark: {e}")
        import traceback
        traceback.print_exc()
