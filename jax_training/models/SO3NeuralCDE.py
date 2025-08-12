import jax
import jax.numpy as jnp
from flax import nnx
import diffrax
from typing import Callable, Optional, Tuple, Any, List, Union

from jax_training.utils.so3 import symmetric_orthogonalization, gramschmidt_to_rotmat, rodrigues
from jax_training.utils.savitzky_golay_so3 import so3_savitzky_golay_filter, SO3PolynomialPath


class SO3FirstDerivativePath(diffrax.AbstractPath):
    """Control path for first derivatives from SO3PolynomialPath for MultiTerm integration."""
    so3_path: SO3PolynomialPath
    
    def __init__(self, so3_path):
        self.so3_path = so3_path
    
    @property
    def t0(self):
        return self.so3_path.t0
    
    @property
    def t1(self):
        return self.so3_path.t1
    
    def evaluate(self, t0, t1=None, left=True):
        """Evaluate control for ControlTerm integration."""
        if t1 is None:
            # Point evaluation - return first derivative at t0
            # OPTIMIZED: Use dedicated first_derivative method
            first_deriv = self.so3_path.first_derivative(t0)
            return first_deriv[1:]  # Skip time channel
        else:
            # Increment evaluation - integrate first derivative from t0 to t1
            # Use midpoint rule for better accuracy
            t_mid = 0.5 * (t0 + t1)
            # OPTIMIZED: Use dedicated first_derivative method
            first_deriv_mid = self.so3_path.first_derivative(t_mid)
            return first_deriv_mid[1:] * (t1 - t0)
    
    def derivative(self, t, left=True):
        """Return first derivative at time t."""
        # OPTIMIZED: Use dedicated first_derivative method
        first_deriv = self.so3_path.first_derivative(t)
        return first_deriv[1:]


class SO3SecondDerivativePath(diffrax.AbstractPath):
    """Control path for second derivatives from SO3PolynomialPath for MultiTerm integration."""
    so3_path: SO3PolynomialPath
    
    def __init__(self, so3_path):
        self.so3_path = so3_path
    
    @property
    def t0(self):
        return self.so3_path.t0
    
    @property
    def t1(self):
        return self.so3_path.t1
    
    def evaluate(self, t0, t1=None, left=True):
        """Evaluate control for ControlTerm integration."""
        if t1 is None:
            # Point evaluation - return second derivative at t0
            # OPTIMIZED: Use dedicated second_derivative method
            second_deriv = self.so3_path.second_derivative(t0)
            return second_deriv[1:]  # Skip time channel
        else:
            # Increment evaluation - integrate second derivative from t0 to t1
            # Use midpoint rule for better accuracy
            t_mid = 0.5 * (t0 + t1)
            # OPTIMIZED: Use dedicated second_derivative method
            second_deriv_mid = self.so3_path.second_derivative(t_mid)
            return second_deriv_mid[1:] * (t1 - t0)
    
    def derivative(self, t, left=True):
        """Return second derivative at time t."""
        # OPTIMIZED: Use dedicated second_derivative method
        second_deriv = self.so3_path.second_derivative(t)
        return second_deriv[1:]


class CDEFunc(nnx.Module):
    """Neural vector field for the Controlled Differential Equation.
    
    This implements the neural function f(t, z) that returns a matrix of shape
    (hidden_channels, input_channels) for computing the product f(t, z) dX/dt.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        bottle_neck_channel: int,
        *,
        rngs: nnx.Rngs
    ):
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bottle_neck_channel = bottle_neck_channel
        
        # Neural network layers
        self.linear1 = nnx.Linear(hidden_channels, bottle_neck_channel, rngs=rngs)
        self.linear2 = nnx.Linear(bottle_neck_channel, bottle_neck_channel, rngs=rngs)
        self.linear3 = nnx.Linear(bottle_neck_channel, bottle_neck_channel, rngs=rngs)
        self.linear4 = nnx.Linear(bottle_neck_channel, hidden_channels * input_channels, rngs=rngs)
        
        # Track number of function evaluations (useful for debugging)
        self.nfe = 0
    
    def __call__(self, t: float, z: jnp.ndarray, args=None) -> jnp.ndarray:
        """Compute the vector field f(t, z).
        
        Args:
            t: Time (scalar)
            z: Hidden state of shape (..., hidden_channels)
            
        Returns:
            Matrix of shape (..., hidden_channels, input_channels)
        """
        # TODO: Function evaluation counter disabled due to JAX tracing constraints
        # self.nfe += 1  # Cannot mutate during tracing
        
        # Forward pass through the neural network
        x = self.linear1(z)
        x = jax.nn.elu(x)
        x = self.linear2(x)
        x = jax.nn.elu(x)
        x = self.linear3(x)
        x = jax.nn.elu(x)
        x = self.linear4(x)
        x = jax.nn.tanh(x)
        
        # Reshape to matrix form
        batch_shape = z.shape[:-1]
        x = x.reshape(batch_shape + (self.hidden_channels, self.input_channels))
        
        return x


class SecondOrderCDEFunc(nnx.Module):
    """Second-order CDE function with two separate neural networks."""
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        bottle_neck_channel: int,
        *,
        rngs: nnx.Rngs
    ):
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bottle_neck_channel = bottle_neck_channel
        
        # Split the rngs for two separate networks
        rngs1 = nnx.Rngs(params=jax.random.split(rngs.params())[0])
        rngs2 = nnx.Rngs(params=jax.random.split(rngs.params())[1])
        
        # First neural network for velocity (first derivative)
        self.func1 = CDEFunc(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            bottle_neck_channel=bottle_neck_channel,
            rngs=rngs1
        )
        
        # Second neural network for acceleration (second derivative)
        self.func2 = CDEFunc(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            bottle_neck_channel=bottle_neck_channel,
            rngs=rngs2
        )
    
    def get_velocity_func(self):
        """Get the velocity (first derivative) neural function."""
        return self.func1
    
    def get_acceleration_func(self):
        """Get the acceleration (second derivative) neural function."""
        return self.func2


class InterpolationMethod:
    """Base class for different interpolation methods."""
    
    @staticmethod
    def create_interpolation(t: jnp.ndarray, x: jnp.ndarray) -> Any:
        """Create interpolation object from time points and values."""
        raise NotImplementedError


class HermiteInterpolation(InterpolationMethod):
    """Hermite cubic spline interpolation using diffrax."""
    
    @staticmethod
    def create_interpolation(t: jnp.ndarray, x: jnp.ndarray) -> diffrax.CubicInterpolation:
        """Create Hermite cubic interpolation.
        
        Args:
            t: Time points of shape (batch_size, seq_len)
            x: Values of shape (batch_size, seq_len, input_channels)
            
        Returns:
            diffrax.CubicInterpolation object
        """
        # Concatenate time as first channel (matching torchcde convention)
        t_expanded = jnp.expand_dims(t, -1)  # (batch_size, seq_len, 1)
        x_with_time = jnp.concatenate([t_expanded, x], axis=-1)  # (batch_size, seq_len, input_channels + 1)
        
        # Calculate Hermite coefficients using backward differences
        # Note: diffrax expects (seq_len, batch_size, channels) for coefficients
        x_transposed = jnp.transpose(x_with_time, (1, 0, 2))  # (seq_len, batch_size, channels)
        t_1d = t[0]  # Use first batch's time points (assuming all batches have same times)
        
        coeffs = diffrax.backward_hermite_coefficients(ys=x_transposed, ts=t_1d)
        
        # Create cubic interpolation
        return diffrax.CubicInterpolation(ts=t_1d, coeffs=coeffs)


class LinearInterpolation(InterpolationMethod):
    """Linear interpolation using diffrax."""
    
    @staticmethod 
    def create_interpolation(t: jnp.ndarray, x: jnp.ndarray) -> diffrax.LinearInterpolation:
        """Create linear interpolation.
        
        Args:
            t: Time points of shape (batch_size, seq_len)
            x: Values of shape (batch_size, seq_len, input_channels)
            
        Returns:
            diffrax.LinearInterpolation object
        """
        # Concatenate time as first channel
        t_expanded = jnp.expand_dims(t, -1)
        x_with_time = jnp.concatenate([t_expanded, x], axis=-1)
        
        # Transpose for diffrax format
        x_transposed = jnp.transpose(x_with_time, (1, 0, 2))
        t_1d = t[0]
        
        return diffrax.LinearInterpolation(ts=t_1d, ys=x_transposed)


class SavitzkyGolayInterpolation(InterpolationMethod):
    """Savitzky-Golay filtering interpolation for SO(3) data.
    
    This interpolation method applies Savitzky-Golay smoothing once to fit
    polynomial coefficients, then creates a functional interpolation using
    those coefficients. This avoids piecewise smoothing that could introduce
    discontinuities in the Neural CDE.
    """
    
    def __init__(self, polynomial_order: int = 3, learnable_weights: Optional[jnp.ndarray] = None, 
                 second_order: bool = False):
        """Initialize Savitzky-Golay interpolation.
        
        Args:
            polynomial_order: Order of polynomial for fitting (default: 3)
            learnable_weights: Learnable weights for data points (shape: 12*3=36)
            second_order: Whether to include second-order derivatives
        """
        self.polynomial_order = polynomial_order
        self.learnable_weights = learnable_weights
        self.second_order = second_order
    
    def create_interpolation(self, t: jnp.ndarray, x: jnp.ndarray) -> Any:
        """Create proper SO(3) polynomial interpolation using diffrax-compatible path.
        
        Args:
            t: Time points of shape (batch_size, seq_len)
            x: Values of shape (batch_size, seq_len, input_channels)
            
        Returns:
            List of SO3PolynomialPath objects for each batch element
        """
        # Handle both 3D and 4D input shapes
        if len(x.shape) == 4:
            # Input is already (batch_size, seq_len, 3, 3)
            batch_size, seq_len, _, _ = x.shape
            R = x
        elif len(x.shape) == 3:
            # Input is (batch_size, seq_len, input_channels)
            batch_size, seq_len, input_channels = x.shape
            if input_channels == 9:
                # Reshape 9D to rotation matrices
                R = x.reshape(batch_size, seq_len, 3, 3)
            else:
                raise ValueError(f"SavitzkyGolayInterpolation only supports 9D rotation matrices, got {input_channels}D")
        else:
            raise ValueError(f"SavitzkyGolayInterpolation expects 3D or 4D input, got {len(x.shape)}D")
        
        t_1d = t[0]  # Use first batch's time points (assuming all batches use same times)
        
        # Prepare weights for the entire batch
        weights_batch = None
        if self.learnable_weights is not None:
            # Resize weights to match sequence length if needed
            if len(self.learnable_weights) != seq_len:
                # Interpolate weights to match sequence length
                weight_indices = jnp.linspace(0, len(self.learnable_weights) - 1, seq_len)
                weights_interp = jnp.interp(weight_indices, jnp.arange(len(self.learnable_weights)), self.learnable_weights)
            else:
                weights_interp = self.learnable_weights
            
            # Broadcast weights to all batch elements
            weights_batch = jnp.broadcast_to(weights_interp[None, :], (batch_size, seq_len))
        
        # Create single batched polynomial path
        path = SO3PolynomialPath(
            R=R,  # (batch_size, seq_len, 3, 3)
            t=t_1d,  # (seq_len,)
            p=self.polynomial_order,
            weight=weights_batch,  # (batch_size, seq_len) or None
            second_order=self.second_order
        )
        
        return path




class SO3NeuralCDE(nnx.Module):
    """SO(3) Neural Controlled Differential Equation model.
    
    This implements a neural CDE specifically designed for rotational dynamics on SO(3).
    It supports different interpolation methods and handles various output representations
    (9D rotation matrices, 6D continuous representations, quaternions).
    """
    
    def __init__(
        self,
        input_channel: int,
        latent_channels: int, 
        hidden_channels: int,
        output_channel: int,
        interpolation_method: str = 'hermite',
        method: str = 'tsit5',
        atol: float = 1e-5,
        rtol: float = 1e-5,
        max_steps: int = 4096,
        sg_learnable_weights: bool = True,
        sg_polynomial_order: int = 3,
        second_order: bool = False,
        use_refit: bool = False,
        *,
        rngs: nnx.Rngs
    ):
        self.input_channel = input_channel
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.output_channel = output_channel
        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.max_steps = max_steps
        self.second_order = second_order
        self.use_refit = use_refit
        
        # Store initial tolerances for reference
        self.initial_atol = atol
        self.initial_rtol = rtol
        
        # Learnable weights for Savitzky-Golay (will be sized based on sequence length)
        if interpolation_method == 'savitzky_golay' or interpolation_method == 'sg':
            if sg_learnable_weights:
                # We'll initialize with default size, but it should be sized based on actual sequence length
                self.sg_weights = nnx.Param(jnp.ones(12))  # Per-time-point weights
            else:
                self.sg_weights = None  # No learnable weights - use fixed uniform weights
        else:
            self.sg_weights = None
        
        # Choose interpolation method
        if interpolation_method == 'hermite':
            self.interpolation_method = HermiteInterpolation()
        elif interpolation_method == 'linear':
            self.interpolation_method = LinearInterpolation()
        elif interpolation_method == 'savitzky_golay' or interpolation_method == 'sg':
            self.interpolation_method = SavitzkyGolayInterpolation(
                polynomial_order=sg_polynomial_order, 
                learnable_weights=self.sg_weights.value if self.sg_weights is not None else None,
                second_order=second_order
            )
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}. "
                           f"Supported methods: 'hermite', 'linear', 'savitzky_golay'/'sg'")
        
        # Neural vector field function(s)
        if second_order:
            # Two separate neural networks for second-order dynamics
            self.func = SecondOrderCDEFunc(
                input_channels=input_channel,  # Each network handles input_channels without time
                hidden_channels=latent_channels,
                bottle_neck_channel=hidden_channels,
                rngs=rngs
            )
        else:
            # Single neural network for first-order dynamics
            self.func = CDEFunc(
                input_channels=input_channel + 1,  # Standard with time channel
                hidden_channels=latent_channels,
                bottle_neck_channel=hidden_channels,
                rngs=rngs
            )
        
        # Initial value network layers
        self.initial_linear1 = nnx.Linear(input_channel + 1, latent_channels, rngs=rngs)
        self.initial_linear2 = nnx.Linear(latent_channels, latent_channels, rngs=rngs)
        
        # Output network layers
        self.output_linear1 = nnx.Linear(latent_channels, latent_channels, rngs=rngs)
        self.output_linear2 = nnx.Linear(latent_channels, output_channel + 1, rngs=rngs)  # +1 to match original
        
        # Set the appropriate post-processing function based on output dimension
        if output_channel == 9:
            self.postprocess_fn = symmetric_orthogonalization
        elif output_channel == 6:
            self.postprocess_fn = gramschmidt_to_rotmat
        elif output_channel == 4 or output_channel == 3:
            self.postprocess_fn = lambda x: x
        else:
            self.postprocess_fn = lambda x: x
    
    def create_control_path_single(self, t: jnp.ndarray, x: jnp.ndarray) -> Any:
        """Create interpolated control path for a single sample.
        
        Args:
            t: Time points of shape (seq_len,)
            x: Input data of shape (seq_len, input_channels) or (seq_len, 3, 3)
            
        Returns:
            Interpolation object that can be used as base for derivative paths
        """
        # Flatten input if it's 3D (seq_len, 3, 3) -> (seq_len, 9)
        if len(x.shape) == 3:
            seq_len, _, _ = x.shape
            x_flat = x.reshape(seq_len, -1)
        else:
            x_flat = x
        
        # For both first and second order, create standard interpolation with time channel
        # For single sample, we need to create interpolation that removes batch dimension in coefficients
        # Add batch dimension temporarily for interpolation creation, then adjust the output
        t_batch = t[None, :]  # (1, seq_len)
        x_batch = x_flat[None, :]  # (1, seq_len, input_channels)
        
        interpolation = self.interpolation_method.create_interpolation(t_batch, x_batch)
        
        # Handle different interpolation return types
        if hasattr(interpolation, 'is_batched'):
            # For SavitzkyGolayInterpolation, we now get a single batched path
            # For single sample processing, we need to handle the batch dimension
            return interpolation
        elif hasattr(interpolation, 'coeffs'):
            # For CubicInterpolation, coeffs is a tuple of arrays with shape (seq_len-1, 1, input_channels+1)
            # We need to squeeze the batch dimension to get (seq_len-1, input_channels+1)
            new_coeffs = tuple(jnp.squeeze(coeff, axis=1) for coeff in interpolation.coeffs)
            return diffrax.CubicInterpolation(ts=interpolation.ts, coeffs=new_coeffs)
        elif hasattr(interpolation, 'ys'):
            # For LinearInterpolation, ys has shape (seq_len, 1, input_channels+1)
            # We need to squeeze the batch dimension to get (seq_len, input_channels+1)
            new_ys = jnp.squeeze(interpolation.ys, axis=1)
            return diffrax.LinearInterpolation(ts=interpolation.ts, ys=new_ys)
        else:
            return interpolation
    
    def solve_cde_single(
        self, 
        control: Any, 
        z0: jnp.ndarray, 
        t_eval: jnp.ndarray
    ) -> jnp.ndarray:
        """Solve the controlled differential equation for a single sample.
        
        Args:
            control: Interpolated control path (base path for creating derivative paths)
            z0: Initial condition of shape (latent_channels,)
            t_eval: Evaluation times (1D array)
            
        Returns:
            Solution at evaluation times of shape (len(t_eval), latent_channels)
        """
        # Create control term based on order
        if self.second_order:
            # For second-order, require SO3PolynomialPath and use MultiTerm approach
            if not (hasattr(control, 'derivative') and hasattr(control, 'second_order')):
                raise ValueError(
                    "Second-order Neural CDE requires SO3PolynomialPath with second_order=True. "
                    f"Got control type: {type(control)}"
                )
            
            # MultiTerm approach with separate derivative paths for better performance
            first_path = SO3FirstDerivativePath(control)
            second_path = SO3SecondDerivativePath(control)
            
            # Create separate ControlTerms for each derivative
            velocity_func = self.func.get_velocity_func()
            acceleration_func = self.func.get_acceleration_func()
            
            control_term1 = diffrax.ControlTerm(velocity_func, first_path)
            control_term2 = diffrax.ControlTerm(acceleration_func, second_path)
            
            # Combine with MultiTerm
            control_term = diffrax.MultiTerm(control_term1, control_term2)
        else:
            # For first-order, use standard control term
            control_term = diffrax.ControlTerm(self.func, control).to_ode()
        
        # Set up solver
        if self.method == 'tsit5':
            solver = diffrax.Tsit5()
        elif self.method == 'dopri5':
            solver = diffrax.Dopri5()
        elif self.method == 'euler':
            solver = diffrax.Euler()
        else:
            raise ValueError(f"Unknown solver method: {self.method}")
        
        # Configure step size controller based on solver
        if self.method == 'euler':
            # Euler method doesn't provide error estimates, use constant step size
            stepsize_controller = diffrax.ConstantStepSize()
            dt0 = 0.01  # Fixed step size
        else:
            # Adaptive methods can use PID controller
            stepsize_controller = diffrax.PIDController(rtol=self.rtol, atol=self.atol)
            dt0 = None
        
        # Solve the differential equation
        solution = diffrax.diffeqsolve(
            terms=control_term,
            solver=solver,
            t0=t_eval[0],
            t1=t_eval[-1],
            dt0=dt0,
            y0=z0,
            saveat=diffrax.SaveAt(ts=t_eval),
            stepsize_controller=stepsize_controller,
            max_steps=self.max_steps
        )
        
        return solution.ys, solution.stats
    
    def update_tolerances(self, atol: float, rtol: float) -> None:
        """
        Update the ODE solver tolerances dynamically.
        
        This allows for tolerance scheduling during training, where tolerances
        can start relaxed for fast early training and become stricter for
        accurate final convergence.
        
        Args:
            atol: New absolute tolerance
            rtol: New relative tolerance
        """
        self.atol = atol
        self.rtol = rtol
    
    def get_solver_stats(self):
        """Get the last solver statistics."""
        return getattr(self, '_last_solver_stats', {
            'num_steps': 0, 'num_accepted_steps': 0, 
            'num_rejected_steps': 0, 'max_steps_reached': False
        })
    
    def __call__(
        self, 
        t_recon: jnp.ndarray, 
        t_fut: jnp.ndarray,
        x: jnp.ndarray,
        *args,
        t_eval: Optional[jnp.ndarray] = None,
        return_solver_stats: bool = False,
        **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[dict]]:
        """Forward pass through the SO(3) Neural CDE.
        
        Args:
            t_recon: Reconstruction time points of shape (batch_size, seq_len) for per-sample irregular times
            t_fut: Future time points of shape (batch_size, n_future) for per-sample irregular times
            x: Input data of shape (batch_size, seq_len, input_channels) or (batch_size, seq_len, 3, 3)
            *args: Additional arguments (e.g., omega, moi)
            t_eval: Optional evaluation times. If None, uses concatenation of t_recon and t_fut
            return_solver_stats: Whether to return solver statistics
            
        Returns:
            Tuple of (reconstructions, predictions, solver_stats)
        """
        batch_size = x.shape[0]
        
        # Handle backward compatibility: if time vectors are 1D (shared), broadcast to per-sample
        if len(t_recon.shape) == 1:
            print("INFO: Broadcasting shared time vectors to per-sample. Consider using irregular sampling for full Neural CDE capabilities.")
            t_recon = jnp.broadcast_to(t_recon[None, :], (batch_size, t_recon.shape[0]))
            t_fut = jnp.broadcast_to(t_fut[None, :], (batch_size, t_fut.shape[0]))
        
        # Ensure we have per-sample time vectors
        assert t_recon.shape[0] == batch_size, f"t_recon batch size {t_recon.shape[0]} != {batch_size}"
        assert t_fut.shape[0] == batch_size, f"t_fut batch size {t_fut.shape[0]} != {batch_size}"
        
        # Use vmap to handle batching in the JAX way - each sample gets its own time grid
        if return_solver_stats:
            batched_forward = jax.vmap(self.forward_single_with_stats, in_axes=(0, 0, 0))
            recon, pred, stats_list = batched_forward(t_recon, t_fut, x)
            # Aggregate stats across batch - extract num_steps from each stats dict
            # Since we can't use Python dict methods in JAX, we'll handle this differently
            total_steps = jnp.sum(stats_list)  # stats_list should be array of num_steps
            avg_stats = {'num_steps': (total_steps / batch_size).astype(float)}
            return recon, pred, avg_stats
        else:
            batched_forward = jax.vmap(self.forward_single, in_axes=(0, 0, 0))
            recon, pred = batched_forward(t_recon, t_fut, x)
            return recon, pred, None
    
    def forward_single_with_stats(
        self,
        t_recon: jnp.ndarray,
        t_fut: jnp.ndarray, 
        x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Forward pass for a single sample that returns solver statistics.
        
        Args:
            t_recon: Reconstruction time points of shape (seq_len,)
            t_fut: Future time points of shape (n_future,)
            x: Input data of shape (seq_len, input_channels) or (seq_len, 3, 3)
            
        Returns:
            Tuple of (reconstructions, predictions, num_steps)
        """
        # Flatten input if needed
        if len(x.shape) == 3:
            seq_len, _, _ = x.shape
            x_flat = x.reshape(seq_len, -1)
        else:
            x_flat = x
            
        # Step 1: Reconstruction phase
        control = self.create_control_path_single(t_recon, x)
        
        # Get initial value
        if hasattr(control, 'interval'):
            t_start = control.interval[0]
        else:
            t_start = t_recon[0]
        
        X0 = control.evaluate(t_start)
        z0 = self.initial_linear1(X0)
        z0 = jax.nn.elu(z0)
        z0 = self.initial_linear2(z0)  # Shape: (latent_channels,)
        
        # Solve CDE for reconstruction phase
        z_T_recon, recon_stats = self.solve_cde_single(control, z0, t_recon)  # Shape: (n_recon, latent_channels)
        
        # Apply output network to get reconstructions
        output_recon = self.output_linear1(z_T_recon)
        output_recon = jax.nn.elu(output_recon)
        output_recon = self.output_linear2(output_recon)  # Shape: (n_recon, output_channel + 1)
        
        # Remove the +1 dimension and apply post-processing
        output_recon = output_recon[..., 1:]  # Remove first dimension
        recon = jax.vmap(self.postprocess_fn)(output_recon)
        
        # Step 2: Prediction phase - autoregressive only if refitting enabled
        if self.use_refit:
            # Note: Autoregressive mode not implemented for stats tracking yet
            pred_recon, pred = self._predict_autoregressive(t_recon, t_fut, x_flat, recon, z_T_recon[-1])
            # Return recon_stats for now
            return pred_recon, pred, recon_stats.get('num_steps', 0)
        else:
            # Standard non-autoregressive prediction: solve CDE for entire sequence at once
            t_eval = jnp.concatenate([t_recon, t_fut])
            z_T_full, pred_stats = self.solve_cde_single(control, z0, t_eval)  # Shape: (n_recon + n_fut, latent_channels)
            
            # Apply output network to entire sequence
            output_full = self.output_linear1(z_T_full)
            output_full = jax.nn.elu(output_full)
            output_full = self.output_linear2(output_full)  # Shape: (n_recon + n_fut, output_channel + 1)
            
            # Remove the +1 dimension and apply post-processing
            output_full = output_full[..., 1:]  # Remove first dimension
            processed_output = jax.vmap(self.postprocess_fn)(output_full)  # Shape: (n_recon + n_fut, 3, 3) or (n_recon + n_fut, output_channel)
            
            # Split into reconstruction and prediction parts
            n_recon = t_recon.shape[0]
            recon_std = processed_output[:n_recon]
            pred = processed_output[n_recon:]
            
            # Return combined stats (use prediction stats as they include more solver steps)
            num_steps = pred_stats.get('num_steps', 0)
            return recon_std, pred, num_steps
    
    def forward_single(
        self,
        t_recon: jnp.ndarray,
        t_fut: jnp.ndarray, 
        x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass for a single sample (backward compatibility).
        
        Args:
            t_recon: Reconstruction time points of shape (seq_len,)
            t_fut: Future time points of shape (n_future,)
            x: Input data of shape (seq_len, input_channels) or (seq_len, 3, 3)
            
        Returns:
            Tuple of (reconstructions, predictions)
        """
        recon, pred, _ = self.forward_single_with_stats(t_recon, t_fut, x)
        return recon, pred

    def _predict_autoregressive(
        self,
        t_recon: jnp.ndarray,
        t_fut: jnp.ndarray,
        x_flat: jnp.ndarray,
        recon: jnp.ndarray,
        z0: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Optimized autoregressive prediction with refitting using jax.lax.scan.
        
        This method is only called when self.use_refit=True, so it always
        uses reconstructed data instead of original input data.
        
        The optimization uses jax.lax.scan to JIT compile the entire autoregressive
        loop, eliminating Python overhead and enabling XLA optimizations.
        """
        # Use reconstructed data for refitting (this method is only called when refitting is enabled)
        base_data = recon.reshape(recon.shape[0], -1)
        n_future = t_fut.shape[0]
        pred_shape = recon.shape[1:]  # Shape for each prediction (same as reconstruction)
        
        def prediction_step(carry, step_inputs):
            """Single prediction step for scan.
            
            Args:
                carry: (current_z, accumulated_predictions)
                step_inputs: (step_idx, t_prev, t_curr)
                
            Returns:
                new_carry: (new_z, updated_predictions)
                output: pred_step
            """
            current_z, accumulated_predictions = carry
            step_idx, t_prev, t_curr = step_inputs
            
            # Build current data sequence: base_data + predictions so far
            # Use a mask-based approach that's JIT-friendly
            # Create masks for selecting valid data up to current step
            step_mask = jnp.arange(n_future) < step_idx
            
            # Apply mask to predictions - set invalid entries to zeros
            masked_predictions = jnp.where(
                step_mask[:, None, None],  # Broadcast to (n_future, 3, 3)
                accumulated_predictions,
                jnp.zeros_like(accumulated_predictions)
            )
            
            # For current data: always use base_data + masked predictions (zeros for invalid)
            pred_data_flat = masked_predictions.reshape(-1, base_data.shape[1])
            current_data = jnp.concatenate([base_data, pred_data_flat], axis=0)
            
            # For times: base times + masked future times (zeros for invalid)
            masked_fut_times = jnp.where(step_mask, t_fut, 0.0)
            current_times = jnp.concatenate([t_recon, masked_fut_times], axis=0)
            
            # Create control path for current step
            # The interpolation will handle the padded zeros appropriately
            control = self.create_control_path_single(current_times, current_data)
            
            # Solve CDE for one step
            t_eval = jnp.array([t_prev, t_curr])
            z_step, _ = self.solve_cde_single(control, current_z, t_eval)
            new_z = z_step[-1]
            
            # Generate prediction
            output_step = self.output_linear1(new_z.reshape(1, -1))
            output_step = jax.nn.elu(output_step)
            output_step = self.output_linear2(output_step)
            output_step = output_step[..., 1:]  # Remove first dimension
            pred_step = self.postprocess_fn(output_step[0])
            
            # Update accumulated predictions
            new_predictions = accumulated_predictions.at[step_idx].set(pred_step)
            
            return (new_z, new_predictions), pred_step
        
        # Prepare scan inputs
        step_indices = jnp.arange(n_future)
        t_prev = jnp.concatenate([t_recon[-1:], t_fut[:-1]])  # Previous time for each step
        scan_inputs = (step_indices, t_prev, t_fut)
        
        # Initialize carry state
        # Pre-allocate predictions array with correct shape
        initial_predictions = jnp.zeros((n_future,) + pred_shape)
        initial_carry = (z0, initial_predictions)
        
        # Run scan - this will be JIT compiled for efficiency
        final_carry, predictions = jax.lax.scan(
            prediction_step,
            initial_carry,
            scan_inputs
        )
        
        return recon, predictions


def create_so3_neural_cde(
    input_channel: int,
    latent_channels: int,
    hidden_channels: int,
    output_channel: int,
    interpolation_method: str = 'hermite',
    method: str = 'tsit5',
    atol: float = 1e-5,
    rtol: float = 1e-5,
    max_steps: int = 4096,
    sg_learnable_weights: bool = True,
    sg_polynomial_order: int = 3,
    second_order: bool = False,
    use_refit: bool = False,
    *,
    rngs: nnx.Rngs
) -> SO3NeuralCDE:
    """Factory function to create an SO(3) Neural CDE model.
    
    Args:
        input_channel: Number of input channels
        latent_channels: Number of latent channels in the CDE state
        hidden_channels: Number of hidden channels in the bottleneck
        output_channel: Number of output channels (9 for rotation matrices, 6 for 6D repr, etc.)
        interpolation_method: Interpolation method ('hermite', 'linear', or 'savitzky_golay'/'sg')
        method: ODE solver method ('tsit5', 'dopri5', 'euler')
        atol: Absolute tolerance for ODE solver
        rtol: Relative tolerance for ODE solver
        max_steps: Maximum number of solver steps
        rngs: Random number generators
        
    Returns:
        Initialized SO3NeuralCDE model
    """
    return SO3NeuralCDE(
        input_channel=input_channel,
        latent_channels=latent_channels,
        hidden_channels=hidden_channels,
        output_channel=output_channel,
        interpolation_method=interpolation_method,
        method=method,
        atol=atol,
        rtol=rtol,
        max_steps=max_steps,
        sg_learnable_weights=sg_learnable_weights,
        sg_polynomial_order=sg_polynomial_order,
        second_order=second_order,
        use_refit=use_refit,
        rngs=rngs
    )


# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("Testing SO(3) Neural CDE implementation...")
    
    # Model parameters
    input_channel = 9  # For 3x3 rotation matrices
    latent_channels = 32
    hidden_channels = 32
    output_channel = 9
    
    # Initialize model
    key = jax.random.key(42)
    rngs = nnx.Rngs(params=key)
    
    model = create_so3_neural_cde(
        input_channel=input_channel,
        latent_channels=latent_channels,
        hidden_channels=hidden_channels,
        output_channel=output_channel,
        interpolation_method='hermite',
        method='tsit5',
        rngs=rngs
    )
    
    print(f"Model created successfully!")
    print(f"  Input channels: {input_channel}")
    print(f"  Latent channels: {latent_channels}")
    print(f"  Output channels: {output_channel}")
    print(f"  Interpolation: Hermite cubic splines")
    print(f"  Solver: Tsit5")
    
    # Create test data
    batch_size = 16
    seq_len = 12
    n_future = 8
    
    # Create time grids
    t_recon = jnp.linspace(0, 1.2, seq_len)[None, :].repeat(batch_size, axis=0)  # (batch_size, seq_len)
    t_fut = jnp.linspace(1.3, 2.1, n_future)[None, :].repeat(batch_size, axis=0)  # (batch_size, n_future)
    
    # Create random rotation matrix data
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (batch_size, seq_len, 3, 3))
    
    print(f"\nTest data shapes:")
    print(f"  t_recon: {t_recon.shape}")
    print(f"  t_fut: {t_fut.shape}")
    print(f"  x: {x.shape}")
    
    try:
        print("\nRunning forward pass...")
        start_time = time.time()
        
        recon, pred, stats = model(t_recon, t_fut, x)
        
        forward_time = time.time() - start_time
        
        print(f"Forward pass completed!")
        print(f"  Time taken: {forward_time:.3f} seconds")
        print(f"  Reconstruction shape: {recon.shape}")
        print(f"  Prediction shape: {pred.shape}")
        print(f"  Function evaluations: {model.func.nfe}")
        
        # Check that outputs are valid rotation matrices
        def check_rotation_matrix(R):
            """Check if matrix is a valid rotation matrix."""
            # Check orthogonality: R @ R.T = I
            is_orthogonal = jnp.allclose(R @ jnp.transpose(R, (0, 1, 3, 2)), 
                                       jnp.eye(3)[None, None, :, :], atol=1e-3)
            # Check determinant = 1
            det = jnp.linalg.det(R)
            is_proper = jnp.allclose(det, 1.0, atol=1e-3)
            return is_orthogonal, is_proper
        
        recon_orth, recon_proper = check_rotation_matrix(recon)
        pred_orth, pred_proper = check_rotation_matrix(pred)
        
        print(f"\nRotation matrix validation:")
        print(f"  Reconstruction - Orthogonal: {jnp.all(recon_orth)}, Proper: {jnp.all(recon_proper)}")
        print(f"  Prediction - Orthogonal: {jnp.all(pred_orth)}, Proper: {jnp.all(pred_proper)}")
        
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
