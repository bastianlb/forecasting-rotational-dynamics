#!/usr/bin/env python
"""
Tolerance scheduling for ODE solvers during training.
Similar to learning rate scheduling but for solver tolerances.
"""
import jax.numpy as jnp
from typing import Union, Literal


class ToleranceScheduler:
    """
    Scheduler for ODE solver tolerances during training.
    Allows starting with relaxed tolerances for fast initial training
    and gradually increasing precision for accurate final convergence.
    """
    
    def __init__(
        self,
        initial_atol: float = 1e-3,
        initial_rtol: float = 1e-3,
        final_atol: float = 1e-4,
        final_rtol: float = 1e-4,
        scheduler_type: Literal["linear", "exponential", "cosine"] = "exponential",
        warmup_epochs: int = 5,
        total_epochs: int = 20
    ):
        """
        Initialize tolerance scheduler.
        
        Args:
            initial_atol: Initial absolute tolerance (relaxed)
            initial_rtol: Initial relative tolerance (relaxed)
            final_atol: Final absolute tolerance (strict)
            final_rtol: Final relative tolerance (strict)
            scheduler_type: Type of scheduling curve
            warmup_epochs: Number of epochs to reach final tolerance
            total_epochs: Total training epochs (for cosine scheduling)
        """
        self.initial_atol = initial_atol
        self.initial_rtol = initial_rtol
        self.final_atol = final_atol
        self.final_rtol = final_rtol
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
        # Validate inputs
        assert initial_atol >= final_atol, "Initial tolerance should be >= final tolerance"
        assert initial_rtol >= final_rtol, "Initial tolerance should be >= final tolerance"
        assert warmup_epochs > 0, "Warmup epochs must be positive"
        
    def get_tolerances(self, epoch: int) -> tuple[float, float]:
        """
        Get current tolerances for the given epoch.
        
        Args:
            epoch: Current training epoch (0-indexed)
            
        Returns:
            Tuple of (atol, rtol) for current epoch
        """
        if epoch >= self.warmup_epochs:
            # After warmup, use final tolerances
            return self.final_atol, self.final_rtol
        
        # Calculate progress through warmup period
        progress = epoch / self.warmup_epochs
        
        if self.scheduler_type == "linear":
            atol = self._linear_schedule(progress, self.initial_atol, self.final_atol)
            rtol = self._linear_schedule(progress, self.initial_rtol, self.final_rtol)
        elif self.scheduler_type == "exponential":
            atol = self._exponential_schedule(progress, self.initial_atol, self.final_atol)
            rtol = self._exponential_schedule(progress, self.initial_rtol, self.final_rtol)
        elif self.scheduler_type == "cosine":
            atol = self._cosine_schedule(progress, self.initial_atol, self.final_atol)
            rtol = self._cosine_schedule(progress, self.initial_rtol, self.final_rtol)
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
        
        return atol, rtol
    
    def _linear_schedule(self, progress: float, initial: float, final: float) -> float:
        """Linear interpolation between initial and final values."""
        return initial + progress * (final - initial)
    
    def _exponential_schedule(self, progress: float, initial: float, final: float) -> float:
        """Exponential decay from initial to final values."""
        # Use log space for smooth exponential transition
        log_initial = jnp.log(initial)
        log_final = jnp.log(final)
        log_current = log_initial + progress * (log_final - log_initial)
        return float(jnp.exp(log_current))
    
    def _cosine_schedule(self, progress: float, initial: float, final: float) -> float:
        """Cosine annealing from initial to final values."""
        cosine_progress = 0.5 * (1 + jnp.cos(jnp.pi * progress))
        return final + (initial - final) * cosine_progress
    
    def __repr__(self) -> str:
        return (
            f"ToleranceScheduler("
            f"initial={self.initial_atol:.2e}/{self.initial_rtol:.2e}, "
            f"final={self.final_atol:.2e}/{self.final_rtol:.2e}, "
            f"type={self.scheduler_type}, warmup={self.warmup_epochs})"
        )


def create_tolerance_scheduler_from_config(cfg) -> Union[ToleranceScheduler, None]:
    """
    Create tolerance scheduler from configuration.
    
    Args:
        cfg: Configuration object with MODEL.TOLERANCE_SCHEDULING section
        
    Returns:
        ToleranceScheduler instance or None if scheduling disabled
    """
    if not cfg.MODEL.TOLERANCE_SCHEDULING.ENABLE:
        return None
    
    return ToleranceScheduler(
        initial_atol=cfg.MODEL.TOLERANCE_SCHEDULING.INITIAL_ATOL,
        initial_rtol=cfg.MODEL.TOLERANCE_SCHEDULING.INITIAL_RTOL,
        final_atol=cfg.MODEL.TOLERANCE_SCHEDULING.FINAL_ATOL,
        final_rtol=cfg.MODEL.TOLERANCE_SCHEDULING.FINAL_RTOL,
        scheduler_type=cfg.MODEL.TOLERANCE_SCHEDULING.SCHEDULER_TYPE,
        warmup_epochs=cfg.MODEL.TOLERANCE_SCHEDULING.WARMUP_EPOCHS,
        total_epochs=cfg.TRAIN.NUM_EPOCHS
    )