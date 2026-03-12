"""
Apex AMP (Automatic Mixed Precision) compatibility module.

The Apex amp module is not available on the current master branch.
This module provides a compatibility wrapper that delegates to PyTorch's native AMP implementation.

For users migrating from older Apex versions that had a dedicated AMP module,
use torch.cuda.amp instead:
    from torch.cuda.amp import autocast, GradScaler
"""

# Provide PyTorch's native AMP for backward compatibility
from torch.cuda.amp import autocast, GradScaler

__all__ = ["autocast", "GradScaler"]
