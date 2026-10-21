"""
Neural-network workflow for XAS/XMCD angular-momentum reconstruction.

The ``src`` package contains the complete reproducible workflow:

1. ``prepare_dataset`` prepares XAS/XMCD samples and target values.
2. ``split_dataset`` creates fixed train/validation/test assignments.
3. ``audit_dataset`` checks duplicates, leakage, and label consistency.
4. ``preprocess`` fits preprocessing statistics on training data only.
5. ``data_loader`` constructs PyTorch datasets and data loaders.
6. ``model`` defines the multi-output neural network for Sz and Lz.
7. ``train`` trains and selects the evaluation checkpoint.
8. ``metrics`` calculates regression metrics.
9. ``evaluate`` evaluates the complete saved test subset.

Modules are intentionally not imported here. This keeps ``import src``
lightweight and avoids loading NumPy, pandas, or PyTorch until needed.
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
]