"""TokenForge: a 7B model from raw weights to a fault-tested deployment on one L4.

Subpackages are imported lazily so the serving layer runs without torch/triton.
"""

__version__ = "0.1.0"
