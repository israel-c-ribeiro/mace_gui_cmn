"""
App-wide constants for MACE-Interactive-Lite.
"""

APP_NAME = "CMN-MACE Studio"
APP_VERSION = "1.0.0"

# Supported file extensions for the file uploader
SUPPORTED_EXTENSIONS = ["vasp", "cif", "xyz", "traj", "extxyz"]

# MACE model sizes available via mace_mp
MACE_MODELS: list[str] = ["small", "medium", "large"]

# Geometry optimisers available through ASE
OPTIMIZERS: list[str] = ["FIRE", "BFGS", "LBFGS"]

# MD ensembles
MD_ENSEMBLES: list[str] = ["NVT-Langevin", "NVE-VelocityVerlet"]

# Float precision options
DTYPES: list[str] = ["float64", "float32"]

# Vacuum threshold (Å) used to classify a structure as a slab
VACUUM_THRESHOLD: float = 8.0
