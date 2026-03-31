import os
import platform
import torch

THREAD_VARS = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_NUM_THREADS", "NUMEXPR_NUM_THREADS"]

def disable_numpy_multithreading():
    """
    Disable NumPy multithreading by setting environment variables.
    This should be called before importing NumPy to take effect.
    """
    for var in THREAD_VARS:
        os.environ[var] = "1"
    # Suppress macOS Objective-C runtime warnings during multiprocessing spawn
    if platform.system() == "Darwin":
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

def use_deterministic_cuda():
    """
    Set environment variables to ensure deterministic behavior in CUDA.
    This should be called before importing PyTorch or any other CUDA-dependent libraries.
    """
    # See https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    # Experimentally, both of these seem fine
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
def get_device():
    # The following would work on recent PyTorch:
    # (torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")
    # but we want to support PyTorch 2.2, so:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    else:
        return "cpu"
