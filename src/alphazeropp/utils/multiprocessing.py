import os
import warnings
import itertools
from multiprocessing import Pool


class MultiprocessingManager:
    """Manages multiprocessing state and parallel execution."""
    
    def __init__(self, *objects_with_mp_support):
        """
        Args:
            *objects_with_mp_support: Objects with push_multiprocessing() and pop_multiprocessing() methods
        """
        self.objects = objects_with_mp_support
        self.stashes = []
        self._is_pushed = False
    
    def push(self):
        """Save multiprocessing state from all objects."""
        if self._is_pushed:
            raise RuntimeError("State already pushed. Call pop() first.")
        
        for obj in self.objects:
            self.stashes.append(obj.push_multiprocessing())
        self._is_pushed = True
        return self
    
    def pop(self):
        """Restore multiprocessing state to all objects."""
        if not self._is_pushed:
            raise RuntimeError("State not pushed. Call push() first.")
        
        for obj, stash in zip(self.objects, self.stashes):
            obj.pop_multiprocessing(stash)
        
        self.stashes = []
        self._is_pushed = False
        return self
    
    @staticmethod
    def starmap(fn, arg_tuples, n_procs=None):
        """
        Execute function across multiple processes or sequentially.
        
        Args:
            fn: Function to execute
            arg_tuples: Iterable of argument tuples to pass to fn
            n_procs: Number of processes (None=all cores, <0=no multiprocessing)
        
        Returns:
            List of results from fn(*args) for each args in arg_tuples
        """
        if n_procs is None or n_procs >= 0:
            with Pool(processes=n_procs) as pool:
                results = pool.starmap(fn, arg_tuples)
        else:
            results = list(itertools.starmap(fn, arg_tuples))
        return results


def validate_multiprocessing_setup(use_multiprocessing, thread_vars=None):
    """Validate NumPy threading is disabled for multiprocessing."""
    if not use_multiprocessing:
        return
    
    if thread_vars is None:
        thread_vars = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                       "VECLIB_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
    
    if not all([os.environ.get(var, None) == "1" for var in thread_vars]):
        warnings.warn(
            f"Multiprocessing enabled but NumPy multithreading not disabled. "
            f"Disable by setting: {','.join(thread_vars)}"
        )
