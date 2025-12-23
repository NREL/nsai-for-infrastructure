from nsai_experiments.general_az_1p.setup_utils import (
    disable_numpy_multithreading,
    use_deterministic_cuda,
)


def pytest_sessionstart(session):
    disable_numpy_multithreading()
    use_deterministic_cuda()
