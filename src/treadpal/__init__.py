import os

# Before anything imports torch: OpenMP threads otherwise spin-wait between the
# beat detector's calls, which kept ~15 cores busy for a few percent of work
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

__version__ = "0.1.0"
