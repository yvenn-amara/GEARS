"""
Evaluation utilities shared across GEARS benchmarking work.

Currently home to the last-N-occurrences windowing primitive used by both
the persistence-bootstrap sampler and the windowed GMM fit in the
persistence-vs-GMM rolling-origin benchmark (see
``gears/models/persistence_sampler.py`` and Session 3 of the benchmark
prompt).
"""

from gears.evaluation.windowing import sessions_in_last_n_occurrences

__all__ = ["sessions_in_last_n_occurrences"]
