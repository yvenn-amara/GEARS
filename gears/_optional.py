"""
Helper to manage optional dependencies for GEARS.
"""

from __future__ import annotations

def require(module_name: str, extra: str) -> None:
    """
    Verify that an optional dependency is installed.
    If not, raise an ImportError with a clear message on how to install it.
    """
    try:
        __import__(module_name)
    except ImportError:
        raise ImportError(
            f"The optional dependency '{module_name}' is required for this feature. "
            f"Please install it using: pip install \"gears-ev[{extra}]\""
        ) from None
