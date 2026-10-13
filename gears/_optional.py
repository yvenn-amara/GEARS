"""
One place for the "optional dependency is missing" error.

Every feature that needs a package outside the base install calls
:func:`require` and so gets the same message, with the pip command that fixes it.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module_name: str, extra: str | None = None, *, feature: str | None = None) -> ModuleType:
    """
    Import and return an optional dependency, or raise a helpful ``ImportError``.

    Parameters
    ----------
    module_name : str
        Module to import, e.g. ``"torch"``.
    extra : str, optional
        Name of the GEARS extra that installs it (``"vae"``, ``"dl"``). The
        message then reads ``pip install "gears-ev[<extra>]"``; without an
        extra it reads ``pip install <module_name>``.
    feature : str, optional
        What needs the module, named in the message (e.g. ``"VAE session models"``).

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ImportError
        If the module (or a module it imports) is not installed. The original
        error is chained, so a broken sub-dependency is still visible.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        needed_for = f" (needed for {feature})" if feature else ""
        install = f'pip install "gears-ev[{extra}]"' if extra else f"pip install {module_name}"
        missing = "" if e.name == module_name else f" (missing module: '{e.name}')"
        raise ImportError(
            f"Missing optional dependency '{module_name}'{needed_for}{missing}. "
            f"Install it with: {install}"
        ) from e
