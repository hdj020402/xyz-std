"""
XYZ Standardization Tools

A toolkit for standardizing atom ordering in XYZ files using InChI canonical
order and 3D-aware hydrogen ordering.

Features:
- XYZ format parsing and formatting
- InChI-based canonical atom ordering
- 3D-aware hydrogen ordering on heavy atoms
- OpenBabel and RDKit backend support
"""

__version__ = "0.1.0"
__author__ = "Dejun Hu"
__email__ = "hudejun2002@gmail.com"
__license__ = "MIT"

from typing import Literal

from xyz_std.io import (
    xyz_to_rdkit_mol,
    xyz_to_symbols_coords,
    format_xyz,
    write_multi_xyz,
    standardize_xyz,
)
from xyz_std.atom_order import get_standard_atom_order

# Log level type
LogLevel = Literal["none", "error", "warning", "info", "debug"]

# Current log level (module-level state)
_current_log_level: LogLevel = "error"


def set_log_level(level: LogLevel) -> None:
    """
    Set the log level for RDKit and OpenBabel libraries.

    Args:
        level: Log level, one of 'none', 'error', 'warning', 'info', 'debug'.
            - 'none': Disable all logging output
            - 'error': Show only errors
            - 'warning': Show warnings and errors
            - 'info': Show info, warnings, and errors
            - 'debug': Show all log messages

    Example:
        >>> from xyz_std import set_log_level
        >>> set_log_level('none')  # Silence all logs
        >>> set_log_level('error')  # Show only errors (default)
    """
    global _current_log_level

    level = level.lower()
    valid_levels = ("none", "error", "warning", "info", "debug")
    if level not in valid_levels:
        raise ValueError(f"Invalid log level '{level}'. Must be one of: {valid_levels}")

    _current_log_level = level

    # Configure RDKit log level
    from rdkit import RDLogger
    logger = RDLogger.logger()

    if level == "none":
        RDLogger.DisableLog('rdApp.*')
    elif level == "error":
        logger.setLevel(RDLogger.ERROR)
    elif level == "warning":
        logger.setLevel(RDLogger.WARNING)
    elif level == "info":
        logger.setLevel(RDLogger.INFO)
    elif level == "debug":
        logger.setLevel(RDLogger.DEBUG)

    # Configure OpenBabel log level
    from openbabel import openbabel

    # OpenBabel log levels: 0=Error, 1=Warning, 2=Info, 3=Debug
    level_map = {
        "error": 0,
        "warning": 1,
        "info": 2,
        "debug": 3,
    }

    if level == "none":
        openbabel.obErrorLog.StopLogging()
    else:
        openbabel.obErrorLog.SetOutputLevel(level_map[level])


def get_log_level() -> LogLevel:
    """
    Get the current log level.

    Returns:
        Current log level string.
    """
    return _current_log_level


# Set default log level on import
set_log_level("error")

__all__ = [
    "standardize_xyz",
    "get_standard_atom_order",
    "xyz_to_rdkit_mol",
    "xyz_to_symbols_coords",
    "format_xyz",
    "write_multi_xyz",
    "set_log_level",
    "get_log_level",
]
