"""
Utility functions for the Integrity Code Series Week 8 project.
"""

import numpy as np
import os


def hours_to_years(hours: float) -> float:
    """Convert hours to calendar years (8760 hr/yr)."""
    return hours / 8760.0


def years_to_hours(years: float) -> float:
    """Convert calendar years to hours."""
    return years * 8760.0


def kelvin_to_celsius(T_K: float) -> float:
    """Convert Kelvin to Celsius."""
    return T_K - 273.15


def celsius_to_kelvin(T_C: float) -> float:
    """Convert Celsius to Kelvin."""
    return T_C + 273.15


def pa_to_mpa(pressure_pa: float) -> float:
    """Convert Pascals to Megapascals."""
    return pressure_pa / 1.0e6


def mpa_to_pa(pressure_mpa: float) -> float:
    """Convert Megapascals to Pascals."""
    return pressure_mpa * 1.0e6


def m_to_mm(length_m: float) -> float:
    """Convert meters to millimeters."""
    return length_m * 1000.0


def mm_to_m(length_mm: float) -> float:
    """Convert millimeters to meters."""
    return length_mm / 1000.0


def m_to_inch(length_m: float) -> float:
    """Convert meters to inches."""
    return length_m / 0.0254


def inch_to_m(length_inch: float) -> float:
    """Convert inches to meters."""
    return length_inch * 0.0254


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def percentile_labels(data: np.ndarray,
                       percentiles=(5, 50, 95)) -> dict:
    """
    Compute labeled percentiles.

    Returns
    -------
    dict : percentile -> value
    """
    return {p: float(np.percentile(data, p)) for p in percentiles}
