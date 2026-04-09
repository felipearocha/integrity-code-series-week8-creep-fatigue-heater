"""
Pytest configuration and shared fixtures for the Integrity Code Series Week 8 test suite.

This configuration ensures the src package is importable from tests.
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np


@pytest.fixture
def config():
    """Fixture providing access to config module."""
    from src import config
    return config


@pytest.fixture
def creep_engine():
    """Fixture providing access to creep_engine module."""
    from src import creep_engine
    return creep_engine


@pytest.fixture
def oxidation():
    """Fixture providing access to oxidation module."""
    from src import oxidation
    return oxidation


@pytest.fixture
def fatigue():
    """Fixture providing access to fatigue module."""
    from src import fatigue
    return fatigue


@pytest.fixture
def creep_fatigue():
    """Fixture providing access to creep_fatigue module."""
    from src import creep_fatigue
    return creep_fatigue


@pytest.fixture
def tube_model():
    """Fixture providing access to tube_model module."""
    from src import tube_model
    return tube_model


@pytest.fixture
def surrogate():
    """Fixture providing access to surrogate module."""
    from src import surrogate
    return surrogate


@pytest.fixture
def monte_carlo():
    """Fixture providing access to monte_carlo module."""
    from src import monte_carlo
    return monte_carlo


@pytest.fixture
def cybersecurity():
    """Fixture providing access to cybersecurity module."""
    from src import cybersecurity
    return cybersecurity
