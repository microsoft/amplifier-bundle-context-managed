"""Fixtures for the host-owned Boundary context manager."""

import pytest
from amplifier_module_context_managed.boundary import BoundaryContextManager


@pytest.fixture
def context_no_disk():
    return BoundaryContextManager()