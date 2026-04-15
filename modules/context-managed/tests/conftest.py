"""
Test fixtures for context-managed module tests.

Provides three core fixtures:
- tmp_session_dir: temporary session directory (backed by pytest's tmp_path)
- context: ManagedContextManager with a real session directory for disk-backed tests
- context_no_disk: ManagedContextManager with session_dir=None for in-memory tests
"""

import pytest

from amplifier_module_context_managed import ManagedContextManager


@pytest.fixture
def tmp_session_dir(tmp_path):
    """Temporary session directory for disk-backed tests."""
    return tmp_path


@pytest.fixture
def context(tmp_session_dir):
    """ManagedContextManager backed by a temporary session directory."""
    return ManagedContextManager(session_dir=tmp_session_dir)


@pytest.fixture
def context_no_disk():
    """ManagedContextManager with no session directory (in-memory only)."""
    return ManagedContextManager(session_dir=None)
