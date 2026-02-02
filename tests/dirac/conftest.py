"""Dirac test fixtures — skip all tests when QCI_TOKEN is not set."""

import os
import pytest


def pytest_collection_modifyitems(config, items):
    token = os.environ.get("QCI_TOKEN")
    if not token:
        skip = pytest.mark.skip(reason="QCI_TOKEN not set")
        for item in items:
            if "dirac" in str(item.fspath):
                item.add_marker(skip)
