## Pytest config

import pytest
import hypothesis

# Load the Hypothesis plugin here, because we disabled it in in pyproject.toml, in order
# to ensure that pytest-cov is loaded first.
# If we don't do this, coverage reports will be wrong.
# See: https://hypothesis.readthedocs.io/en/latest/strategies.html#interaction-with-pytest-cov
pytest_plugins = [
    "hypothesis.extra.pytestplugin",
]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--max-examples", default=None, type=int, help="Hypothesis max_examples setting.")


def pytest_configure(config: pytest.Config) -> None:
    test_max_examples = (
        config.getoption("--max-examples")
        or _getenv_positive_integer("TEST_MAX_EXAMPLES")
        or 100
    )
    hypothesis.settings.register_profile("dev", print_blob=True, max_examples=test_max_examples)
    hypothesis.settings.load_profile("dev")
    print(hypothesis.settings.get_profile("dev"))


## Helpers

import os
from typing import NewType

PositiveInt = NewType("PositiveInt", int)


def _convert_positive_integer(y: int) -> PositiveInt:
    if y <= 0:
        raise ValueError("A positive integer was expected.")
    return PositiveInt(y)


def _parse_positive_integer(x: str) -> PositiveInt:
    return _convert_positive_integer(int(x))


def _getenv_positive_integer(k: str) -> PositiveInt | None:
    if (val := os.environ.get(k)) is not None:
        return _parse_positive_integer(val)
    else:
        return None
