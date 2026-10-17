"""Keep live Reasoning State generation disabled in offline tests."""

import os
import pytest

@pytest.fixture(autouse=True)
def _disable_online_reasoning_state_in_unit_tests():
    keys = ("EVOMEMSTATE_REASONING_STATE_ENABLED", "MEMSYCO_REASONING_STATE_ENABLED")
    saved = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
