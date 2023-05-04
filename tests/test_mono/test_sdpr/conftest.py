import pytest
import os
import sys
from test_mono.utils import read_yaml_params

# metafunc is an artifact of pytest
def pytest_generate_tests(metafunc):
    # check if params is a requested fixture
    if "params" in metafunc.fixturenames:
        # list of parameters from yaml files
        file_dir = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.join(file_dir, "sdpr_*.yaml")
        metafunc.parametrize("params", read_yaml_params(yaml_path))
