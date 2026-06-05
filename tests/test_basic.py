
import pyPyramid
import numpy as np

# Pytest will discover and run all test functions named `test_*` or `*_test`.

def test_version():
    """ check pyPyramid exposes a version attribute """
    assert hasattr(pyPyramid, "__version__")
    assert isinstance(pyPyramid.__version__, str)

