"""Workaround to ensure that files can be imported from the src directory

This workaround is necessary since Jupyter notebooks do not have the equivalent of
a __file__ attribute, so instead this module has to be imported before importing anything
from the src subdirectory.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
