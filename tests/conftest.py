import sys
import os

file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir)

# Only some tests use the arch (architecture of intel or ibm) command line parameter
# https://stackoverflow.com/questions/40880259/how-to-pass-arguments-in-pytest-by-command-line
def pytest_addoption(parser):
    parser.addoption("--arch", action="store", default="Architecture (intel or ibm)")


