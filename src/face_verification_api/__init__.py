# read version from installed package
from importlib.metadata import version
__version__ = version("face_verification_api")

# Path: face_verification_api.py

from .face_verification_api import app
