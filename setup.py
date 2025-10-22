"""
Setup script for eval-ai-library.
For modern installation, use pyproject.toml with pip install.
"""
from setuptools import setup

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="eval-ai-library",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
