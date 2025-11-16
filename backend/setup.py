# setup.py
from setuptools import setup, find_packages

setup(
    name="hilbert_information_laboratory",
    version="0.1.0",
    description="Hilbert Information Laboratory backend and analysis tools",
    author="Swift Fox",
    packages=find_packages(
        exclude=(
            "tests",
            "docs",
            "frontend",
            "node_modules",
            "dist",
            "build",
        )
    ),
    python_requires=">=3.10",
)
