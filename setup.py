"""Setup script for module

To install, use

    python -m pip install .

or, for an editable install,

    python -m pip install --editable .

"""

from setuptools import setup

long_description = """
Python classes for implementing Lagrangian neural networks with built-in conservation laws.

For further details and information on how to use this module, see README.md
"""

# Extract requirements from requirements.txt file
with open("requirements.txt", "r", encoding="utf8") as f:
    requirements = [line.strip() for line in f.readlines()]

# Run setup
setup(
    name="conservative_nn",
    author="Eike Mueller",
    author_email="e.mueller@bath.ac.uk",
    description="Lagrangian neural networks with built-in conservation laws",
    long_description=long_description,
    version="1.0.0",
    packages=["conservative_nn"],
    package_dir={"": "src"},
    package_data={"conservative_nn": ["random_normal_table.json"]},
    install_requires=[
        'importlib-metadata; python_version == "3.8"',
    ]
    + requirements,
    url="https://github.com/eikehmueller/mlconservation_code",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: POSIX :: Linux",
    ],
)
