from setuptools import setup, find_packages

setup(
    name="xyz-std",
    version="0.1.0",
    author="Dejun Hu",
    author_email="hudejun2002@gmail.com",
    description="Standardize atom ordering in XYZ files using InChI canonical order and 3D-aware hydrogen ordering",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "rdkit>=2023.03.1",
        "numpy>=1.21.0",
        # openbabel must be installed via conda: conda install openbabel -c conda-forge
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
