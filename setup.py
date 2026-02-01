"""
Setup configuration for h2_plant package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / 'README.md'
long_description = readme_path.read_text() if readme_path.exists() else ''

setup(
    name='h2_plant',
    version='2.0.0',
    description='Modular dual-path hydrogen production plant simulation system',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Hydrogen Production Team',
    author_email='team@h2plant.example.com',

    packages=find_packages(exclude=['tests', 'examples', 'docs']),

    install_requires=[
        'numpy>=1.21.0',
        'numba>=0.55.0',
        'pyyaml>=6.0',
        'jsonschema>=4.0.0',
        'h5py>=3.0.0',  # Optional: for HDF5 checkpoints
    ],

    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'mypy>=0.990',
            'black>=22.0.0',
            'flake8>=5.0.0',
        ],
        'coolprop': [
            'CoolProp>=6.4.0',  # For LUT generation
        ],
        'viz': [
            'matplotlib>=3.5.0',
            'plotly>=5.0.0',
        ]
    },

    entry_points={
        'console_scripts': [
            'h2-simulate=h2_plant.simulation.runner:main',
        ],
    },

    python_requires='>=3.9',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)