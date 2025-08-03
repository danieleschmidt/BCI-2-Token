"""
Setup script for BCI-2-Token package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "BCI-2-Token: Brain-Computer Interface to Token Translator"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [
            line.strip() 
            for line in fh.readlines() 
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "transformers>=4.20.0",
        "tqdm>=4.64.0"
    ]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "isort>=5.10.0",
        "flake8>=5.0.0",
    ],
    "devices": [
        "pyserial>=3.5",
        "pylsl>=1.16.0",
    ],
    "privacy": [
        "opacus>=1.4.0",
    ],
    "signal": [
        "mne>=1.5.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
    ],
    "realtime": [
        "pyserial>=3.5",
        "pylsl>=1.16.0",
        "mne>=1.5.0",
    ],
    "all": [
        "mne>=1.5.0",
        "pyserial>=3.5", 
        "pylsl>=1.16.0",
        "opacus>=1.4.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "isort>=5.10.0",
        "flake8>=5.0.0",
    ]
}

setup(
    name="bci2token",
    version="0.1.0",
    author="Terragon Labs",
    author_email="contact@terragonlabs.com",
    description="Brain-Computer Interface to Token Translator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terragon-labs/bci-2-token",
    project_urls={
        "Bug Tracker": "https://github.com/terragon-labs/bci-2-token/issues",
        "Documentation": "https://bci2token.readthedocs.io/",
        "Source Code": "https://github.com/terragon-labs/bci-2-token",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "bci2token=bci2token.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "brain-computer interface",
        "BCI",
        "EEG",
        "ECoG", 
        "neural decoding",
        "language models",
        "LLM",
        "tokenization",
        "differential privacy",
        "real-time processing"
    ],
)