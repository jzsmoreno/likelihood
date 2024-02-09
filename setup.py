from pathlib import Path

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / "likelihood"
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version

setuptools.setup(
    name="likelihood",  # Replace with your own username
    version=about["__version__"],
    author="J. A. Moreno-Guerra",
    author_email="jzs.gm27@gmail.com",
    description="A package that performs the maximum likelihood algorithm.",
    py_modules=["likelihood"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jzsmoreno/likelihood/",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy<2.0.0",
        "matplotlib",
        "corner",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
