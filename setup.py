from pathlib import Path

import setuptools

# Parse the requirements.txt file
with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / "likelihood"
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version

setuptools.setup(
    name="likelihood",
    version=about["__version__"],
    author="J. A. Moreno-Guerra",
    author_email="jzs.gm27@gmail.com",
    maintainer="Jafet Castañeda",
    maintainer_email="jafetcc17@gmail.com",
    description="A package that performs the maximum likelihood algorithm.",
    py_modules=["likelihood"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jzsmoreno/likelihood/",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    extras_require={
        "full": ["networkx", "pyvis", "tensorflow==2.15.0", "keras-tuner", "scikit-learn"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
