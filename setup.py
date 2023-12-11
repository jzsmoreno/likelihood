import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="likelihood",  # Replace with your own username
    version="1.2.1",
    author="J. A. Moreno-Guerra",
    author_email="jzs.gm27@gmail.com",
    description="A package that performs the maximum likelihood algorithm.",
    py_modules=["likelihood"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jzsmoreno/likelihood/",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "corner",
        "numba",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
