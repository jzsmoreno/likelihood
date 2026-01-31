from setuptools import find_packages, setup
from setuptools_rust import Binding, RustExtension

setup(
    packages=find_packages(),
    rust_extensions=[
        RustExtension(
            "likelihood.rust_py_integration",
            path="Cargo.toml",
            binding=Binding.PyO3,
        )
    ],
    zip_safe=False,
)
