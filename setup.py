from setuptools import setup, find_packages

setup(
    name="math-toolkit",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
    ],
    python_requires=">=3.7",
    author="Josue Amaral",
    description="Comprehensive mathematical optimization and search library with binary search paradigm",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="optimization, binary-search, gradient-descent, adam, linear-systems",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)


