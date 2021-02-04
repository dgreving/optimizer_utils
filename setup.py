import os
from setuptools import setup, find_packages


# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="optimizer_utils",
    version="0.9.9",
    author="David Greving",
    author_email="david.greving@gmail.com",
    description=("Functionality for parameter optimization"),
    license="BSD",
    keywords="",
    url="",
    packages=find_packages(),
    long_description=read('README.md'),
    install_requires=['numpy', 'scipy'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
