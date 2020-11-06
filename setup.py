import os
from setuptools import setup, find_packages


# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="xray_diffraction",
    version="1.0.0",
    author="David Greving",
    author_email="david.greving@gmail.com",
    description=("Functionality for creating xray diffraction"),
    license="BSD",
    keywords="example documentation tutorial",
    url="",
    packages=find_packages(),
    long_description=read('README.md'),
    install_requires=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
