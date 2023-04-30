from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Counterfactual Generation Package'
LONG_DESCRIPTION = 'A package that makes it easy generate counterfactuals in a fast manner'

setup(
    name="FastCG",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Linoy Cohen",
    author_email="colinoy@gmail.com",
    license='MIT',
    packages=find_packages(),
    install_requires=[],
    keywords='conversion',
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)
