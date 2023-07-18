from setuptools import setup, find_packages

VERSION = '0.0.4'
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
    install_requires=["ipython==8.10.0",
                      "joblib==1.2.0",
                      "matplotlib==3.7.0",
                      "numpy==1.23.5",
                      "pandas==1.5.3",
                      "scikit_learn==1.2.1",
                      "scipy==1.10.1",
                      "setuptools==65.5.0",
                      "tqdm==4.64.1"],
    keywords='conversion',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)
