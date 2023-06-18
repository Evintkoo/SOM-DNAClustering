import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="som-dna-clustering",
    version="0.1.0",
    description="A DNA clustering with implementation of Self Organizing Map",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Evintkoo/SOM-DNAClustering", #add the github link later - normal link
    author="Evint Leovonzko",
    author_email="evint.koo@gmail.com",
    license="None",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    packages=["som_dna_clustering"],
    include_package_data=True,
    install_requires=["numpy", "sklearn", "pandas"],
)