from setuptools import setup, find_packages

setup(
    name="freqdetr",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[],
    description="FreqDETR: baseline scaffold (Transformer-based) for corn lesion detection",
)