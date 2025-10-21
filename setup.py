from setuptools import setup, find_packages

setup(
    name="freqdetr",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.1",
        "torchvision>=0.16",
        "opencv-python",
        "albumentations",
        "pyyaml",
        "timm",
        "einops",
        "numpy",
        "tqdm",
        "tensorboard",
    ],
    description="FreqDETR: baseline scaffold (Transformer-based) for corn lesion detection",
)