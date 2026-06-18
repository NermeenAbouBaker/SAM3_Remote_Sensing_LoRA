"""
SAM3 LoRA — setup
"""

from setuptools import setup, find_packages

setup(
    name="sam3-lora",
    version="0.1.0",
    description="LoRA fine-tuning for SAM3",
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "pycocotools>=2.0.6",
        "scikit-image>=0.21.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=23.0.0"],
    },
)
