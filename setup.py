from setuptools import setup, find_packages

setup(
    name='elatentlpips',
    version='0.1.0',
    description='Minimal Unofficial PyTorch Implementation of E-Latent LPIPS',
    packages=find_packages(include=['elatentlpips']),
    author='Joonghyuk Shin',
    install_requires=[
        "torch",
        "scipy",
        "numpy",
        "ninja",
        "safetensors",
        "diffusers",
        'accelerate', 
    ],
    extras_require={
        "train":[
            'torchvision',
            'datasets', 
            'transformers', 
            'wandb', 
            'lpips',
        ]
    }
)