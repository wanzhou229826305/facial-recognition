from setuptools import setup, find_packages

setup(
    name='workshop',
    version='0.2',
    packages=find_packages(exclude=['*test*']),
    install_requires=[
        # List your project's dependencies here
    ],
)
