from setuptools import setup, find_packages

setup(
    name='orca_pipeline',
    version='0.1.0',
    description='A pipeline for NEB optimization using ORCA.',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'run_pipeline=orca_pipeline.scripts.run_pipeline:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
