from setuptools import setup, find_packages

print("find_packages:", find_packages())
setup(
    name='ck_edge_maker',
    version="1.1.0",
    description="A script for making spectra from the hdf5 dataset of eigenvalues and dynamical structure factors",
    long_description="A script for making spectra from the hdf5 dataset of eigenvalues and dynamical structure factors",
    url='https://github.com/nmdl-mizo/ck_edge_maker',
    author='kiyou, nmdl-mizo',
    author_email='',
    license='MIT',
    classifiers=[
        # https://pypi.python.org/pypi?:action=list_classifiers
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
    ],
    keywords='tools',
    install_requires=["numpy", "h5py", "tqdm"],
    extras_require={
        'pyg': ['scipy', 'torch', 'torch_geometric'],
    },
    packages=find_packages(),
    entry_points={
        'console_scripts':[
            'ck_edge_maker = ck_edge_maker.cli:main',
        ],
    },
)
