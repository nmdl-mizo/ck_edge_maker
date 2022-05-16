# C-K edge maker

[![DOI](https://zenodo.org/badge/447862253.svg)](https://zenodo.org/badge/latestdoi/447862253)

A dedicated Python script for making smeared C-K edge spectra from the hdf5 dataset of eigenvalues and dynamical structure factors

## Overview

- Some functions for making Gaussiean smeared spectra database from the hdf5 dataset of eigenvalues and dynamical structure factors
- Commandline interface `ck_edge_maker`

## Reference

The hdf5 dataset of eigenvalues and dynamical structure factors will be available along with a data paper

[1] [Shibata, K., Kikumasa, K., Kiyohara, S., T. Mizoguchi, "Simulated carbon K edge spectral database of organic molecules" *Sci Data* **9**, 214 (2022)](https://doi.org/10.1038/s41597-022-01303-8).

## Install

Clone this repository and run `pip install`:

``` bash
$ git clone https://github.com/nmdl-mizo/ck_edge_maker.git
$ cd ck_edge_maker
$ pip intall .
```

or directry run `pip install`:

``` bash
$ pip install git+https://github.com/nmdl-mizo/ck_edge_maker
```

To uninstall, use pip:

``` bash
$ pip uninstall ck-edge-maker
```

## Usage

Calculate C-K edge spectra with Gaussian smearing from hdf5 spectral dataset

- positional arguments:
  - `{site,mol}`            calculation task, site or mol
  - `input`                 path to the input database hdf5 file
  - `output`                path to the output hdf5 file

- optional arguments:
  - `-h, --help`            show this help message and exit
  - `--sigma SIGMA`         Gaussian smearing parameter in eV. default 0.3 eV
  - `--margin MARGIN`       margin in eV for sampling energy range. ignored when both min and max is specified. default 3.0 eV
  - `--res RES`             resolution in eV for sampling energy range, default 0.1 eV
  - `--min MIN`             minimum energy in eV for sampling energy range
  - `--max MAX`             maximum energy in eV for sampling energy range
  - `--sum`                 calculate sum of dynamic structure factors for molecular spectra if specified. calculate weighted average if not specified. only valid when task = mol
  - `--nototal`             do not include total spectra if specified
  - `-f, --forceoverwrite`  Overwrite output file if specified
 
 
## Examples

First, prepare the hdf5 dataset of eigenvalues and dynamical structure factors.
The following is an example where the dataset is stored in the current directory with the name "site_eigen_dsf.hdf5".

```bash
# Create a hdf5 of molecular spectral dataset named "mol_spectra_0.5eV.hdf5" with 0.5eV Gaussian smearing, 0.1eV sampling step, and 5eV margin
ck_edge_maker mol site_eigen_dsf.hdf5 mol_spectra_0.5eV.hdf5 -f --sigma 0.5 --res 0.1 --margin 5
```

```bash
# Create a hdf5 of site specific spectral dataset named "site_spectra_0.5eV.hdf5" with 0.5eV Gaussian smearing, 0.1eV sampling step, and 5eV margin
ck_edge_maker site site_eigen_dsf.hdf5 site_spectra_0.5eV.hdf5 -f --sigma 0.5 --res 0.1 --margin 5
```

## Requirements

- python (=>3)
- h5py
- numpy
- tqdm

## Author

- kiyou
- Mizoguchi Lab.

## Licence

The source code is licensed MIT.
