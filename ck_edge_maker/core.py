#!/usr/bin/env python
"""
A script for making spectra from the hdf5 dataset of eigenvalues and dynamical structure factors
"""

import numpy as np

def get_site_spectrum_set(site_group):
    """
    Get excitation energy, multiplicity, and pairs of eigenvalues and dynamical structure factors from a site group in the dataset

    Parameters
    --------
    mol_group : hdf5 group
        a molecule group in the hdf5 database

    Returns
    --------
    dict
        dictionary of excitation energy, multiplicity, eigenvalues and dynamical structure factors
    """
    ev = site_group["eigenvalues"][()]
    dsf = site_group["dsf"][()]
    nele = site_group.attrs["n_electron"]
    multiplicity = site_group.attrs["multiplicity"]
    exen = site_group.attrs["excitation_energy"]
    ev_valid_index = (ev - ev[(nele-1)//2]) >= 0.
    ev_valid = ev[ev_valid_index]
    ev_valid -= ev_valid[0]
    dsf_valid = dsf[ev_valid_index]
    return {"dsf": dsf_valid, "ev":ev_valid, "exen": exen, "multiplicity":multiplicity}


def get_mol_spectrum_set(mol_group, average=True):
    """
    Get pairs of eigenvalues and dynamical structure factors from a molecule group in the dataset

    Parameters
    --------
    mol_group : hdf5 group
        a molecule group in the hdf5 database
    average : bool, default True
        whether to calculate weighted average (True) or sum (False) of dynamic structure factors

    Returns
    --------
    dict
        dictionary of eigenvalues and dynamical structure factors
    """
    mol_dsf_list = list()
    mol_ev_list = list()
    multiplicity_sum = 0.
    for _, site_group in mol_group.items():
        sss = get_site_spectrum_set(site_group)
        mol_dsf_list.append(sss["dsf"] * sss["multiplicity"])
        mol_ev_list.append(sss["ev"] + sss["exen"])
        multiplicity_sum += sss["multiplicity"]
    mol_ev = np.concatenate(mol_ev_list)
    mol_dsf = np.concatenate(mol_dsf_list)
    if average:
        mol_dsf /= multiplicity_sum
    return {"dsf": mol_dsf[mol_ev.argsort()], "ev": mol_ev[mol_ev.argsort()]}


def get_smeared_intensity(ss, sigma=0.3, energies=None, e_margin=1., e_resolution=1e-3, include_total=True):
    """
    Get smeared spectral intensity from spectrum set

    Parameters
    --------
    ss : dict
        dictionary returned by get_mol_spectrum_set() or get_site_spectrum_set()
    sigma : float, default 0.3
        standard deviation for Gaussian smearing
    energies : array-like or None, default None
        if None specified, sampling with step of e_resolution (eV) with margin of e_margin (eV)
    e_margin : float
        margin in eV for sampling energy range. Only affects when None is specified for energies.
    e_resolution : float
        resolution in eV for sampling energy range. Only affects when None is specified for energies.
    include_total : bool, default True
        calculate total spectrum by averaging over the three spectra and include

    Returns
    --------
    dict
        dictionary contains energies and smeared intensity
    """
    if energies is None:
        energies = np.arange(
            np.floor(np.min(ss["ev"]) - e_margin),
            np.ceil(np.max(ss["ev"]) + e_margin),
            e_resolution,
        )
    gaussian = lambda x, c, w, s: w / (np.sqrt(2*np.pi)*s) * np.exp(-((x-c)/s)**2/2)
    smeared_intensity = np.array([np.sum([gaussian(energies, e, w, sigma) for e, w in zip(ss["ev"], dsf)], axis=0) for dsf in ss["dsf"].T]).T
    smeared_intensity *= 2. # spin degree of freedom
    if include_total:
        smeared_intensity = np.hstack([np.mean(smeared_intensity, axis=-1).reshape(-1, 1), smeared_intensity])
    return {"energies": energies, "smeared_intensity": smeared_intensity}
