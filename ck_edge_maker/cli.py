#!/usr/bin/env python
"""
A script for CLI interface of ck_edge_maker
"""
from ck_edge_maker import *
import sys
import os
import argparse
from tqdm import tqdm


def main():
    """CLI interface for ck_edge_maker"""
    ap = argparse.ArgumentParser(description="Calculate C-K edge spectra with Gaussian smearing from hdf5 spectral dataset")
    ap.add_argument("task", type=str, choices=["site", "mol"], help="calculation task, site or mol")
    ap.add_argument("input", type=str, help="path to the input database hdf5 file")
    ap.add_argument("output", type=str, help="path to the output hdf5 file")
    ap.add_argument("--sigma", type=float, default=0.3, help="Gaussian smearing parameter in eV. default 0.3 eV")
    ap.add_argument("--margin", type=float, default=3.0, help="margin in eV for sampling energy range. ignored when both min and max is specified. default 3.0 eV")
    ap.add_argument("--res", type=float, default=0.1, help="resolution in eV for sampling energy range, default 0.1 eV")
    ap.add_argument("--min", type=float, help="minimum energy in eV for sampling energy range")
    ap.add_argument("--max", type=float, help="maximum energy in eV for sampling energy range")
    ap.add_argument("--sum", action="store_false", help="calculate sum of dynamic structure factors for molecular spectra if specified. calculate weighted average if not specified. only valid when task = mol")
    ap.add_argument("--nototal", action="store_false", help="do not include total spectra if specified")
    ap.add_argument("-f", "--forceoverwrite", action="store_true", help="Overwrite output file if specified")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"{args.input} not found.")
    if os.path.isfile(args.output) and not args.forceoverwrite:
        raise FileExistsError(f"{args.output} already exists. Abort to avoid overwriting. Please remove {args.output} and retry if you want.")
    if args.min is not None and args.max is not None:
        energies = np.arange(args.min, args.max, args.res)
        print(
            f"""
            Start calculating smeared spectra with:
            task = {args.task}
            input filename = {args.input}
            output filename = {args.output}
            sigma = {args.sigma} eV
            e_min = {args.min} eV
            e_max = {args.max} eV
            e_resolution = {args.res} eV
            with total = {args.nototal}
            """
        )
    else:
        energies = None
        print(
            f"""
            Start calculating smeared spectra with:
            task = {args.task}
            input filename = {args.input}
            output filename = {args.output}
            sigma = {args.sigma} eV
            e_margin = {args.margin} eV
            e_resolution = {args.res} eV
            with total = {args.nototal}
            """
        )

    with h5py.File(args.input, "r") as f_in, h5py.File(args.output, "w") as f_out:
        for mol_id, f_in_mol in tqdm(sorted(f_in.items(), key=lambda x: int(x[0]))):
#            if int(mol_id) > 100:
#                continue
            f_out_mol = f_out.create_group(f"/{mol_id}")
            if args.task == "mol":
                mss = get_mol_spectrum_set(f_in_mol, average=args.sum)
                msi = get_smeared_intensity(
                    mss,
                    energies=energies,
                    sigma=args.sigma,
                    e_margin=args.margin,
                    e_resolution=args.res,
                    include_total=args.nototal
                )
                f_out_mol.create_dataset(
                    name="energies",
                    data=msi["energies"],
                    compression="gzip",
                    compression_opts=9
                )
                f_out_mol.create_dataset(
                    name="spectrum",
                    data=msi["smeared_intensity"],
                    compression="gzip",
                    compression_opts=9
                )
            elif args.task == "site":
                for site_id, f_in_site in sorted(f_in_mol.items(), key=lambda x: int(x[0])):
                    f_out_site = f_out_mol.create_group(f"{site_id}")
                    sss = get_site_spectrum_set(f_in_site)
                    ssi = get_smeared_intensity(
                        sss,
                        energies=energies,
                        sigma=args.sigma,
                        e_margin=args.margin,
                        e_resolution=args.res,
                        include_total=args.nototal
                    )
                    f_out_site.create_dataset(
                        name="energies",
                        data=ssi["energies"],
                        compression="gzip",
                        compression_opts=9
                    )
                    f_out_site.create_dataset(
                        name="spectrum",
                        data=ssi["smeared_intensity"],
                        compression="gzip",
                        compression_opts=9
                    )
                    for param in ["multiplicity", "final_energy_gs", "final_energy_ex", "excitation_energy"]:
                        f_out_site.attrs[param] = f_in_site.attrs[param]


if __name__ == "__main__":
    main()