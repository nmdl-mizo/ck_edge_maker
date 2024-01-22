"""
Dataset of Organic C-K edge and corresponding structures in QM9 based on torch_geometric.dataset.qm9
"""

import os
from hashlib import blake2b, md5
import json
import torch
import h5py
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets.qm9 import QM9, conversion
from scipy.interpolate import interp1d


class CK(QM9):
    """
    Dataset of Organic C-K edge and corresponding structures
    """

    #raw_url = "https://figshare.com/ndownloader/files/31947896"  # site_spectra_0.5eV.hdf5

    def __init__(self, root, energies, site_spectra_hdf5_filename="site_spectra_0.5eV.hdf5",
                 mean_normalize=True, max_normalize=False, scale=None,
                 transform=None, directional=False):
        self.root = root
        self.site_spectra_hdf5_filename = site_spectra_hdf5_filename
        if isinstance(energies, np.ndarray):
            self.energies = energies
        elif isinstance(energies, (list, tuple)):
            if len(energies) != 3:
                raise RuntimeError(
                    "energies must be specified a list or tuple with length of 3,"
                    f" but the length of the given argument is {len(energies)}."
                )
            self.energies = np.linspace(*energies)
        self.transform = transform
        self.mean_normalize = mean_normalize
        self.max_normalize = max_normalize
        self.scale = scale
        self.directional = directional
        super().__init__(root=self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        with open(os.path.splitext(self.processed_paths[0])[0] + ".json", "rt") as f:
            self.process_dict = json.load(f)

    @property
    def raw_file_names(self):
        return super().raw_file_names

    @property
    def raw_hdf5_md5(self):
        checksum_md5 = md5()
        with open(os.path.join(self.raw_dir, self.site_spectra_hdf5_filename), 'rb') as f:
            for chunk in iter(lambda: f.read(2048 * checksum_md5.block_size), b''):
                checksum_md5.update(chunk)
        checksum_digest = checksum_md5.hexdigest()
        return checksum_digest

    @property
    def process_hash(self):
        return blake2b(repr((
            self.raw_hdf5_md5, self.energies, self.transform,
            self.mean_normalize, self.max_normalize, self.scale, self.directional)).encode("utf-8"),
            digest_size=5).hexdigest()

    @property
    def processed_file_names(self):
        process_hash = self.process_hash
        return [f"data_v3_site_{process_hash}.pt", ]

    def download(self):
        super().download()

    def process(self):
        import sys
        from tqdm import tqdm
        from torch_geometric.utils import one_hot, scatter
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            data_list = [
                Data(**data_dict, natoms=len(data_dict["z"]))
                for data_dict in data_list
            ]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

        else:
            types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
            bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

            with open(self.raw_paths[1], 'r') as f:
                target = [[float(x) for x in line.split(',')[1:20]]
                          for line in f.read().split('\n')[1:-1]]
                y = torch.tensor(target, dtype=torch.float)
                y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)
                y = y * conversion.view(1, -1)

            with open(self.raw_paths[2], 'r') as f:
                skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

            suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                       sanitize=False)

            data_list = []
            for i, mol in enumerate(tqdm(suppl)):
                if i in skip:
                    continue

                N = mol.GetNumAtoms()

                conf = mol.GetConformer()
                pos = conf.GetPositions()
                pos = torch.tensor(pos, dtype=torch.float)

                type_idx = []
                atomic_number = []
                aromatic = []
                sp = []
                sp2 = []
                sp3 = []
                num_hs = []
                for atom in mol.GetAtoms():
                    type_idx.append(types[atom.GetSymbol()])
                    atomic_number.append(atom.GetAtomicNum())
                    aromatic.append(1 if atom.GetIsAromatic() else 0)
                    hybridization = atom.GetHybridization()
                    sp.append(1 if hybridization == HybridizationType.SP else 0)
                    sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                    sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

                z = torch.tensor(atomic_number, dtype=torch.long)

                rows, cols, edge_types = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    rows += [start, end]
                    cols += [end, start]
                    edge_types += 2 * [bonds[bond.GetBondType()]]

                edge_index = torch.tensor([rows, cols], dtype=torch.long)
                edge_type = torch.tensor(edge_types, dtype=torch.long)
                edge_attr = one_hot(edge_type, num_classes=len(bonds))

                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_type = edge_type[perm]
                edge_attr = edge_attr[perm]

                row, col = edge_index
                hs = (z == 1).to(torch.float)
                num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

                x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
                x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                                  dtype=torch.float).t().contiguous()
                x = torch.cat([x1, x2], dim=-1)

                name = mol.GetProp('_Name')
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

                data = Data(
                    x=x,
                    z=z,
                    pos=pos,
                    edge_index=edge_index,
                    smiles=smiles,
                    edge_attr=edge_attr,
                    y=y[i].unsqueeze(0),
                    name=name,
                    idx=i,
                    natoms=len(z)
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        def get_ck_info_from_data_dict(data_dict, f_sp):
            id_mol = int(data_dict["name"].split("_")[-1])
            id_site = list(map(int, f_sp[f'{id_mol}'].keys()))
            n_site = len(id_site)
            n_edge = data_dict["edge_index"].shape[-1]
            n_node = data_dict["z"].shape[0]
            node_mask = torch.Tensor(
                [i in id_site for i in range(data_dict["z"].shape[0])]).bool()
            multiplicity = torch.Tensor(
                [
                    0 if i not in id_site else f_sp[f'{id_mol}/{i}'].attrs["multiplicity"]
                    for i in range(data_dict["z"].shape[0])
                ])
            return {"id_mol": id_mol, "id_site": id_site, "n_site": n_site, "n_edge": n_edge,
                    "n_node": n_node, "node_mask": node_mask, "multiplicity": multiplicity}

        def get_ck_spectra_from_data_dict(data_dict, f_sp):
            id_mol = int(data_dict["name"].split("_")[-1])
            spectra_list = []
            for i_site in range(data_dict["z"].shape[0]):
                try:
                    spectra_list.append(torch.from_numpy(
                        self._get_sp(id_mol, i_site, f_sp=f_sp)))
                except:
                    if not self.directional:
                        spectra_list.append(torch.zeros(len(self.energies)))
                    else:
                        spectra_list.append(torch.zeros(3, len(self.energies)))
            if not self.directional:
                return {"spectra": torch.vstack(spectra_list)}
            else:
                return {"spectra": torch.stack(spectra_list)}

        with h5py.File(
                os.path.join(self.raw_dir, self.site_spectra_hdf5_filename),
                mode="r", libver='latest', swmr=True) as f_sp:
            id_mol_ck = torch.tensor(
                list(map(int, sorted(f_sp.keys(), key=int))))
            data_temp_list = []
            for data_dict in data_list:
                if int(data_dict["name"].split("_")[-1]) in id_mol_ck:
                    ck_info = get_ck_info_from_data_dict(data_dict, f_sp)
                    spectra_dict = get_ck_spectra_from_data_dict(
                        {**data_dict.to_dict(), **ck_info}, f_sp)
                    d = Data(
                        **data_dict.to_dict(),
                        **ck_info,
                        **spectra_dict
                    )
                    data_temp_list.append(d)
        data_list = data_temp_list

        self.process_dict = {
            "directional": self.directional,
            "energies": self.energies.tolist()
        }
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        if self.mean_normalize:
            mean_temp_list = []
            for d in data_list:
                mean_temp = torch.mean(d.spectra, dim=-1)
                mean_temp_list.append(torch.mean(
                    mean_temp[torch.nonzero(mean_temp)]))
            self.sp_mean = torch.mean(torch.tensor(mean_temp_list))
            self.process_dict.update({
                "mean_normalize": self.mean_normalize,
                "sp_mean": self.sp_mean.item()
            })

        elif self.max_normalize:
            self.sp_max = torch.max([torch.max(d.spectra) for d in data_list])
            self.process_dict.update({
                "max_normalize": self.mean_normalize,
                "sp_max": self.sp_max.item()
            })

        if self.mean_normalize:
            for d in data_list:
                d.spectra /= self.sp_mean
        elif self.max_normalize:
            for d in data_list:
                d.spectra /= self.sp_max
        elif self.scale is not None:
            for d in data_list:
                d.spectra *= self.scale
            self.process_dict.update({
                "scale": self.scale,
            })
        data = self.collate(data_list)
        torch.save(data, self.processed_paths[0])
        with open(os.path.splitext(self.processed_paths[0])[0] + ".json", "wt") as f:
            json.dump(self.process_dict, f, indent=2)
        return

    def _get_sp(self, id_mol, id_site, f_sp):
        en = f_sp[f"{id_mol}/{id_site}/energies"][:]
        en_ex = f_sp[f"{id_mol}/{id_site}"].attrs["excitation_energy"]
        sp = f_sp[f"{id_mol}/{id_site}/spectrum"][:]
        if self.transform is None:
            if self.directional:
                sp_interp = np.array([
                    interp1d(
                        x=en + en_ex, y=y,
                        kind="linear", bounds_error=False, fill_value=0.
                    )(self.energies)
                    for y in sp[:, 1:].T
                ])
            else:
                y = sp[:, 0]
                sp_interp = interp1d(
                    x=en + en_ex, y=y,
                    kind="linear", bounds_error=False, fill_value=0.)(self.energies)
        else:
            sp_interp = self.transform(en, sp)
        return sp_interp.astype(np.float32)

    def get(self, idx):
        data = super().get(idx)
        return data

    def n_atom_list(self, z=None):
        """
        Get list of number of atoms in each molecular graph

        if z == None, count number of all elements
        if integer is specified, count number of specific elements with atomic number z
        """
        if z is None:
            return torch.stack([g.n_node for g in self])
        elif isinstance(z, int):
            return torch.stack([(g.z == z).sum() for g in self])
        else:
            raise RuntimeError(
                f"z must be None or int, but {type(z)} is specified.")
