### Min
import os
import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Data, Dataset
import numpy as np
import random
from functools import lru_cache
import warnings

warnings.filterwarnings('ignore')

### Import advanced preprocessing
try:
    from advanced_preprocessing import AdvancedPreprocessor
    ADVANCED_PREPROCESSING_AVAILABLE = True
except ImportError:
    ADVANCED_PREPROCESSING_AVAILABLE = False
    print("Warning: Advanced preprocessing not available")

### Import enhanced SMILES processing
try:
    from featurize_smiles import smiles_to_features, augment_smiles, create_augmented_dataset
    ENHANCED_SMILES_AVAILABLE = True
except ImportError:
    ENHANCED_SMILES_AVAILABLE = False
    print("Warning: Enhanced SMILES processing not available")


### Convert SMILES to PyG graph with configurable node and edge features
def smiles_to_graph(smiles, feature_config=None):
    if feature_config is None:
        feature_config = {
            "atomic_num": True,
            "mass": True,
            "aromatic": True,
            "degree": True,
            "formal_charge": True,
            "num_hs": True,
            "hybridization": True,
            "bond_type": True,
            "bond_aromatic": True,
            "bond_ring": True,
        }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    atom_features = []
    for atom in mol.GetAtoms():
        features = []
        if feature_config.get("atomic_num", True):
            features.append(atom.GetAtomicNum())
        if feature_config.get("mass", True):
            features.append(atom.GetMass())
        if feature_config.get("aromatic", True):
            features.append(int(atom.GetIsAromatic()))
        if feature_config.get("degree", True):
            features.append(atom.GetDegree())
        if feature_config.get("formal_charge", True):
            features.append(atom.GetFormalCharge())
        if feature_config.get("num_hs", True):
            features.append(atom.GetTotalNumHs())
        if feature_config.get("hybridization", True):
            hyb = [0] * 6
            hyb_idx = int(atom.GetHybridization())
            if 0 <= hyb_idx < 6:
                hyb[hyb_idx] = 1
            features.extend(hyb)
        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_feats = []

        if feature_config.get("bond_type", True):
            bond_type = bond.GetBondType()
            bond_feats.extend([
                int(bond_type == Chem.rdchem.BondType.SINGLE),
                int(bond_type == Chem.rdchem.BondType.DOUBLE),
                int(bond_type == Chem.rdchem.BondType.TRIPLE),
                int(bond_type == Chem.rdchem.BondType.AROMATIC),
            ])

        if feature_config.get("bond_aromatic", True):
            bond_feats.append(int(bond.GetIsAromatic()))
        if feature_config.get("bond_ring", True):
            bond_feats.append(int(bond.IsInRing()))

        edge_index.append([start, end])
        edge_index.append([end, start])
        edge_attr.append(bond_feats)
        edge_attr.append(bond_feats)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


### Enhanced SMILES augmentation with multiple strategies
def augment_smiles_enhanced(smiles, strategy='random', max_attempts=10):
    if ENHANCED_SMILES_AVAILABLE:
        return augment_smiles(smiles, strategy, max_attempts)
    else:
        ### Fallback to basic augmentation
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        for _ in range(max_attempts):
            try:
                if strategy == 'random_rotation':
                    ### Random rotation around single bonds
                    mol_aug = Chem.RWMol(mol)
                    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
                    if rotatable_bonds > 0:
                        for _ in range(min(3, rotatable_bonds)):
                            bond_idx = random.choice([i for i in range(mol_aug.GetNumBonds())])
                            mol_aug.SetBond(bond_idx, mol_aug.GetBondWithIdx(bond_idx))
                        return Chem.MolToSmiles(mol_aug)

                elif strategy == 'stereo_flip':
                    ### Flip stereochemistry
                    mol_aug = Chem.RWMol(mol)
                    for atom in mol_aug.GetAtoms():
                        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                    return Chem.MolToSmiles(mol_aug)

                elif strategy == 'bond_order_change':
                    ### Change bond orders randomly
                    mol_aug = Chem.RWMol(mol)
                    bonds = list(mol_aug.GetBonds())
                    if bonds:
                        bond = random.choice(bonds)
                        if bond.GetBondType() == Chem.BondType.SINGLE:
                            mol_aug.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                            mol_aug.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.DOUBLE)
                        elif bond.GetBondType() == Chem.BondType.DOUBLE:
                            mol_aug.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                            mol_aug.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
                        return Chem.MolToSmiles(mol_aug)

                elif strategy == 'ring_break':
                    ### Break rings randomly
                    mol_aug = Chem.RWMol(mol)
                    rings = mol_aug.GetRingInfo().AtomRings()
                    if rings:
                        ring = random.choice(rings)
                        if len(ring) > 3:
                            atom1, atom2 = random.sample(ring, 2)
                            mol_aug.RemoveBond(atom1, atom2)
                            return Chem.MolToSmiles(mol_aug)

                elif strategy == 'substitution':
                    ### Substitute atoms randomly
                    mol_aug = Chem.RWMol(mol)
                    atoms = list(mol_aug.GetAtoms())
                    if atoms:
                        atom = random.choice(atoms)
                        if atom.GetSymbol() == 'C':
                            atom.SetAtomicNum(7)  # C -> N
                            return Chem.MolToSmiles(mol_aug)
                        elif atom.GetSymbol() == 'N':
                            atom.SetAtomicNum(8)  # N -> O
                            return Chem.MolToSmiles(mol_aug)

                elif strategy == 'addition':
                    ### Add small groups
                    mol_aug = Chem.RWMol(mol)
                    mol_aug.AddAtom(Chem.Atom('C'))
                    mol_aug.AddAtom(Chem.Atom('O'))
                    mol_aug.AddBond(mol_aug.GetNumAtoms() - 2, mol_aug.GetNumAtoms() - 1, Chem.BondType.SINGLE)
                    return Chem.MolToSmiles(mol_aug)

                elif strategy == 'deletion':
                    ### Delete terminal atoms
                    mol_aug = Chem.RWMol(mol)
                    atoms = list(mol_aug.GetAtoms())
                    terminal_atoms = [a for a in atoms if a.GetDegree() == 1]
                    if terminal_atoms:
                        atom = random.choice(terminal_atoms)
                        mol_aug.RemoveAtom(atom.GetIdx())
                        return Chem.MolToSmiles(mol_aug)

            except:
                continue

        return smiles


### Create augmented dataset with multiple strategies
def create_augmented_dataset_enhanced(smiles_list, augmentation_ratio=0.5, max_augmentations=3):
    if ENHANCED_SMILES_AVAILABLE:
        return create_augmented_dataset(smiles_list, augmentation_ratio, max_augmentations)
    else:
        ### Fallback to basic augmentation
        augmented_smiles = []
        original_smiles = []

        for smiles in smiles_list:
            augmented_smiles.append(smiles)
            original_smiles.append(smiles)

            if random.random() < augmentation_ratio:
                num_augmentations = random.randint(1, max_augmentations)
                for _ in range(num_augmentations):
                    strategy = random.choice(['random_rotation', 'stereo_flip', 'bond_order_change',
                                              'ring_break', 'substitution', 'addition', 'deletion'])
                    aug_smiles = augment_smiles_enhanced(smiles, strategy)
                    if aug_smiles != smiles and Chem.MolFromSmiles(aug_smiles) is not None:
                        augmented_smiles.append(aug_smiles)
                        original_smiles.append(smiles)

        return augmented_smiles, original_smiles


### Custom dataset for polymer graphs with advanced preprocessing and caching
class PolymerDataset(Dataset):
    ### Multi-task: predict Tg, FFV, Tc, Density, Rg
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

    def __init__(self, csv_file, feature_config=None, augmentation=False, augmentation_ratio=0.5,
                 root=None, transform=None, pre_transform=None, cache_graphs=True):
        self.csv_file = csv_file
        self.data_frame = pd.read_csv(csv_file)
        self.feature_config = feature_config
        self.augmentation = augmentation
        self.augmentation_ratio = augmentation_ratio
        self.cache_graphs = cache_graphs

        ### Initialize graph cache
        if self.cache_graphs:
            self.graph_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0

        ### Initialize advanced preprocessor
        if ADVANCED_PREPROCESSING_AVAILABLE:
            self.preprocessor = AdvancedPreprocessor()
        else:
            self.preprocessor = None

        ### Check if target columns exist
        self.has_targets = all(col in self.data_frame.columns for col in self.target_cols)
        if self.has_targets:
            self.means = self.data_frame[self.target_cols].mean()
            self.stds = self.data_frame[self.target_cols].std()
        else:
            self.means = None
            self.stds = None

        ### Apply advanced preprocessing if available
        if self.preprocessor is not None:
            self._apply_advanced_preprocessing()

        ### Create augmented data if augmentation is enabled
        if self.augmentation:
            self._create_augmented_data()

        super().__init__(root, transform, pre_transform)

    def _apply_advanced_preprocessing(self):
        ### Apply advanced data preprocessing
        if self.preprocessor is not None:
            ### Data quality checking
            quality_report = self.preprocessor.check_data_quality(self.data_frame)
            print(f"Data quality report: {quality_report}")

            ### Feature engineering
            self.data_frame = self.preprocessor.engineer_features(self.data_frame)

            ### Outlier detection and handling
            self.data_frame = self.preprocessor.handle_outliers(self.data_frame)

            ### Feature scaling
            if self.has_targets:
                self.data_frame = self.preprocessor.scale_features(self.data_frame, self.target_cols)

    def _create_augmented_data(self):
        ### Create augmented data for training
        print("Creating augmented dataset...")
        smiles_list = self.data_frame['SMILES'].tolist()
        augmented_smiles, original_smiles = create_augmented_dataset_enhanced(
            smiles_list, self.augmentation_ratio, max_augmentations=3
        )

        ### Create augmented dataframe
        aug_data = []
        for i, (aug_smiles, orig_smiles) in enumerate(zip(augmented_smiles, original_smiles)):
            if aug_smiles != orig_smiles:
                orig_row = self.data_frame[self.data_frame['SMILES'] == orig_smiles].iloc[0]
                aug_row = orig_row.copy()
                aug_row['SMILES'] = aug_smiles
                aug_row['is_augmented'] = True
                aug_data.append(aug_row)

        if aug_data:
            aug_df = pd.DataFrame(aug_data)
            self.data_frame = pd.concat([self.data_frame, aug_df], ignore_index=True)
            print(f"Added {len(aug_data)} augmented samples")
            print(f"Total dataset size: {len(self.data_frame)}")

    def len(self):
        return len(self.data_frame)

    def get(self, idx):
        smiles = self.data_frame.loc[idx, 'SMILES']

        ### Check cache first
        if self.cache_graphs and smiles in self.graph_cache:
            self.cache_hits += 1
            graph = self.graph_cache[smiles].clone()
        else:
            self.cache_misses += 1
            graph = smiles_to_graph(smiles, feature_config=self.feature_config)

            ### Cache the graph
            if self.cache_graphs:
                self.graph_cache[smiles] = graph.clone()

        ### If targets exist, process y and mask
        if self.has_targets:
            y_raw = self.data_frame.loc[idx, self.target_cols].values.astype(np.float32)
            mask = ~np.isnan(y_raw)

            ### Apply advanced normalization if available
            if self.preprocessor is not None:
                y_norm = self.preprocessor.normalize_targets(y_raw, self.means.values, self.stds.values)
            else:
                y_norm = np.where(mask, (y_raw - self.means.values) / self.stds.values, 0.0)

            graph.y = torch.tensor(y_norm, dtype=torch.float)
            graph.mask = torch.tensor(mask.astype(np.float32), dtype=torch.float)
        else:
            graph.y = None
            graph.mask = None

        return graph

    def __getitem__(self, idx):
        if isinstance(idx, list) or isinstance(idx, torch.Tensor) or isinstance(idx, np.ndarray):
            graphs = [self.get(i) for i in idx]
            return graphs
        else:
            return self.get(idx)

    def denormalize(self, y_norm):
        ### y_norm: shape [5] or [batch, 5]
        means = self.means.values if self.means is not None else None
        stds = self.stds.values if self.stds is not None else None

        if means is not None and stds is not None:
            if self.preprocessor is not None:
                return self.preprocessor.denormalize_targets(y_norm, means, stds)
            else:
                return y_norm * torch.tensor(stds, dtype=torch.float) + torch.tensor(means, dtype=torch.float)
        else:
            return y_norm

    def get_feature_stats(self):
        ### Get feature statistics
        if self.preprocessor is not None:
            return self.preprocessor.get_feature_statistics(self.data_frame)
        return None

    def get_data_quality_report(self):
        ### Get comprehensive data quality report
        if self.preprocessor is not None:
            return self.preprocessor.get_quality_report(self.data_frame)
        return None

    def get_cache_stats(self):
        ### Get cache performance statistics
        if self.cache_graphs:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
            return {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': hit_rate,
                'cache_size': len(self.graph_cache)
            }
        return None

    def clear_cache(self):
        ### Clear graph cache to free memory
        if self.cache_graphs:
            self.graph_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            print("Graph cache cleared")

    def set_cache_enabled(self, enabled=True):
        ### Enable or disable graph caching
        self.cache_graphs = enabled
        if not enabled:
            self.clear_cache()

    def get_augmentation_stats(self):
        ### Get augmentation statistics
        if 'is_augmented' in self.data_frame.columns:
            aug_count = self.data_frame['is_augmented'].sum()
            total_count = len(self.data_frame)
            return {
                'total_samples': total_count,
                'augmented_samples': aug_count,
                'augmentation_ratio': aug_count / total_count if total_count > 0 else 0
            }
        return None
### #%#