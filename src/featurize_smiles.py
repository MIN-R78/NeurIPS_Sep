### Min
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
import random

### Define enhanced molecular descriptors (15 total)
descriptor_names = [
    'MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
    'RingCount', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAromaticRings',
    'NumSaturatedRings', 'FractionCsp3', 'NumHeteroatoms', 'NumAmideBonds'
]

### Define SMILES augmentation strategies
augmentation_strategies = [
    'random_rotation', 'stereo_flip', 'bond_order_change', 'ring_break',
    'substitution', 'addition', 'deletion'
]


def calc_descriptors(mol):
    ### Calculate enhanced descriptors for a molecule
    try:
        desc = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NHOHCount(mol),
            Descriptors.NOCount(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.FractionCsp3(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.NumAmideBonds(mol)
        ]
        return desc
    except:
        return [0.0] * len(descriptor_names)


def augment_smiles(smiles, strategy='random', max_attempts=10):
    ### Augment SMILES string using various strategies
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


def create_augmented_dataset(smiles_list, augmentation_ratio=0.5, max_augmentations=3):
    ### Create augmented dataset with multiple strategies
    augmented_smiles = []
    original_smiles = []

    for smiles in smiles_list:
        augmented_smiles.append(smiles)
        original_smiles.append(smiles)

        if random.random() < augmentation_ratio:
            num_augmentations = random.randint(1, max_augmentations)
            for _ in range(num_augmentations):
                strategy = random.choice(augmentation_strategies)
                aug_smiles = augment_smiles(smiles, strategy)
                if aug_smiles != smiles and Chem.MolFromSmiles(aug_smiles) is not None:
                    augmented_smiles.append(aug_smiles)
                    original_smiles.append(smiles)

    return augmented_smiles, original_smiles


def smiles_to_features(smiles, use_augmentation=False):
    ### Convert SMILES to descriptors and Morgan fingerprint with optional augmentation
    if use_augmentation and random.random() < 0.3:
        strategy = random.choice(augmentation_strategies)
        smiles = augment_smiles(smiles, strategy)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    desc = calc_descriptors(mol)
    fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
    fp_bits = list(fp)

    ### Add graph-level features (7 total)
    graph_features = [
        mol.GetNumAtoms(),
        mol.GetNumBonds(),
        len(mol.GetRingInfo().AtomRings()),
        Descriptors.FractionCsp3(mol),
        Descriptors.NumRotatableBonds(mol) / max(1, mol.GetNumAtoms()),
        Descriptors.NumHDonors(mol) / max(1, mol.GetNumAtoms()),
        Descriptors.NumHAcceptors(mol) / max(1, mol.GetNumAtoms())
    ]

    return desc + fp_bits + graph_features


def featurize_csv(input_csv, output_csv, use_augmentation=False, augmentation_ratio=0.3):
    ### Featurize CSV with optional augmentation
    df = pd.read_csv(input_csv)

    if use_augmentation:
        print("Creating augmented dataset...")
        smiles_list = df['SMILES'].tolist()
        augmented_smiles, original_smiles = create_augmented_dataset(smiles_list, augmentation_ratio)

        ### Create augmented dataframe
        aug_df = pd.DataFrame({
            'SMILES': augmented_smiles,
            'Original_SMILES': original_smiles,
            'Is_Augmented': [s != o for s, o in zip(augmented_smiles, original_smiles)]
        })

        ### Merge with original data
        df = pd.merge(aug_df, df, left_on='Original_SMILES', right_on='SMILES', how='left')
        df = df.drop(['SMILES_y', 'Original_SMILES'], axis=1)
        df = df.rename(columns={'SMILES_x': 'SMILES'})

    features = []
    valid_idx = []

    for idx, row in df.iterrows():
        feats = smiles_to_features(row['SMILES'], use_augmentation)
        if feats is not None:
            features.append(feats)
            valid_idx.append(idx)
        else:
            print(f"Invalid SMILES at index {idx}: {row['SMILES']}")

    ### Create feature names
    feature_names = descriptor_names + [f'FP_{i}' for i in range(512)] + [
        'NumAtoms', 'NumBonds', 'NumRings', 'FractionCsp3', 'RotBondsRatio',
        'HDonorsRatio', 'HAcceptorsRatio'
    ]

    features_df = pd.DataFrame(features, columns=feature_names, index=valid_idx)
    result = pd.concat([df.loc[valid_idx].reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

    result.to_csv(output_csv, index=False)
    print(f"Features saved to: {output_csv}")
    print(f"Total samples: {len(result)}")
    if use_augmentation:
        aug_count = result['Is_Augmented'].sum()
        print(f"Augmented samples: {aug_count}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')

    train_in = os.path.join(data_dir, 'train_cleaned.csv')
    test_in = os.path.join(data_dir, 'test.csv')
    train_out = os.path.join(data_dir, 'train_features.csv')
    test_out = os.path.join(data_dir, 'test_features.csv')

    ### Process training data with augmentation
    if os.path.exists(train_in):
        featurize_csv(train_in, train_out, use_augmentation=True, augmentation_ratio=0.3)
    else:
        print(f"Training file not found: {train_in}")

    ### Process test data without augmentation
    if os.path.exists(test_in):
        featurize_csv(test_in, test_out, use_augmentation=False)
    else:
        print(f"Test file not found: {test_in}")
### #%#