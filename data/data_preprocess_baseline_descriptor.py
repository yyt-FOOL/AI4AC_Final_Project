import os
import numpy as np
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType
from tqdm import tqdm
from write_lmdb import write_lmdb

random.seed(42)

# === CONFIG ===
train_CSV_PATH = "train.csv"  
test_CSV_PATH = "test.csv"  
FINETUNE_DATA_SAVE_DIR = "../baseline_descriptor/data/finetune"
TEST_DATA_SAVE_DIR = "../baseline_descriptor/data/test"
TRAIN_LMDB = os.path.join(FINETUNE_DATA_SAVE_DIR, "train.lmdb")
VALID_LMDB = os.path.join(FINETUNE_DATA_SAVE_DIR, "valid.lmdb")
TEST_LMDB = os.path.join(TEST_DATA_SAVE_DIR , "valid.lmdb")
TRAIN_RATIO = 0.8
VALID_RATIO = 0.2  # 20% of train as valid
F_SHIFT_MIN, F_SHIFT_MAX = -250, 100

os.makedirs(FINETUNE_DATA_SAVE_DIR, exist_ok=True)
os.makedirs(TEST_DATA_SAVE_DIR, exist_ok=True)

train_data = pd.read_csv(train_CSV_PATH)
test_data = pd.read_csv(test_CSV_PATH)

def get_rdkit_atom_descriptor(mol):
    features = []
    for atom in mol.GetAtoms():
        feat = [
            atom.GetAtomicNum(),
            atom.GetTotalValence(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic()),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()) / float(HybridizationType.SP3),  # Normalize to [0,1] range
        ]
        features.append(feat)
    return np.array(features, dtype=np.float32)

def process_data(data, tag_prefix):
    samples = []
    for i, row in tqdm(data.iterrows(), total=len(data)):
        smiles = row['SMILES']
        shift = row['shift_value']

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        mol = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol, randomSeed=42)
        if res != 0:
            continue
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            continue

        conf = mol.GetConformer()
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coordinates = np.array(conf.GetPositions(), dtype=np.float32)
        atoms_target = np.zeros(len(atoms), dtype=np.float32)
        atoms_target_mask = np.zeros(len(atoms), dtype=np.int64)

        f_indices = [i for i, a in enumerate(atoms) if a == 'F']
        if len(f_indices) == 0:
            continue

        if not (F_SHIFT_MIN <= shift <= F_SHIFT_MAX):
            continue

        for idx in f_indices:
            atoms_target[idx] = float(shift)
            atoms_target_mask[idx] = 1

        try:
            inchikey = Chem.MolToInchiKey(mol)
        except:
            inchikey = None

        atom_descriptor = get_rdkit_atom_descriptor(mol)

        samples.append({
            'atoms': atoms,
            'coordinates': coordinates,
            'atoms_target': atoms_target,
            'atoms_target_mask': atoms_target_mask,
            'smiles': smiles,
            'db_id': f"{tag_prefix}_{i}",
            'mol': mol,
            'inchikey': inchikey,
            'atoms_descriptor': atom_descriptor
        })
    return samples

# === PROCESS ===
train_samples = process_data(train_data, "fluoride")
test_samples = process_data(test_data, "fluoride")

random.shuffle(train_samples)
total = len(train_samples)
train_cut = int(total * TRAIN_RATIO)
train_data = train_samples[:train_cut]
valid_data = train_samples[train_cut:]

print(f"Processed train samples: {total} | Train: {len(train_data)} | Valid: {len(valid_data)}")
print(f"Processed test samples: {len(test_samples)}")

# === WRITE ===
write_lmdb(TRAIN_LMDB, train_data)
write_lmdb(VALID_LMDB, valid_data)
write_lmdb(TEST_LMDB, test_samples)