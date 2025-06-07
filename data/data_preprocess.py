import os
import numpy as np
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from write_lmdb import write_lmdb

random.seed(42)

# === CONFIG ===
CSV_PATH = "FNMR.csv"  
FINETUNE_DATA_SAVE_DIR = "../baseline/data/finetune"
INFER_DATA_SAVE_DIR = "../baseline/data/infer"
TRAIN_LMDB = os.path.join(FINETUNE_DATA_SAVE_DIR, "train.lmdb")
VALID_LMDB = os.path.join(FINETUNE_DATA_SAVE_DIR, "valid.lmdb")
TEST_LMDB = os.path.join(INFER_DATA_SAVE_DIR , "test.lmdb")
TRAIN_RATIO = 0.8
VALID_RATIO = 0.2  # 20% of train as valid
F_SHIFT_MIN, F_SHIFT_MAX = -250, 100   # Valid ^19F NMR chemical shift range (ppm)

os.makedirs(FINETUNE_DATA_SAVE_DIR, exist_ok=True)
os.makedirs(INFER_DATA_SAVE_DIR, exist_ok=True)

data = pd.read_csv(CSV_PATH)

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

    samples.append({
        'atoms': atoms,
        'coordinates': coordinates,
        'atoms_target': atoms_target,
        'atoms_target_mask': atoms_target_mask,
        'smiles': smiles,
        'db_id': f"fluoride_{i}",
        'mol': mol,
        'inchikey': inchikey,
    })

# === SPLIT ===
random.shuffle(samples)
total = len(samples)
train_cut = int(total * TRAIN_RATIO)
train_full = samples[:train_cut]
test_data = samples[train_cut:]

valid_cut = int(len(train_full) * VALID_RATIO)
valid_data = train_full[:valid_cut]
train_data = train_full[valid_cut:]

print(f"Processed samples: {total} | Train: {len(train_data)} | Valid: {len(valid_data)} | Test: {len(test_data)}")

# === WRITE ===
write_lmdb(TRAIN_LMDB, train_data)
write_lmdb(VALID_LMDB, valid_data)
write_lmdb(TEST_LMDB, test_data)