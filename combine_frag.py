from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED

def is_drug_like(mol):
    # Apply Lipinski's Rule of Five and QED to check for drug-likeness
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    qed_score = QED.qed(mol)
    
    return (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10 and qed_score > 0.3)

def add_bond_and_adjust_hydrogens(rwmol, atom1_idx, atom2_idx, bond_type):
    rwmol.AddBond(atom1_idx, atom2_idx, bond_type)
    atom1 = rwmol.GetAtomWithIdx(atom1_idx)
    atom2 = rwmol.GetAtomWithIdx(atom2_idx)
    
    if atom1.GetNumExplicitHs() > 0:
        atom1.SetNumExplicitHs(atom1.GetNumExplicitHs() - 1)
    if atom2.GetNumExplicitHs() > 0:
        atom2.SetNumExplicitHs(atom2.GetNumExplicitHs() - 1)
    
    return rwmol

def add_ester_bond(frag1, frag1_atom_idx, frag2, frag2_atom_idx):
    combined_mol = Chem.RWMol(Chem.CombineMols(frag1, frag2))
    frag2_atom_idx_adjusted = frag2_atom_idx + frag1.GetNumAtoms()

    carbon = Chem.Atom(6)  # Carbon
    oxygen = Chem.Atom(8)  # Oxygen
    carbon_idx = combined_mol.AddAtom(carbon)
    oxygen_idx = combined_mol.AddAtom(oxygen)

    combined_mol = add_bond_and_adjust_hydrogens(combined_mol, frag1_atom_idx, carbon_idx, Chem.BondType.SINGLE)
    combined_mol = add_bond_and_adjust_hydrogens(combined_mol, carbon_idx, oxygen_idx, Chem.BondType.DOUBLE)
    combined_mol = add_bond_and_adjust_hydrogens(combined_mol, oxygen_idx, frag2_atom_idx_adjusted, Chem.BondType.SINGLE)

    Chem.SanitizeMol(combined_mol)
    return combined_mol

def add_amide_bond(frag1, frag1_atom_idx, frag2, frag2_atom_idx):
    combined_mol = Chem.RWMol(Chem.CombineMols(frag1, frag2))
    frag2_atom_idx_adjusted = frag2_atom_idx + frag1.GetNumAtoms()

    carbon = Chem.Atom(6)  # Carbon
    nitrogen = Chem.Atom(7)  # Nitrogen
    carbon_idx = combined_mol.AddAtom(carbon)
    nitrogen_idx = combined_mol.AddAtom(nitrogen)

    combined_mol = add_bond_and_adjust_hydrogens(combined_mol, frag1_atom_idx, carbon_idx, Chem.BondType.SINGLE)
    combined_mol = add_bond_and_adjust_hydrogens(combined_mol, carbon_idx, nitrogen_idx, Chem.BondType.SINGLE)
    combined_mol = add_bond_and_adjust_hydrogens(combined_mol, nitrogen_idx, frag2_atom_idx_adjusted, Chem.BondType.SINGLE)

    Chem.SanitizeMol(combined_mol)
    return combined_mol

def combine_fragments(frag1_smiles, frag2_smiles, num_points=10):
    frag1 = Chem.MolFromSmiles(frag1_smiles)
    frag2 = Chem.MolFromSmiles(frag2_smiles)

    frag1_carbon_atoms = [atom.GetIdx() for atom in frag1.GetAtoms() if atom.GetSymbol() == 'C' and atom.GetDegree() < 4]
    frag2_carbon_atoms = [atom.GetIdx() for atom in frag2.GetAtoms() if atom.GetSymbol() == 'C' and atom.GetDegree() < 4]

    if not frag1_carbon_atoms or not frag2_carbon_atoms:
        raise ValueError("No suitable carbon atoms found in one or both fragments for bond formation.")

    combined_molecules = []

    for i in range(min(num_points, len(frag1_carbon_atoms) * len(frag2_carbon_atoms))):
        frag1_atom_idx = frag1_carbon_atoms[i % len(frag1_carbon_atoms)]
        frag2_atom_idx = frag2_carbon_atoms[i % len(frag2_carbon_atoms)]

        try:
            ester_combined_mol = add_ester_bond(frag1, frag1_atom_idx, frag2, frag2_atom_idx)
            if is_drug_like(ester_combined_mol):
                ester_combined_smiles = Chem.MolToSmiles(ester_combined_mol)
                combined_molecules.append(ester_combined_smiles)
        except Exception as e:
            print(f"Error combining molecules with ester bond at point {i}: {e}")

        try:
            amide_combined_mol = add_amide_bond(frag1, frag1_atom_idx, frag2, frag2_atom_idx)
            if is_drug_like(amide_combined_mol):
                amide_combined_smiles = Chem.MolToSmiles(amide_combined_mol)
                combined_molecules.append(amide_combined_smiles)
        except Exception as e:
            print(f"Error combining molecules with amide bond at point {i}: {e}")

    return combined_molecules

# Example usage
frag1_smiles = 'CC(=O)C(C)c1ccc(CC(C)C)cc1'  # Fragment 1 SMILES
frag2_smiles = 'CCn1cc(C(C)=O)c(=O)c2cc(F)c(C)nc21'  # Fragment 2 SMILES

combined_smiles_list = combine_fragments(frag1_smiles, frag2_smiles)

for idx, combined_smiles in enumerate(combined_smiles_list, 1):
    print(f"Combined Molecule {idx}: {combined_smiles}")
