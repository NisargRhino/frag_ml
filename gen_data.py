import os
from rdkit import Chem
from rdkit.Chem import Recap, AllChem
from tqdm import tqdm
import pandas as pd
from dockstring import load_target
import argparse


def fragment_molecule_recaps(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            print(f"Warning: Failed to parse SMILES: {smiles}")
            return []

        recap_tree = Recap.RecapDecompose(molecule)
        fragments = []

        if recap_tree:
            leaves = recap_tree.GetLeaves()
            if leaves:
                for smile, node in leaves.items():
                    # Properly handle wildcard atoms
                    cleaned_smile = smile.replace('*', 'C')  # Replace wildcard with carbon
                    fragments.append(cleaned_smile)
                    print(f"Fragment SMILES: {cleaned_smile}")
            else:
                print("No leaves found in the Recap tree.")
        else:
            print("Failed to obtain Recap decomposition.")
        
        return fragments

    except Exception as e:
        print(f"Error during fragmentation: {e}")
        return []


def cleanup_molecule_rdkit(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        
        return mol

    except Exception as e:
        print(f"Error during molecule cleanup: {e}")
        return None


def dock_fragments(fragments, target_name, docking_dir, mol2_path, center_coords, box_sizes):
    os.makedirs(docking_dir, exist_ok=True)

    convert_command = f"obabel -imol2 {mol2_path} -opdbqt -O {os.path.join(docking_dir, target_name + '_target.pdbqt')} -xr"
    print(f"Running command: {convert_command}")
    os.system(convert_command)

    conf_path = os.path.join(docking_dir, target_name + '_conf.txt')
    with open(conf_path, 'w') as f:
        f.write(f"""center_x = {center_coords[0]}
center_y = {center_coords[1]}
center_z = {center_coords[2]}

size_x = {box_sizes[0]}
size_y = {box_sizes[1]}
size_z = {box_sizes[2]}""")

    target = load_target(target_name, targets_dir=docking_dir)

    best_score = float('inf')  # Initialize to positive infinity
    best_fragment = None

    for frag in tqdm(fragments):
        try:
            cleaned_mol = cleanup_molecule_rdkit(frag)
            if cleaned_mol is None:
                continue  # Skip docking if cleaning failed

            cleaned_smiles = Chem.MolToSmiles(cleaned_mol)
            score, __ = target.dock(cleaned_smiles)

            if score < best_score:  # Update condition for lower scores being better
                best_score = score
                best_fragment = cleaned_smiles
        except Exception as e:
            print(f"Error docking fragment {frag}: {e}")

    return best_fragment, best_score


def main(input_csv, mol2_path, docking_dir, target_name, center_coords, box_sizes, output_path):
    # Read input CSV
    input_data = pd.read_csv(input_csv)
    
    # Clean column names
    input_data.columns = input_data.columns.str.strip()
    
    # Print column names for verification
    print("Column names in CSV:", input_data.columns)
    
    results = []

    for index, row in input_data.iterrows():
        smiles = row.get('SMILES', None)
        if smiles is None:
            print(f"No 'SMILES' column found for row {index + 1}. Skipping.")
            continue

        fragments = fragment_molecule_recaps(smiles)

        if not fragments:
            print(f"No fragments were generated for input {index + 1}.")
            continue

        print(f"Generated {len(fragments)} fragments for input {index + 1}.")

        best_fragment, best_score = dock_fragments(fragments, target_name, docking_dir, mol2_path,
                                                   center_coords, box_sizes)

        if best_fragment is not None:
            results.append({'Name': row['name'], 'SMILES': smiles, 'BestFragment': best_fragment, 'BestScore': best_score})

    if not results:
        print("No successful docking results.")
        return

    # Save results to the output CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drug fragmentation, cleanup, and docking script')

    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file with SMILES strings')
    parser.add_argument('--mol2_path', type=str, required=True, help='Path to mol2 file')
    parser.add_argument('--docking_dir', type=str, default='dockdir', help='Docking directory name/path')
    parser.add_argument('--target_name', type=str, default='target', help='Target name')
    parser.add_argument('--center_coords', type=float, nargs=3, help='Center coordinates for docking box (X Y Z)')
    parser.add_argument('--box_sizes', type=float, nargs=3, help='Box sizes for docking (X Y Z)')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for the results CSV')

    args = parser.parse_args()

    main(args.input_csv, args.mol2_path, args.docking_dir, args.target_name, args.center_coords, args.box_sizes,
         args.output_path)
