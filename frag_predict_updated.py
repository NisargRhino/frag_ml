import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

# List of specific descriptors to calculate
specific_descriptor_names = [
    'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinEStateIndex', 'MolWt',
    'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'BCUT2D_MWHI',
    'BCUT2D_MRHI', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v',
    'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n',
    'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
    'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
    'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
    'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10',
    'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7',
    'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
    'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6',
    'SlogP_VSA7', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11',
    'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
    'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10',
    'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6',
    'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'HeavyAtomCount',
    'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
    'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
    'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
    'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
    'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_OH', 'fr_Ar_N',
    'fr_Ar_OH', 'fr_C_O', 'fr_C_O_noCOO', 'fr_NH0', 'fr_NH1', 'fr_NH2',
    'fr_Ndealkylation2', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide',
    'fr_aniline', 'fr_aryl_methyl', 'fr_benzene', 'fr_bicyclic', 'fr_ether',
    'fr_halogen', 'fr_methoxy', 'fr_para_hydroxylation', 'fr_phenol',
    'fr_phenol_noOrthoHbond', 'fr_unbrch_alkane'
]

# Function to calculate specific descriptors for SMILES
def calculate_specific_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(specific_descriptor_names)
        return calculator.CalcDescriptors(mol)
    else:
        return [np.nan] * len(specific_descriptor_names)

# Function to calculate Morgan fingerprints for SMILES
def calculate_morgan_fingerprints(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    else:
        return np.array([np.nan] * n_bits)

# Function to generate the best fragment for a given drug SMILES
def generate_best_fragment(smiles, model, feature_columns):
    fingerprints = calculate_morgan_fingerprints(smiles)
    descriptors = calculate_specific_descriptors(smiles)
    if any(np.isnan(fingerprints)) or any(np.isnan(descriptors)):
        return "Invalid SMILES input"
    
    input_data = pd.DataFrame([list(fingerprints) + list(descriptors)], columns=feature_columns)
    
    # Ensure input_data has all required feature columns
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = np.nan
    input_data = input_data[feature_columns]
    
    # Handle missing values
    input_data = imputer.transform(input_data)
    input_data = scaler.transform(input_data)
    
    # Predict the fragment
    fragment_label = model.predict(input_data)[0]
    fragment_smiles = label_encoder_smiles.inverse_transform([int(fragment_label)])[0]
    
    return fragment_smiles

# Function to clean up a molecule
def cleanup_molecule_rdkit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)

# Load the trained model and other objects
best_model = joblib.load('best_model.pkl')
label_encoder_smiles = joblib.load('label_encoder_smiles.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

# Define feature columns (this should match the feature columns used during training)
feature_columns = list(f'FP_{i}' for i in range(2048)) + specific_descriptor_names

# Drug SMILES input
drug_smiles = "enter smiles"

# Generate and print the best fragment
best_fragment = generate_best_fragment(drug_smiles, best_model, feature_columns)
cleaned_fragment_smiles = cleanup_molecule_rdkit(best_fragment)

print("Best fragment SMILES:", cleaned_fragment_smiles)
