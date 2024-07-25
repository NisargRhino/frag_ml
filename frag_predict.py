import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, SanitizeMol
from rdkit.ML.Descriptors import MoleculeDescriptors

# List all available molecular descriptors
descriptor_names = [desc_name[0] for desc_name in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

# Define a function to calculate all available molecular descriptors for SMILES
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return calculator.CalcDescriptors(mol)
    else:
        return [np.nan] * len(descriptor_names)

# Define a function to calculate Morgan fingerprints for SMILES
def calculate_morgan_fingerprints(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    else:
        return np.array([np.nan] * n_bits)

# Function to generate the best fragment for a given drug SMILES and other properties
def generate_best_fragment(smiles, model, feature_columns):
    fingerprints = calculate_morgan_fingerprints(smiles)
    descriptors = calculate_descriptors(smiles)
    if any(np.isnan(fingerprints)) or any(np.isnan(descriptors)):
        return "Invalid SMILES input"
    
    input_data = pd.DataFrame([list(fingerprints) + list(descriptors)], columns=feature_columns)
    
    # Ensure input_data has all required feature columns
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = np.nan
    input_data = input_data[feature_columns]
    
    # Handle missing values
    input_data = imputer.fit_transform(input_data)
    input_data = scaler.fit_transform(input_data)
    
    # Predict the fragment
    fragment_label = model.predict(input_data)[0]
    fragment_smiles = label_encoder_smiles.inverse_transform([int(fragment_label)])[0]
    
    return fragment_smiles
def cleanup_molecule_rdkit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)

# Load the trained model and other objects
best_model = joblib.load('best_model_GradientBoosting.pkl')
label_encoder_smiles = joblib.load('label_encoder_smiles.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

# Load feature columns
feature_columns = joblib.load('feature_columns.pkl')

# Example drug SMILES input
drug_smiles = "Enter Drug Smiles"

# Generate and print the best fragment
best_fragment = generate_best_fragment(drug_smiles, best_model, feature_columns)
cleaned_fragment_smiles = cleanup_molecule_rdkit(best_fragment)

print("Best fragment SMILES:", cleaned_fragment_smiles)
