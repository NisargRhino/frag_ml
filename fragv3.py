import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import rdMolDescriptors
from xgboost import XGBRegressor
from sklearn.svm import SVR
from rdkit.Chem import SanitizeMol
# Load the dataset
file_path = '/Users/nisargshah/Documents/cs/ml3/frag_dock/dataset14.csv'
data = pd.read_csv(file_path)

# Define a function to calculate molecular descriptors for SMILES
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.TPSA(mol)
            
        ]
    else:
        return [np.nan] * 6

# Define a function to calculate Morgan fingerprints for SMILES
def calculate_morgan_fingerprints(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    else:
        return np.array([np.nan] * n_bits)

# Calculate descriptors and fingerprints for the drug SMILES
descriptor_columns = ['MolWt', 'MolLogP', 'NumRotatableBonds', 'NumHAcceptors', 'NumHDonors', 'TPSA']
descriptors = data['DRUG SMILES'].apply(calculate_descriptors)
descriptors_df = pd.DataFrame(descriptors.tolist(), columns=descriptor_columns)

fingerprints = data['DRUG SMILES'].apply(calculate_morgan_fingerprints)
fingerprints_df = pd.DataFrame(fingerprints.tolist(), columns=[f'FP_{i}' for i in range(2048)])

# Combine the original data with the new descriptors and fingerprints
data = pd.concat([data, descriptors_df, fingerprints_df], axis=1)

# Drop rows with missing values in target column (if any)
data.dropna(subset=['FRAG_SMILES'], inplace=True)

# Encode the fragment SMILES
label_encoder_smiles = LabelEncoder()
data['Fragment_Label'] = label_encoder_smiles.fit_transform(data['FRAG_SMILES'])

# Define feature columns and target column
feature_columns = list(fingerprints_df.columns) + descriptor_columns + [ 
    'Molecular Weight', 'LogP', '#Rotatable Bonds', '#Acceptors', '#Donors',
    'TPSA', 'Absorption\nWater solubility', 'Absorption\nCaco2 permeability', 'Absorption\nSkin Permeability',
    'Distribution\nVDss (human)', 
    'Distribution\nBBB permeability', 
    'Toxicity\nOral Rat Acute Toxicity (LD50)',  'docking score '
]

target_column = 'Fragment_Label'

# Impute missing values in feature columns
imputer = SimpleImputer(strategy='mean')
data[feature_columns] = imputer.fit_transform(data[feature_columns])

# Scale the feature columns
scaler = StandardScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Split the data into training and testing sets
X = data[feature_columns]
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models to be compared
models = {
    "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),
    "SVR": SVR(),
}

# Define the parameter grids for GridSearchCV


param_grids = {
    "RandomForest": {
        'n_estimators': [200, 500, 1000],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "GradientBoosting": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "XGBoost": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    },
    "SVR": {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 1],
        'kernel': ['linear', 'poly', 'rbf']
    },
}


# Perform GridSearchCV to find the best model and hyperparameters
best_models = {}
for model_name, model in models.items():
    print(f"Optimizing {model_name}...")
    try:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=KFold(n_splits=5, shuffle=True, random_state=42), verbose=2, n_jobs=-1, error_score=np.nan)
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    except ValueError as e:
        print(f"Error during training {model_name}: {e}")

# Evaluate all models on the test set and compare their performances
model_performances = {}
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_performances[model_name] = {
        "mse": mse,
        "r2": r2
    }
    print(f"{model_name} Mean Squared Error: {mse}, R2 Score: {r2}")

# Select the best model based on R2 score
best_model_name = max(model_performances, key=lambda name: model_performances[name]['r2'])
best_model = best_models[best_model_name]
print(f"Best model: {best_model_name}")

# Function to generate fragments from a molecule using RDKit
def generate_fragments(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return [Chem.MolToSmiles(frag) for frag in frags]

# Function to generate the best fragment for a given drug SMILES and other properties
def generate_best_fragment(smiles, properties, model, feature_columns):
    fingerprints = calculate_morgan_fingerprints(smiles)
    if any(np.isnan(fingerprints)):
        return "Invalid SMILES input"
    
    input_data = pd.DataFrame([list(fingerprints) + properties], columns=feature_columns)
    
    # Ensure input_data has all required feature columns
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = np.nan
    input_data = input_data[feature_columns]
    
    # Impute and scale the input data
    input_data = imputer.transform(input_data)
    input_data = scaler.transform(input_data)
    
    predicted_label = model.predict(input_data)[0]
    predicted_fragment = label_encoder_smiles.inverse_transform([int(predicted_label)])[0]
    
    # Verify if the predicted fragment is part of the original drug molecule
    fragments = generate_fragments(smiles)
    return predicted_fragment

    #if predicted_fragment in fragments:
     #   return predicted_fragment
    #else:
     #   return "No valid fragment found in the original molecule"

# Example usage
drug_smiles = 'C[C@H]1[C@H]([C@H](C[C@@H](O1)O[C@H]2C[C@@](CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)C)O)N)O'
properties = [
    527.526,  # Molecular Weight
    1.0289,    # LogP
    4,      # #Rotatable Bonds
    11,      # #Acceptors
    5,      # #Donors
    185.84,      # TPSA
    -3.964488555,    # Absorption\nWater solubility
    -6.215190228,    # Absorption\nCaco2 permeability
    0.428500405,    # Absorption\nSkin Permeability
    23.20626101,    # Distribution\nVDss (human)
    0.0777454529,    # Distribution\nBBB permeability
    3.25506531,  # Toxicity\nOral Rat Acute Toxicity (LD50)
    -8.9    #docking score
]
descriptors = calculate_descriptors(drug_smiles)

properties_df = pd.DataFrame([properties], columns=[
    'Molecular Weight', 'LogP', '#Rotatable Bonds', '#Acceptors', '#Donors',
    'TPSA', 'Absorption\nWater solubility', 'Absorption\nCaco2 permeability', 'Absorption\nSkin Permeability',
    'Distribution\nVDss (human)', 
    'Distribution\nBBB permeability', 
    'Toxicity\nOral Rat Acute Toxicity (LD50)',  'docking score '
])

properties_df = pd.concat([properties_df, pd.DataFrame([descriptors], columns=descriptor_columns)], axis=1)

# Convert the properties DataFrame to a list
properties_combined = properties_df.iloc[0].tolist()
# Combine the original data with the new descriptors and fingerprints


# Function to clean up the molecule using RDKit and return canonical SMILES
def cleanup_molecule_rdkit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        SanitizeMol(mol)  # Sanitize the molecule
        cleaned_smiles = Chem.MolToSmiles(mol, canonical=True)
        return cleaned_smiles
    else:
        return None

# Predicting the best fragment and cleaning it up
best_fragment = generate_best_fragment(drug_smiles, properties_combined, best_model, feature_columns)
print("Best Fragment:", best_fragment)

if best_fragment != "No valid fragment found in the original molecule":
    cleaned_fragment_smiles = cleanup_molecule_rdkit(best_fragment)
    if cleaned_fragment_smiles:
        print("Cleaned-Up Best Fragment SMILES:", cleaned_fragment_smiles)
    else:
        print("Failed to clean up the fragment SMILES.")
else:
    print("No valid fragment found in the original molecule.")
