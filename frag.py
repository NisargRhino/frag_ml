import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

# Load the cleaned CSV file
file_path = 'dataset5.csv'
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

# Apply the descriptor function to the drug SMILES
descriptor_columns = ['MolWt', 'MolLogP', 'NumRotatableBonds', 'NumHAcceptors', 'NumHDonors', 'TPSA']
descriptors = data['DRUG SMILES'].apply(calculate_descriptors)
descriptors_df = pd.DataFrame(descriptors.tolist(), columns=descriptor_columns)

# Combine the original data with the new descriptors
data = pd.concat([data, descriptors_df], axis=1)

# Drop rows with missing values in target column (if any)
data.dropna(subset=['SMILES'], inplace=True)

# Encode the fragment SMILES
label_encoder_smiles = LabelEncoder()
data['Fragment_Label'] = label_encoder_smiles.fit_transform(data['SMILES'])

# Encode categorical features
categorical_features = [
    'Absorption P-glycoprotein substrate', 'Absorption P-glycoprotein I inhibitor', 'Absorption P-glycoprotein II inhibitor',
    'Distribution Fraction unbound (human)', 'Distribution BBB permeability', 'Distribution CNS permeability',
    'Metabolism CYP2D6 substrate', 'Metabolism CYP3A4 substrate', 'Metabolism CYP1A2 inhibitior',
    'Metabolism CYP2C19 inhibitior', 'Metabolism CYP2C9 inhibitior', 'Metabolism CYP2D6 inhibitior',
    'Metabolism CYP3A4 inhibitior', 'Excretion Renal OCT2 substrate', 'Toxicity AMES toxicity',
    'Toxicity hERG I inhibitor', 'Toxicity hERG II inhibitor', 'Toxicity Hepatotoxicity', 'Toxicity Skin Sensitisation'
]

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define feature columns and target column
feature_columns = descriptor_columns + [
    'Molecular Weight', 'LogP', '#Rotatable Bonds', '#Acceptors', '#Donors',
    'Surface Area', 'Absorption Water solubility', 'Absorption Caco2 permeability',
    'Absorption Intestinal absorption (human)', 'Absorption Skin Permeability',
    'Absorption P-glycoprotein substrate', 'Absorption P-glycoprotein I inhibitor',
    'Absorption P-glycoprotein II inhibitor', 'Distribution VDss (human)',
    'Distribution Fraction unbound (human)', 'Distribution BBB permeability',
    'Distribution CNS permeability', 'Metabolism CYP2D6 substrate', 'Metabolism CYP3A4 substrate',
    'Metabolism CYP1A2 inhibitior', 'Metabolism CYP2C19 inhibitior', 'Metabolism CYP2C9 inhibitior',
    'Metabolism CYP2D6 inhibitior', 'Metabolism CYP3A4 inhibitior', 'Excretion Total Clearance',
    'Excretion Renal OCT2 substrate', 'Toxicity AMES toxicity', 'Toxicity Max. tolerated dose (human)',
    'Toxicity hERG I inhibitor', 'Toxicity hERG II inhibitor', 'Toxicity Oral Rat Acute Toxicity (LD50)',
    'Toxicity Oral Rat Chronic Toxicity (LOAEL)', 'Toxicity Hepatotoxicity', 'Toxicity Skin Sensitisation',
    'Toxicity T.Pyriformis toxicity', 'Toxicity Minnow toxicity'
]
target_column = 'Fragment_Label'

# Impute missing values in feature columns
imputer = SimpleImputer(strategy='mean')
data[feature_columns] = imputer.fit_transform(data[feature_columns])

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
    "MLP": MLPRegressor(random_state=42, max_iter=1000)
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
    "MLP": {
        'hidden_layer_sizes': [(100,), (100, 50), (100, 50, 25)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01]
    }
}

# Perform GridSearchCV to find the best model and hyperparameters
best_models = {}
for model_name, model in models.items():
    print(f"Optimizing {model_name}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=KFold(n_splits=5, shuffle=True, random_state=42), verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

# Evaluate all models on the test set and compare their performances
model_performances = {}
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_performances[model_name] = {"mse": mse, "r2": r2}
    print(f"{model_name} Mean Squared Error: {mse}, R2 Score: {r2}")

# Select the best model based on R2 Score
best_model_name = max(model_performances, key=lambda x: model_performances[x]['r2'])
best_model = best_models[best_model_name]
print(f"Best model: {best_model_name} with R2 Score: {model_performances[best_model_name]['r2']}")

# Function to generate the best fragment for a given drug SMILES
def generate_best_fragment(smiles, model, label_encoder, feature_columns):
    descriptors = calculate_descriptors(smiles)
    if any(np.isnan(descriptors)):
        return "Invalid SMILES input"
    
    input_data = pd.DataFrame([descriptors], columns=descriptor_columns)
    
    # Ensure input_data has all required feature columns
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = np.nan
    input_data = input_data[feature_columns]
    
    # Impute missing values
    input_data = imputer.transform(input_data)
    
    predicted_label = model.predict(input_data)[0]
    predicted_fragment = label_encoder.inverse_transform([int(predicted_label)])[0]
    
    return predicted_fragment

# Clean up the molecule using RDKit
def cleanup_molecule_rdkit(smiles):
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)
    
    # Optimize the molecule
    AllChem.UFFOptimizeMolecule(mol)
    
    # Remove hydrogens if needed
    mol = Chem.RemoveHs(mol)
    
    return mol

# Generate the best fragment for a given drug SMILES
drug_smiles = 'CC[C@@]1(O)CCc2c(O)cc(C(/C(=C\C=O)/C=C\C=C\OC)=O)c(O)c2C1'
best_fragment = generate_best_fragment(drug_smiles, best_model, label_encoder_smiles, feature_columns)
print("Best Fragment:", best_fragment)

# Clean up the best fragment
cleaned_fragment = cleanup_molecule_rdkit(best_fragment)
cleaned_fragment_smiles = Chem.MolToSmiles(cleaned_fragment)

# Print the cleaned-up SMILES
print("Cleaned-Up Best Fragment SMILES:", cleaned_fragment_smiles)
