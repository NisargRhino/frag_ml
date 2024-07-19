import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from xgboost import XGBRegressor
from sklearn.svm import SVR
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
    "SVR": SVR()
}

# Define the parameter grids for RandomizedSearchCV
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
    }
}

# Perform RandomizedSearchCV to find the best model and hyperparameters
best_models = {}
for model_name, model in models.items():
    print(f"Optimizing {model_name}...")
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grids[model_name], n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_models[model_name] = random_search.best_estimator_
    print(f"Best parameters for {model_name}: {random_search.best_params_}")

# Evaluate all models on the test set and compare their performances
model_performances = {}
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    model_performances[model_name] = mse
    print(f"{model_name} Mean Squared Error: {mse}")

# Select the best model based on MSE
best_model_name = min(model_performances, key=model_performances.get)
best_model = best_models[best_model_name]
print(f"Best model: {best_model_name} with MSE: {model_performances[best_model_name]}")

# Function to generate fragmentations for a given drug SMILES
def generate_fragmentations(smiles, model, label_encoder, feature_columns, num_fragments=10):
    descriptors = calculate_descriptors(smiles)
    if any(np.isnan(descriptors)):
        return ["Invalid SMILES input"]
    
    input_data = pd.DataFrame([descriptors], columns=descriptor_columns)
    
    # Ensure input_data has all required feature columns
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = np.nan
    input_data = input_data[feature_columns]
    
    # Impute missing values
    input_data = imputer.transform(input_data)
    
    predicted_labels = model.predict(input_data)
    predicted_fragments = label_encoder.inverse_transform(predicted_labels.astype(int))
    
    return predicted_fragments[:num_fragments]

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

# Generate the top 10 fragments for a given drug SMILES
drug_smiles = 'CC[C@@]1(O)CCc2c(O)cc(C(/C(=C\C=O)/C=C\C=C\OC)=O)c(O)c2C1'
top_fragments = generate_fragmentations(drug_smiles, best_model, label_encoder_smiles, feature_columns, num_fragments=10)
print("Top 10 Fragments:", top_fragments)

# Clean up the top fragments
cleaned_fragments = [cleanup_molecule_rdkit(smiles) for smiles in top_fragments if smiles != "Invalid SMILES input"]
cleaned_fragments_smiles = [Chem.MolToSmiles(frag) for frag in cleaned_fragments]

# Print the cleaned-up SMILES
print("Cleaned-Up Top 10 Fragments SMILES:")
for smiles in cleaned_fragments_smiles:
    print(smiles)
