import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from rdkit import Chem
from rdkit.Chem import Descriptors
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

# Load the dataset
file_path = 'dataset7.csv'
data = pd.read_csv(file_path)

# Print the column names to verify
print("Column names in CSV file:", data.columns)

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
data.dropna(subset=['DRUG SMILES'], inplace=True)

# Encode the fragment SMILES
label_encoder_smiles = LabelEncoder()
data['Fragment_Label'] = label_encoder_smiles.fit_transform(data['FRAG_SMILES'])

# Encode categorical features
categorical_features = [
    'Absorption\nP-glycoprotein substrate', 'Absorption\nP-glycoprotein I inhibitor', 'Absorption\nP-glycoprotein II inhibitor',
    'Distribution\nFraction unbound (human)', 'Distribution\nBBB permeability', 'Distribution\nCNS permeability',
    'Metabolism\nCYP2D6 substrate', 'Metabolism\nCYP3A4 substrate', 'Metabolism\nCYP1A2 inhibitior',
    'Metabolism\nCYP2C19 inhibitior', 'Metabolism\nCYP2C9 inhibitior', 'Metabolism\nCYP2D6 inhibitior',
    'Metabolism\nCYP3A4 inhibitior', 'Excretion\nRenal OCT2 substrate', 'Toxicity\nAMES toxicity',
    'Toxicity\nhERG I inhibitor', 'Toxicity\nhERG II inhibitor', 'Toxicity\nHepatotoxicity', 'Toxicity\nSkin Sensitisation'
]

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define feature columns and target column
feature_columns = descriptor_columns + [
    'Molecular Weight', 'LogP', '#Rotatable Bonds', '#Acceptors', '#Donors',
    'Surface Area', 'Absorption\nWater solubility', 'Absorption\nCaco2 permeability',
    'Absorption\nIntestinal absorption (human)', 'Absorption\nSkin Permeability',
    'Absorption\nP-glycoprotein substrate', 'Absorption\nP-glycoprotein I inhibitor',
    'Absorption\nP-glycoprotein II inhibitor', 'Distribution\nVDss (human)',
    'Distribution\nFraction unbound (human)', 'Distribution\nBBB permeability',
    'Distribution\nCNS permeability', 'Metabolism\nCYP2D6 substrate', 'Metabolism\nCYP3A4 substrate',
    'Metabolism\nCYP1A2 inhibitior', 'Metabolism\nCYP2C19 inhibitior', 'Metabolism\nCYP2C9 inhibitior',
    'Metabolism\nCYP2D6 inhibitior', 'Metabolism\nCYP3A4 inhibitior', 'Excretion\nTotal Clearance',
    'Excretion\nRenal OCT2 substrate', 'Toxicity\nAMES toxicity', 'Toxicity\nMax. tolerated dose (human)',
    'Toxicity\nhERG I inhibitor', 'Toxicity\nhERG II inhibitor', 'Toxicity\nOral Rat Acute Toxicity (LD50)',
    'Toxicity\nOral Rat Chronic Toxicity (LOAEL)', 'Toxicity\nHepatotoxicity', 'Toxicity\nSkin Sensitisation',
    'Toxicity\nT.Pyriformis toxicity', 'Toxicity\nMinnow toxicity'
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

# Function to generate fragments from a given drug SMILES (placeholder)
def generate_fragments(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    frags = Chem.rdmolops.FragmentOnBRICSBonds(mol)
    return [Chem.MolToSmiles(frag) for frag in frags]

# Function to generate the best fragment for a given drug SMILES
def generate_best_fragment(smiles, model, label_encoder, feature_columns):
    descriptors = calculate_descriptors(smiles)
    descriptors_df = pd.DataFrame([descriptors], columns=descriptor_columns)
    
    for col in feature_columns:
        if col not in descriptors_df.columns:
            descriptors_df[col] = np.nan

    fragments = generate_fragments(smiles)
    if not fragments:
        return None, None
    
    fragment_descriptors = [calculate_descriptors(frag) for frag in fragments]
    fragment_df = pd.DataFrame(fragment_descriptors, columns=descriptor_columns)
    fragment_df = fragment_df.fillna(0)
    
    predictions = model.predict(fragment_df)
    best_fragment_idx = np.argmax(predictions)
    best_fragment = fragments[best_fragment_idx]
    
    return best_fragment, predictions[best_fragment_idx]

# Example usage: generate the best fragment for a given drug SMILES
example_smiles = 'C1CC1C2=CC=CC=C2'
best_fragment, prediction = generate_best_fragment(example_smiles, best_model, label_encoder_smiles, feature_columns)
print(f"Best fragment for {example_smiles}: {best_fragment} with prediction: {prediction}")
