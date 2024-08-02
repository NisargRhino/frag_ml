import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import joblib

# Load dataset
file_path = 'enter path to csv'
data = pd.read_csv(file_path)

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

# Calculate specific descriptors for SMILES
def calculate_specific_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(specific_descriptor_names)
        return calculator.CalcDescriptors(mol)
    else:
        return [np.nan] * len(specific_descriptor_names)

# Calculate Morgan fingerprints for SMILES
def calculate_morgan_fingerprints(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    else:
        return np.array([np.nan] * n_bits)

# Calculate descriptors and fingerprints for the drug SMILES
descriptors = data['DRUG SMILES'].apply(calculate_specific_descriptors)
descriptors_df = pd.DataFrame(descriptors.tolist(), columns=specific_descriptor_names)

fingerprints = data['DRUG SMILES'].apply(calculate_morgan_fingerprints)
fingerprints_df = pd.DataFrame(fingerprints.tolist(), columns=[f'FP_{i}' for i in range(2048)])

# Combine the original data with the new descriptors and fingerprints
data = pd.concat([data, descriptors_df, fingerprints_df], axis=1)

# Encode the fragment SMILES
label_encoder_smiles = LabelEncoder()
data['Fragment_Label'] = label_encoder_smiles.fit_transform(data['FRAG_SMILES'])

# Define feature columns and target column
feature_columns = list(fingerprints_df.columns) + specific_descriptor_names
target_column = 'Fragment_Label'

# Impute missing values in feature columns
imputer = SimpleImputer(strategy='median')
data[feature_columns] = imputer.fit_transform(data[feature_columns])

# Scale the feature columns
scaler = StandardScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Split the data into training and testing sets
X = data[feature_columns]
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor(),
    'SVR': SVR(),
    'AdaBoost': AdaBoostRegressor()
}

# Define the parameter grid for each model
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'SVR': {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }
}

# Perform GridSearchCV for each model and store the best model
best_models = {}
for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, n_jobs=-1, scoring='r2')
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best Params for {model_name}: {grid_search.best_params_}")

# Evaluate each model on the test set and print the metrics
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - RMSE: {rmse}, MAE: {mae}, R2: {r2}")

# Choose the best model based on R2 score
best_model_name = max(best_models, key=lambda name: r2_score(y_test, best_models[name].predict(X_test)))
best_model = best_models[best_model_name]
print(f"Best model: {best_model_name}")

# Save the best model, imputer, scaler, and label encoder
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder_smiles, 'label_encoder_smiles.pkl')
