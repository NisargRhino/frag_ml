import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.ML.Descriptors import MoleculeDescriptors
from xgboost import XGBRegressor
from sklearn.svm import SVR
import joblib

# Load the dataset (first 100 rows)
file_path = 'enter file path'
data_org = pd.read_csv(file_path, nrows=1000)

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
        fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    else:
        return np.array([np.nan] * n_bits)

# Calculate descriptors and fingerprints for the drug SMILES
descriptors = data_org['DRUG SMILES'].apply(calculate_descriptors)
descriptors_df = pd.DataFrame(descriptors.tolist(), columns=descriptor_names)

fingerprints = data_org['DRUG SMILES'].apply(calculate_morgan_fingerprints)
fingerprints_df = pd.DataFrame(fingerprints.tolist(), columns=[f'FP_{i}' for i in range(2048)])

# Combine the original data with the new descriptors and fingerprints
data = pd.concat([descriptors_df, fingerprints_df], axis=1)

# Drop rows with missing values in target column (if any)
data_org.dropna(subset=['FRAG_SMILES'], inplace=True)

# Encode the fragment SMILES
label_encoder_smiles = LabelEncoder()
data_org['Fragment_Label'] = label_encoder_smiles.fit_transform(data_org['FRAG_SMILES'])

# Define feature columns and target column
feature_columns = list(fingerprints_df.columns) + descriptor_names
target_column = 'Fragment_Label'

# Impute missing values in feature columns
imputer = SimpleImputer(strategy='mean')
data[feature_columns] = imputer.fit_transform(data[feature_columns])

# Scale the feature columns
scaler = StandardScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Split the data into training and testing sets
X = data[feature_columns]
y = data_org[target_column]
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

# Save the best model and other objects
joblib.dump(best_model, f'best_model_{best_model_name}.pkl')
joblib.dump(label_encoder_smiles, 'label_encoder_smiles.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')
