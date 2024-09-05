import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import Levenshtein
import pandas as pd

# Load the trained model and tokenizer
model = RobertaForMaskedLM.from_pretrained('/Users/nisargshah/Documents/cs/ml4/frag_ml/model-mTOR')
tokenizer = RobertaTokenizer.from_pretrained('/Users/nisargshah/Documents/cs/ml4/frag_ml/tokenizer-mTOR')
model.eval()

# Load the unique SMILES dataset
unique_smiles_df = pd.read_csv('/Users/nisargshah/Documents/cs/ml4/frag_ml/unique_smile5.csv')
unique_smiles_list = unique_smiles_df['SMILES'].tolist()

def predict_fragment_smiles(smiles, model, tokenizer, max_length=128):
    inputs = tokenizer(smiles, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    logits = outputs.logits
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_smiles = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    
    return predicted_smiles

def tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def string_similarity(smiles1, smiles2):
    distance = Levenshtein.distance(smiles1, smiles2)
    max_len = max(len(smiles1), len(smiles2))
    if max_len == 0:
        return 1.0
    return 1 - (distance / max_len)

def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def find_closest_valid_smiles(predicted_smiles, unique_smiles_list):
    closest_smiles = None
    highest_similarity = -1
    for smiles in unique_smiles_list:
        similarity = string_similarity(predicted_smiles, smiles)
        if similarity > highest_similarity:
            highest_similarity = similarity
            closest_smiles = smiles
    return closest_smiles


new_drug_smiles = ""  # Replace with the input SMILES
predicted_fragment_smiles = predict_fragment_smiles(new_drug_smiles, model, tokenizer)
print("intial smiles: ", predicted_fragment_smiles)
if not is_valid_smiles(predicted_fragment_smiles):
    print("Predicted SMILES is invalid. Finding the closest valid SMILES...")
    closest_valid_smiles = find_closest_valid_smiles(predicted_fragment_smiles, unique_smiles_list)
    predicted_fragment_smiles = closest_valid_smiles
    print("Closest valid SMILES:", closest_valid_smiles)

print("Predicted Fragment SMILES:", predicted_fragment_smiles)

actual_fragment_smiles = ""  # Replace with the actual fragment SMILES in order to test accuracy
similarity = tanimoto_similarity(predicted_fragment_smiles, actual_fragment_smiles)
print("Tanimoto Similarity:", similarity)

# Calculate string similarity
string_sim = string_similarity(predicted_fragment_smiles, actual_fragment_smiles)
print("String Similarity:", string_sim)
