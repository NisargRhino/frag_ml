import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import Levenshtein
import pandas as pd

# Load the trained model and tokenizer
model = RobertaForMaskedLM.from_pretrained('C:\\Users\\nisar\\cs\ml3\\frag_ml\\model_drug_classification')#enter path of the model from train_model.py
tokenizer = RobertaTokenizer.from_pretrained('C:\\Users\\nisar\\cs\ml3\\frag_ml\\tokenizer_drug_classification')#enter path of the tokenizer from train_model.py
model.eval()

# Load the unique SMILES dataset
unique_smiles_df = pd.read_csv('./drug_classification_data_df.csv')# enter the path of unique_smile5.csv
unique_smiles_list = unique_smiles_df['smiles'].tolist()

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


new_drug_smiles = "CSc1ccc(C(=O)c2[nH]c(=O)[nH]c2C)cc1"  # Replace with the input SMILES
#new_drug_smiles = "c1cc(CN2CCCNCCNCCCNCC2)ccc1CN1CCCNCCNCCCNCC1"  # Replace with the input SMILES
#new_drug_smiles = "C[C@@H]1CCC2C[C@@H](/C(=C/C=C/C=C/[C@H](C[C@H](C(=O)[C@@H]([C@@H](/C(=C/[C@H](C(=O)C[C@H](OC(=O)[C@@H]3CCCCN3C(=O)C(=O)[C@@]1(O2)O)[C@H](C)C[C@@H]4CC[C@H]([C@@H](C4)OC)O)C)/C)O)OC)C)C)/C)OC"
#new_drug_smiles = "CCCCCCCCOc1ccccc1C(=O)Nc1ccc(C(=O)OCC[N+](C)(CC)CC)cc1"
#new_drug_smiles = "CC[C@H]1OC(=O)[C@H](C)[C@@H](OC2C[C@@](C)(O)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](OC2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O"
#new_drug_smiles = "O=C1C=CCC1"
predicted_tags = predict_fragment_smiles(new_drug_smiles, model, tokenizer)
print("intial predicted tag: ", predicted_tags)
#if not is_valid_smiles(predicted_fragment_smiles):
#    print("Predicted SMILES is invalid. Finding the closest valid SMILES...")
#    closest_valid_smiles = find_closest_valid_smiles(predicted_fragment_smiles, unique_smiles_list)
#    predicted_fragment_smiles = closest_valid_smiles
#    print("Closest valid SMILES:", closest_valid_smiles)

print("Predicted tags:", predicted_tags)

actual_tags = "cardio"  # Replace with the actual fragment SMILES in order to test accuracy
#actual_tags = "antiinfective"  # Replace with the actual fragment SMILES in order to test accuracy

# Calculate string similarity
string_sim = string_similarity(predicted_tags, actual_tags)
print("String Similarity:", string_sim)
