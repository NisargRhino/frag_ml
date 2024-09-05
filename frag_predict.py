import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Load the trained model and tokenizer
model = RobertaForMaskedLM.from_pretrained('enter file path')
tokenizer = RobertaTokenizer.from_pretrained('enter file path')
model.eval()

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

# Example usage
"""
def cleanup_molecule_rdkit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("couldnt convert")
        return None
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)
"""
new_drug_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Replace with your input SMILES
predicted_fragment_smiles = predict_fragment_smiles(new_drug_smiles, model, tokenizer)

#cleaned_fragment_smiles = cleanup_molecule_rdkit(predicted_fragment_smiles)
print("Predicted Fragment SMILES:", predicted_fragment_smiles)

# If you have the actual fragment SMILES for the input drug, you can calculate the Tanimoto similarity
actual_fragment_smiles = "CC(=O)c1ccccc1C"  # Replace with the actual fragment SMILES
similarity = tanimoto_similarity(predicted_fragment_smiles, actual_fragment_smiles)
print("Tanimoto Similarity:", similarity)

