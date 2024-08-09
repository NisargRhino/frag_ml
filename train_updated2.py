import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig, get_linear_schedule_with_warmup
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# Load the dataset
file_path = "/Users/nisargshah/Documents/cs/ml4/mTORcanonical.csv"
data_org = pd.read_csv(file_path, nrows=100)

# Custom dataset class
class SMILESDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        drug_smiles = self.df.iloc[idx]['DRUG SMILES']
        fragment_smiles = self.df.iloc[idx]['FRAG_SMILES']

        inputs = self.tokenizer(drug_smiles, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        
        fragment_inputs = self.tokenizer(fragment_smiles, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        fragment_ids = fragment_inputs['input_ids'].squeeze(0)

        labels = fragment_ids.clone()
        
        return {
            **inputs,
            'labels': labels,
            'actual_fragment_smiles': fragment_smiles
        }

# Initialize tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
config = RobertaConfig.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
model = RobertaForMaskedLM.from_pretrained('seyonec/ChemBERTa-zinc-base-v1', config=config)

# Enable DataParallel to utilize multiple GPUs
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model.to(device)

# Create dataset and dataloader
dataset = SMILESDataset(data_org, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Define optimizer, loss function, and learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
total_steps = len(dataloader) * 10  # 10 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = CrossEntropyLoss()

# Lists to store training metrics
losses = []
tanimoto_similarities = []

# Function to check if a SMILES string is valid
def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

# Function to calculate Tanimoto similarity
def tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# Training loop
model.train()
for epoch in range(10):  # 10 epochs
    epoch_losses = []
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'actual_fragment_smiles'}
        outputs = model(**inputs)
        loss = outputs.loss
        
        loss.mean().backward()
        optimizer.step()
        scheduler.step()

        epoch_losses.append(loss.mean().item())
        print(f"Epoch: {epoch}, Loss: {loss.mean().item()}")
    
    losses.append(epoch_losses)

# Save the trained model and tokenizer
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained("./model")
tokenizer.save_pretrained("./tokenizer")
joblib.dump(config, "./config.pkl")

# Evaluation
true_values = []
predicted_values = []
valid_tanimoto_similarities = []

model.eval()
with torch.no_grad():
    for batch in dataloader:
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'actual_fragment_smiles'}
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)

        # Decode the predicted SMILES and the actual SMILES
        predicted_smiles = tokenizer.decode(predictions[0], skip_special_tokens=True)
        actual_smiles = batch['actual_fragment_smiles'][0]

        # Check if the predicted SMILES is valid
        if is_valid_smiles(predicted_smiles):
            true_values.append(actual_smiles)
            predicted_values.append(predicted_smiles)

            # Calculate Tanimoto similarity only for valid predictions
            similarity = tanimoto_similarity(actual_smiles, predicted_smiles)
            valid_tanimoto_similarities.append(similarity)

# Calculate mean Tanimoto similarity for valid predictions
mean_tanimoto_similarity = np.mean(valid_tanimoto_similarities)

print(f"Mean Tanimoto Similarity (Valid SMILES): {mean_tanimoto_similarity}")

mean_losses = [np.mean(epoch_losses) for epoch_losses in losses]

# Calculate mean Tanimoto similarity per epoch
epoch_mean_similarities = []
for i in range(10):
    start_idx = i * len(dataloader)
    end_idx = start_idx + len(dataloader)
    mean_similarity = np.mean(valid_tanimoto_similarities[start_idx:end_idx])
    epoch_mean_similarities.append(mean_similarity)

# Plot the results
plt.figure(figsize=(14, 6))

# Plot for Training Loss per Epoch
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), mean_losses, marker='o', color='b', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.xticks(range(1, 11))
plt.grid(True)
plt.legend()

# Plot for Mean Tanimoto Similarity per Epoch
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), epoch_mean_similarities, marker='o', color='g', label='Mean Tanimoto Similarity')
plt.xlabel('Epoch')
plt.ylabel('Mean Tanimoto Similarity')
plt.title('Mean Tanimoto Similarity Over Epochs')
plt.xticks(range(1, 11))
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("mTORfigure.png")
