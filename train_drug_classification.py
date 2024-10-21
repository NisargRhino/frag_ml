import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig, get_linear_schedule_with_warmup
import torch.optim as optim
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# Load the dataset
data_csv_file = "./drug_classification_data_df.csv"
data_org = pd.read_csv(data_csv_file)

# Custom dataset class
class SMILESDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        drug_smiles = self.df.iloc[idx]['smiles']
        tags = self.df.iloc[idx]['tags']

        inputs = self.tokenizer(drug_smiles, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}

        tags_inputs = self.tokenizer(tags, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        tags_ids = tags_inputs['input_ids'].squeeze(0)

        labels = tags_ids.clone()
                    
        return {
            **inputs,
            'labels': labels,
            'actual_tags': tags
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

# Function to calculate Tanimoto similarity
def tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# Function to validate SMILES
def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Training loop with SMILES validation
model.train()
for epoch in range(10):  # 10 epochs
    epoch_losses = []
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'actual_tags'}
        outputs = model(**inputs)
        loss = outputs.loss

        # Decode the predicted SMILES
        predictions = outputs.logits.argmax(dim=-1)
        predicted_tags = tokenizer.decode(predictions[0], skip_special_tokens=True)
        
        # Validate the predicted SMILES
        #if not is_valid_smiles(predicted_smiles):
        #    loss += torch.tensor(1.0, device=device)  # Penalize invalid SMILES

        loss.mean().backward()
        optimizer.step()
        scheduler.step()

        epoch_losses.append(loss.mean().item())
        print(f"Epoch: {epoch}, Loss: {loss.mean().item()}")
    
    losses.append(epoch_losses)

# Save the trained model and tokenizer
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained("./model_drug_classification")
tokenizer.save_pretrained("./tokenizer_drug_classification")
joblib.dump(config, "./config.pkl")

# Evaluation
true_values = []
predicted_values = []

model.eval()
with torch.no_grad():
    for batch in dataloader:
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'actual_tags'}
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)

        # Decode the predicted tag and the actual tag
        predicted_tag = tokenizer.decode(predictions[0], skip_special_tokens=True)
        actual_tag = batch['actual_tags'][0]

        true_values.append(actual_tag)
        predicted_values.append(predicted_tag)

        # Calculate Tanimoto similarity
        #similarity = tanimoto_similarity(actual_smiles, predicted_smiles)
        #tanimoto_similarities.append(similarity)

# Calculate mean Tanimoto similarity
#mean_tanimoto_similarity = np.mean(tanimoto_similarities)

#print(f"Mean Tanimoto Similarity: {mean_tanimoto_similarity}")

mean_losses = [np.mean(epoch_losses) for epoch_losses in losses]

# Calculate mean Tanimoto similarity per epoch
#epoch_mean_similarities = []
#for i in range(10):
#    start_idx = i * len(dataloader)
#    end_idx = start_idx + len(dataloader)
#    mean_similarity = np.mean(tanimoto_similarities[start_idx:end_idx])
#    epoch_mean_similarities.append(mean_similarity)

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
#plt.subplot(1, 2, 2)
#plt.plot(range(1, 11), epoch_mean_similarities, marker='o', color='g', label='Mean Tanimoto Similarity')
#plt.xlabel('Epoch')
#plt.ylabel('Mean Tanimoto Similarity')
#plt.title('Mean Tanimoto Similarity Over Epochs')
#plt.xticks(range(1, 11))
#plt.ylim(0, 1)
#plt.grid(True)
#plt.legend()

plt.tight_layout()
plt.savefig("drug_classification_train_results.png")
