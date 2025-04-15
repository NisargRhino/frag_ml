import sys
from io import StringIO
import deepsmiles
from deepsmiles import Converter
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data.dataset import Dataset
from torch.nn.modules.loss import CrossEntropyLoss
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig, get_linear_schedule_with_warmup
import torch.optim as optim
import joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
import rdkit.Chem
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import random_split
from tqdm import tqdm
import warnings
import os
from collections import defaultdict

now = datetime.now()
formatted_time = now.strftime('%y-%m-%d-%H-%M-%S')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
RDLogger.DisableLog('rdApp.warning')  # Hide warnings
RDLogger.DisableLog('rdApp.error')  # Hide errors
RDLogger.DisableLog('rdApp.info')  # Hide info
RDLogger.DisableLog('rdApp.debug')  # Hide debug

NUM_EPOCHS = 10
INVALID_SMILES_PENALTY = 1
L2_lambda = 0.000006

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')
warnings.simplefilter("ignore")
file_path = ""
data_org = pd.read_csv(file_path)


def write_error_counts_to_file(filename):
    with open(filename, 'w') as file:
        for phase, counts in error_counts.items():
            file.write(f"Error counts for {phase} phase:\n")
            for error_type, count in counts.items():
                file.write(f"  {error_type}: {count}\n")
            file.write("\n")


def clean_smiles(smiles):
    if is_valid_smiles(smiles):
        return smiles
    else:
        return


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return default_collate(batch)


def convert_Deep_to_SMILES(deepsmiles_str):
    converter = Converter(rings=True, branches=True)
    try:
        standard_smiles = converter.decode(deepsmiles_str)
    except deepsmiles.exceptions.DecodeError as e:
        print(f"Decode error for {deepsmiles_str}: {e}")
        standard_smiles = None
    # Smiles = converter.decode(DeepSMILES)
    return standard_smiles


def tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return rdkit.Chem.DataStructs.TanimotoSimilarity(fp1, fp2)


def is_valid_smiles_logger(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def is_valid_deepsmiles(deepsmiles):
    try:
        converter = Converter(rings=True, branches=True)
        smiles = converter.decode(deepsmiles)
    except Exception as e:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


class SMILESDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        drug_smiles = self.df.iloc[idx]['DRUG_SMILES_DEEP']
        fragment_smiles = self.df.iloc[idx]['FRAG_SMILES_DEEP']
        if not drug_smiles:
            return None
        inputs = self.tokenizer(drug_smiles, max_length=self.max_length, padding='max_length', truncation=True,  return_tensors="pt")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        fragment_inputs = self.tokenizer(fragment_smiles, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        fragment_ids = fragment_inputs['input_ids'].squeeze(0)

        # Mask pad tokens in labels so they are ignored by loss
        labels = fragment_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            **inputs,
            'labels': labels,
            'actual_drug_smiles': drug_smiles,
            'actual_fragment_smiles': fragment_smiles}


finetuned_path = ''
tokenizer = RobertaTokenizer.from_pretrained(finetuned_path)
config = RobertaConfig.from_pretrained(finetuned_path)
model = RobertaForMaskedLM.from_pretrained(finetuned_path, config=config)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)

dataset = SMILESDataset(data_org, tokenizer)
size_d = len(dataset)
print("Lenght of the dataset", size_d)
train_size = int(0.75 * size_d)
val_size = int(0.15 * size_d)
test_size = len(dataset) - val_size - train_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_dataloader) * NUM_EPOCHS  # 10 epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = CrossEntropyLoss()
losses = []
tanimoto_similarities = []
converter = Converter(rings=True, branches=True)


def tanimoto_similarity_deepsmiles(deepsmiles1, deepsmiles2):
    """
    Compute Tanimoto similarity given two DeepSMILES strings.
    The DeepSMILES are first decoded to standard SMILES, and then
    molecular fingerprints are generated from the SMILES.
    If decoding fails or the molecules cannot be parsed, returns 0.0.
    """
    try:
        # Decode DeepSMILES strings to standard SMILES
        smiles1 = converter.decode(deepsmiles1)
        smiles2 = converter.decode(deepsmiles2)
    except Exception as e:
        # print(f"Error decoding DeepSMILES: {e}")
        return 0.0

    # Parse the SMILES into RDKit molecule objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0

    # Generate Morgan fingerprints (radius=2, 2048 bits) for the molecules
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


error_counts = {
    "train": {
        "unclosed ring": 0,
        "duplicated ring closure": 0,
        "extra close parentheses": 0,
        "extra open parentheses": 0,
        "non-ring atom": 0,
        "can't kekulize": 0,
        "other": 0
    },
    "val": {
        "unclosed ring": 0,
        "duplicated ring closure": 0,
        "extra close parentheses": 0,
        "extra open parentheses": 0,
        "non-ring atom": 0,
        "can't kekulize": 0,
        "other": 0
    },
    "test": {
        "unclosed ring": 0,
        "duplicated ring closure": 0,
        "extra close parentheses": 0,
        "extra open parentheses": 0,
        "non-ring atom": 0,
        "can't kekulize": 0,
        "other": 0
    }
}
error_msgs = []


def is_valid_smiles(smiles, phase):
    old_stderr = sys.stderr
    sio = sys.stderr = StringIO()
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    sys.stderr = old_stderr
    sio.seek(0)
    error_msg = sio.getvalue()
    print("error message:", sio.getvalue())
    phase_errors = error_counts[phase]
    if error_msg:
        if "unclosed ring" in error_msg:
            phase_errors["unclosed ring"] += 1
        elif "duplicated ring closure" in error_msg:
            phase_errors["duplicated ring closure"] += 1
        elif "extra close parentheses" in error_msg:
            phase_errors["extra close parentheses"] += 1
        elif "extra open parentheses" in error_msg:
            phase_errors["extra open parentheses"] += 1
        else:
            print("Not captured ", error_msg)
            phase_errors["other"] += 1
            error_msgs.append(error_msg)
    if mol is not None:
        problems = Chem.DetectChemistryProblems(mol)
        if problems:
            print(problems[0].GetType())
            print(problems[0].Message())
            error_msg = problems[0].Message()
            if "unclosed ring" in error_msg:
                phase_errors["unclosed ring"] += 1
            elif "duplicated ring closure" in error_msg:
                phase_errors["duplicated ring closure"] += 1
            elif "extra close parentheses" in error_msg:
                phase_errors["extra close parentheses"] += 1
            elif "extra open parentheses" in error_msg:
                phase_errors["extra open parentheses"] += 1
            elif "non-ring atom" in error_msg:
                phase_errors["non-ring atom"] += 1
            elif "Can't kekulize" in error_msg:
                phase_errors["can't kekulize"] += 1
            else:
                print("Not captured ", error_msg)
                phase_errors["other"] += 1
                error_msgs.append(error_msg)
    return mol is not None


train_epoch_losses_table = []
train_losses = []
val_losses = []
test_losses = []
train_epoch_losses = []
true_values_epoch = []
predicted_values_epoch = []
t_tanimoto_similarities_per_epoch = []
v_tanimoto_similarities_per_epoch = []
true_values = []
predicted_values = []
loss_values = []
test_losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    mean_t_tanimoto_similarities = []
    mean_v_tanimoto_similarities = []
    train_batch_losses = []

    for batch in tqdm(train_dataloader,
                      desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
        t_tanimoto_similarities_per_batch = []
        optimizer.zero_grad()
        inputs = {key: val.to(device) for key, val in batch.items() if
                  key != 'actual_fragment_smiles' and key != 'actual_drug_smiles'}
        outputs = model(**inputs)
        train_loss = outputs.loss
        predictions = outputs.logits.argmax(dim=-1)
        penalty = 0

        for i in range(len(batch['actual_fragment_smiles'])):
            pred_ids = predictions[i].tolist()
            predicted_smiles = tokenizer.decode(predictions[i], skip_special_tokens=True)
            actual_smiles = batch['actual_fragment_smiles'][i]
            drug_smiles = batch['actual_drug_smiles'][i]
            similarity = tanimoto_similarity_deepsmiles(actual_smiles, predicted_smiles)
            t_tanimoto_similarities_per_batch.append(similarity)
            if not is_valid_deepsmiles(predicted_smiles):
                penalty += torch.tensor(INVALID_SMILES_PENALTY, device=device)
        # l1_norm = sum(p.abs().sum() for p in model.parameters())
        # train_loss += 0.000001 * l1_norm
        # l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        # train_loss += L2_lambda * l2_norm   #Changed train_loss from adding 0.000008 to 0.0000010
        train_loss += penalty
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        train_batch_losses.append(train_loss.mean().item())
        mean_t_tanimoto_similarities.append(np.mean(t_tanimoto_similarities_per_batch))
    t_tanimoto_similarities_per_epoch.append(np.mean(mean_t_tanimoto_similarities))
    train_epoch_losses.append(train_batch_losses)

    model.eval()
    val_loss_over_batch = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Validation {epoch + 1}/{NUM_EPOCHS}"):
            v_tanimoto_similarities_per_batch = []
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'actual_fragment_smiles' and key != 'actual_drug_smiles'}
            outputs = model(**inputs)
            val_loss = outputs.loss

            predictions = outputs.logits.argmax(dim=-1)
            for i in range(len(batch['actual_fragment_smiles'])):
                predicted_smiles = tokenizer.decode(predictions[i], skip_special_tokens=True)
                actual_smiles = batch['actual_fragment_smiles'][i]
                similarity = tanimoto_similarity_deepsmiles(actual_smiles, predicted_smiles)
                v_tanimoto_similarities_per_batch.append(similarity)
                print("Adding tanimoto similarities ", similarity)

            if not is_valid_deepsmiles(predicted_smiles):
                val_loss += torch.tensor(INVALID_SMILES_PENALTY, device=device)

            val_loss_over_batch.append(val_loss.mean().item())
            print("Collected tanimoto similarity", v_tanimoto_similarities_per_batch)
            mean_v_tanimoto_similarities.append(np.mean(v_tanimoto_similarities_per_batch))
    val_loss = np.mean(val_loss_over_batch)
    v_tanimoto_similarities_per_epoch.append(np.mean(mean_v_tanimoto_similarities))
    print(f"Epoch: {epoch + 1}, Losses Train: {np.mean(train_batch_losses)} Val : {val_loss} ")
    train_epoch_losses_table.append((epoch + 1, np.mean(train_batch_losses), val_loss))
    train_losses.append(train_batch_losses)
    val_losses.append(val_loss_over_batch)
model.eval()

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Testing"):
        test_loss = 0.0
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'actual_fragment_smiles' and key != 'actual_drug_smiles'}
        outputs = model(**inputs)
        loss = outputs.loss
        predictions = outputs.logits.argmax(dim=-1)
        predicted_smiles = tokenizer.decode(predictions[0], skip_special_tokens=True)
        # if not is_valid_smiles(predicted_smiles, 'test'):
        #    test_loss += torch.tensor(INVALID_SMILES_PENALTY, device=device)
        actual_smiles = batch['actual_fragment_smiles'][0]
        test_loss += loss.mean().item()
        test_losses.append(test_loss)
        true_values.append(actual_smiles)
        predicted_values.append(predicted_smiles)
        loss_values.append(test_loss)

test_losses = [float(x) if isinstance(x, torch.Tensor) else x for x in test_losses]
epoch_losses_df = pd.DataFrame(train_epoch_losses_table, columns=['epoch', 'train_loss', 'val_loss'])
epoch_losses_df.to_csv("epoch_losses_with_val_" + str(formatted_time) + ".csv", index=False)
test_losses_df = pd.DataFrame(list(zip(true_values, predicted_values, test_losses)), columns=['true_smile', 'predicted_smile', 'loss'])
test_losses_df.to_csv("test_losses_with_val_" + str(formatted_time) + ".csv", index=False)

model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained("./model" + str(formatted_time))
tokenizer.save_pretrained("./tokenizer" + str(formatted_time))
joblib.dump(config, "./config" + str(formatted_time) + ".pkl")
true_values = []
predicted_values = []
model.eval()
batch_index = []
train_epoch_mean_similarities = []
print("Batches ")

train_mean_losses = [np.mean(epoch_losses) for epoch_losses in train_losses]
val_mean_losses = [np.mean(epoch_losses) for epoch_losses in val_losses]

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, NUM_EPOCHS + 1), train_mean_losses, marker='o', color='b', label='Training Loss')
plt.plot(range(1, NUM_EPOCHS + 1), val_mean_losses, marker='x', color='r', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xticks(range(1, NUM_EPOCHS + 1))
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, NUM_EPOCHS + 1), t_tanimoto_similarities_per_epoch, marker='o', color='b', label='Train Mean Tanimoto Similarity')
plt.plot(range(1, NUM_EPOCHS + 1), v_tanimoto_similarities_per_epoch, marker='x', color='r',  label='Val Mean Tanimoto Similarity')
plt.xlabel('Epoch')
plt.ylabel('Mean Tanimoto Similarity')
plt.title('Mean Tanimoto Similarity Over Epochs')
plt.xticks(range(1, NUM_EPOCHS + 1))
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mTORfigure_train_val" + str(formatted_time) + ".png")

print("Error message counts:")
for phase, counts in error_counts.items():
    print(f"Error counts for {phase} phase:")
    for error_type, count in counts.items():
        print(f"  {error_type}: {count}")
    print()
write_error_counts_to_file('error_counts' + str(formatted_time) + '.txt')
