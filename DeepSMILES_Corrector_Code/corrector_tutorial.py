
import os
import pandas as pd
import argparse
from src.modelling import initialize_model, train_model

parser = argparse.ArgumentParser()
parser.add_argument("--folder_raw", type=str, required=True)
parser.add_argument("--folder_out", type=str, required=True)
parser.add_argument("--data_source", type=str, required=True)
parser.add_argument("--invalid_type", type=str, default="none")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--threshold", type=int, default=200)
parser.add_argument("--training", action="store_true")

args = parser.parse_args()

folder_raw = args.folder_raw
folder_out = args.folder_out
data_source = args.data_source
invalid_type = args.invalid_type
epochs = args.epochs
batch_size = args.batch_size
threshold = args.threshold
training = args.training

# DeepSMILES with validation: skip synthetic generation
train_path = os.path.join(folder_raw, data_source.replace(".csv", "_split_train.csv"))
val_path = os.path.join(folder_raw, data_source.replace(".csv", "_split_val.csv"))

if not os.path.exists(train_path) or not os.path.exists(val_path):
    raise FileNotFoundError(
        "Training or validation file not found. Expected: "
        "RawData/smiles_corrector_training_split_train.csv and "
        "RawData/smiles_corrector_training_split_val.csv"
    )


print("[INFO] Using pre-split DeepSMILES dataset with validation.")
print(f"[INFO] Train file: {train_path}")
print(f"[INFO] Val file: {val_path}")

# Initialize and train
model, out, SRC = initialize_model(
    folder_out, train_path, val_path, batch_size, threshold, epochs
)

if training:
    train_model(model, out, SRC)
