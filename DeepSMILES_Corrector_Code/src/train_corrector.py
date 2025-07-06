import sys
import os
import torch

# ✅ Add the parent (project) directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modelling import initialize_model, train_model

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Config ===
folder_out = "Data/"  # Use trailing slash
data_source = "smiles_corrector_training_input_correct"  # filename without .csv
error_source = None  # we're not using external error file now
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold = 100  # max sequence length (adjust if needed)
epochs = 10
layers = 3
batch_size = 16
invalid_type = "none"  # because you already have paired corrupted → correct
num_errors = 1  # unused when invalid_type is "none"

# === Initialize model
model, out, SRC = initialize_model(
    folder_out=folder_out,
    data_source=data_source,
    error_source=error_source,
    device=device,
    threshold=threshold,
    epochs=epochs,
    layers=layers,
    batch_size=batch_size,
    invalid_type=None,  # ✅ disables synthetic corruption
    num_errors=num_errors,
    validation_step=False
)

# === Train the model
train_model(model, out, assess=False)
