import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# Import padding utils and Dataset class
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error, mean_absolute_error # Regression metrics
import matplotlib.pyplot as plt
# Removed seaborn
import os
import sys
import copy
import math # For ceiling division
import ast # To parse stringified lists from CSV
import pickle # To load scaler info and save objects
import json # To save config

# --- Configuration ---
# --- Updated Input File Paths ---
train_csv_path = 'train_agg_regression_60_40.csv' # Path to aggregated training data (60/40 split)
val_csv_path = 'validation_agg_regression_60_40.csv'   # Path to aggregated validation data (60/40 split)
# --- Path to saved scaler info ---
scaler_info_path = 'stock_price_scaler_info.pkl' # Path to the saved scaler dictionary

# --- Output Directory ---
output_dir = 'training_output_bilstm_regressor' # Directory to save results
os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
print(f"Output artifacts will be saved to: '{output_dir}'")

# Model and Training Parameters
config = {
    'epochs': 100,
    'batch_size': 4, # Keep batch size small
    'hidden_size': 1024, # Using scaled up model
    'num_layers': 6, # Using 6 LSTM layers
    'dropout_rate': 0.3, # Using lower dropout rate
    'learning_rate': 1e-5, # May need tuning for regression
    'early_stopping_patience': 15,
    'train_csv_path': train_csv_path,
    'val_csv_path': val_csv_path,
    'scaler_info_path': scaler_info_path,
    'expected_num_features': 4, # ['Total Revenue', 'Net Income', 'Operating Margin', 'EPS']
    'target_name': 'Normalized Stock Price',
    'company_col': 'Ticker',
    'cik_col': 'CIK',
    'data_seq_col': 'Data_Sequence',
    'target_seq_col': 'Target_Sequence',
}

# Column names (convenience variables from config)
company_col = config['company_col']
cik_col = config['cik_col']
data_seq_col = config['data_seq_col']
target_seq_col = config['target_seq_col']
expected_num_features = config['expected_num_features']

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config['device'] = str(device) # Add device to config
print(f"Using device: {device}")

# --- Save Configuration ---
config_save_path = os.path.join(output_dir, 'training_config.json')
try:
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Training configuration saved to '{config_save_path}'")
except Exception as e:
    print(f"Warning: Could not save training configuration: {e}")
print("-" * 30)


# --- Load Scaler Information ---
try:
    with open(scaler_info_path, 'rb') as f:
        scaler_info_dict = pickle.load(f)
    print(f"Successfully loaded scaler info from '{scaler_info_path}' for {len(scaler_info_dict)} companies.")
except FileNotFoundError:
    print(f"Warning: Scaler info file not found at '{scaler_info_path}'. Cannot un-normalize results.")
    scaler_info_dict = None # Set to None if not found
except Exception as e:
    print(f"Warning: Error loading scaler info: {e}. Cannot un-normalize plots.")
    scaler_info_dict = None
print("-" * 30)

# --- Custom Dataset for Aggregated Company Sequences (Regression) ---
class AggregatedCompanyRegressionDataset(Dataset):
    def __init__(self, csv_path, data_col, regression_target_seq_col):
        self.data_sequences = []
        self.target_sequences = []
        self.company_identifiers = [] # Store ticker for scaler lookup

        print(f"Loading aggregated data for dataset from '{csv_path}'...")
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: The file '{csv_path}' was not found.")
            raise
        except Exception as e:
            print(f"An error occurred loading '{csv_path}': {e}")
            raise

        if data_col not in df.columns or regression_target_seq_col not in df.columns:
            print(f"Error: CSV must contain columns '{data_col}' and '{regression_target_seq_col}'.")
            raise ValueError("Missing required sequence columns in CSV")
        if company_col not in df.columns:
             print(f"Warning: CSV missing '{company_col}'. Cannot map results back to specific companies for un-normalization.")


        for index, row in df.iterrows():
            ticker = row.get(company_col, None)
            try:
                data_seq_list = ast.literal_eval(row[data_col])
                target_seq_list = ast.literal_eval(row[regression_target_seq_col])

                data_tensor = torch.tensor(data_seq_list, dtype=torch.float32)
                target_tensor = torch.tensor(target_seq_list, dtype=torch.float32).unsqueeze(1) # Add feature dim

                # --- Data Validation ---
                if data_tensor.ndim != 2 or data_tensor.shape[1] != expected_num_features:
                    # print(f"Warning: Skipping row {index} (Ticker: {ticker}) due to unexpected feature shape: {data_tensor.shape}")
                    continue
                if target_tensor.ndim != 2 or target_tensor.shape[1] != 1: # Target should be (seq_len, 1)
                    # print(f"Warning: Skipping row {index} (Ticker: {ticker}) due to unexpected target shape: {target_tensor.shape}")
                    continue
                if data_tensor.shape[0] != target_tensor.shape[0]:
                    # print(f"Warning: Skipping row {index} (Ticker: {ticker}) due to sequence length mismatch: Data {data_tensor.shape[0]}, Target {target_tensor.shape[0]}")
                    continue
                if data_tensor.shape[0] < 2: # Need at least 2 time steps
                    # print(f"Warning: Skipping row {index} (Ticker: {ticker}) due to insufficient sequence length: {data_tensor.shape[0]}")
                    continue
                # --- End Data Validation ---

                self.data_sequences.append(data_tensor)
                self.target_sequences.append(target_tensor)
                self.company_identifiers.append(ticker) # Store ticker directly

            except (ValueError, SyntaxError, TypeError) as e:
                print(f"Warning: Skipping row {index} (Ticker: {ticker}) due to parsing error: {e}")
            except Exception as e:
                 print(f"Warning: Skipping row {index} (Ticker: {ticker}) due to unexpected error during processing: {e}")

        if not self.data_sequences:
             print("Error: No valid sequences could be loaded or parsed from the CSV.")
             raise ValueError("Dataset creation failed.")
        print(f"Created dataset with {len(self.data_sequences)} company sequences.")

    def __len__(self):
        return len(self.data_sequences)

    def __getitem__(self, idx):
        # Return company ticker along with data
        return self.data_sequences[idx], self.target_sequences[idx], self.company_identifiers[idx]


# --- Custom Collate Function for Padding ---
def collate_fn(batch):
    """Pads sequences within a batch and sorts by length."""
    features = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    company_tickers = [item[2] for item in batch] # Get tickers (list of strings)
    lengths = torch.tensor([len(seq) for seq in features])

    # Pad features with 0.0, targets with a value ignored by the loss (-100.0)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-100.0) # Padding value ignored by loss

    # Sort by lengths descending for pack_padded_sequence
    lengths, sort_idx = lengths.sort(descending=True)
    features_padded = features_padded[sort_idx]
    targets_padded = targets_padded[sort_idx]
    # Sort the company tickers list according to the sort indices
    # Ensure sort_idx is on CPU and convert to list for indexing
    sorted_company_tickers = [company_tickers[i] for i in sort_idx.cpu().tolist()]

    return features_padded, targets_padded, lengths, sorted_company_tickers


# --- 1. Create Datasets ---
try:
    print("--- Creating Training Dataset ---")
    train_dataset = AggregatedCompanyRegressionDataset(train_csv_path, data_seq_col, target_seq_col)
    print("\n--- Creating Validation Dataset ---")
    val_dataset = AggregatedCompanyRegressionDataset(val_csv_path, data_seq_col, target_seq_col)
except Exception as e:
     print(f"Failed to create datasets: {e}")
     sys.exit(1) # Exit if datasets cannot be created

if len(train_dataset) == 0 or len(val_dataset) == 0:
    print("Error: Training or validation dataset is empty after parsing. Exiting.")
    sys.exit(1)

# --- 2. Create DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

print(f"\nCreated DataLoaders.")
print(f"Training companies/sequences: {len(train_dataset)}")
print(f"Validation companies/sequences: {len(val_dataset)}")
print(f"Training batches per epoch: {math.ceil(len(train_dataset)/config['batch_size'])}")
print(f"Validation batches: {math.ceil(len(val_dataset)/config['batch_size'])}")
print("-" * 30)

# --- 3. Build BiLSTM Model (Regressor) ---
class BiLSTMRegressorSeq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(BiLSTMRegressorSeq, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Apply dropout between LSTM layers only if num_layers > 1
        lstm_dropout = dropout_rate if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout_rate) # Dropout after LSTM layer
        # Fully connected layers: LSTM output -> Hidden -> Output
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size) # BiLSTM outputs size hidden_size * 2
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size) # Final output size

    def forward(self, x, lengths):
        # Pack sequence -> LSTM -> Unpack sequence
        # Move lengths to CPU for packing if they are on GPU
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, (hn, cn) = self.lstm(x_packed)
        # pad_packed_sequence unpacks the output, handling padding
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Apply dropout and pass through FC layers
        out = self.dropout(output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out) # (batch, seq_len, output_size=1)
        return out

# Instantiate the model
input_size = expected_num_features
output_size = 1 # Predicting a single value (normalized price) per time step
model = BiLSTMRegressorSeq(
    input_size,
    config['hidden_size'],
    config['num_layers'],
    output_size,
    config['dropout_rate']
).to(device)

print("\n--- Model Summary ---")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")
print("-" * 30)

# --- 4. Define Loss Function (MSE) and Optimizer ---
# Use reduction='none' to calculate loss per element, then average manually using a mask
criterion = nn.MSELoss(reduction='none')
print(f"Using MSELoss (reduction='none') for regression.")
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# --- 5. Training Loop ---
print("\n--- Training Model ---")
train_losses = []
val_losses = [] # Validation loss calculated on the FULL sequence
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None
last_epoch = 0 # Track the last completed epoch

for epoch in range(config['epochs']):
    last_epoch = epoch # Store the current epoch number
    model.train()
    running_train_loss = 0.0
    total_train_steps = 0 # Count total valid (non-padded) time steps in training loss calc

    for i, (inputs, targets, lengths, _) in enumerate(train_loader): # Ignore tickers during training steps
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, lengths) # Normalized predictions (batch, max_len, 1)

        # --- Create Mask for Loss Calculation ---
        # Mask should be True for valid steps, False for padding
        # targets has shape (batch, max_len, 1)
        # lengths has shape (batch,)
        max_len = targets.size(1)
        # Create indices (0, 1, ..., max_len-1) and broadcast to (batch, max_len)
        indices = torch.arange(max_len).unsqueeze(0).to(device) # (1, max_len)
        # Compare indices with lengths; True where index < length
        mask = indices < lengths.unsqueeze(1).to(device) # (batch, max_len)
        # Unsqueeze mask to match target dimensions (batch, max_len, 1)
        mask = mask.unsqueeze(2)
        # --- End Mask Creation ---

        loss_unreduced = criterion(outputs, targets) # Calculate MSE for all elements (batch, max_len, 1)
        # Apply mask: Keep loss where mask is True, set to 0 where False
        loss_masked = torch.where(mask, loss_unreduced, torch.tensor(0.0, device=device))

        # Average the loss ONLY over the valid (masked) steps
        num_valid_steps = mask.sum()

        if num_valid_steps > 0:
            loss = loss_masked.sum() / num_valid_steps # Average loss per valid step
            loss.backward()
            optimizer.step()
            # Accumulate the sum of losses and the count of valid steps for epoch average
            running_train_loss += loss_masked.sum().item() # Sum of losses in batch
            total_train_steps += num_valid_steps.item()   # Total valid steps in batch
        # else: batch contained only padding or invalid sequences, skip update

    # Calculate average training loss for the epoch
    epoch_train_loss = running_train_loss / total_train_steps if total_train_steps > 0 else 0.0
    train_losses.append(epoch_train_loss)

    # --- Validation Loop (Evaluate on FULL Sequence) ---
    model.eval()
    running_val_loss_full = 0.0
    total_val_steps_full = 0 # Count valid steps in validation
    with torch.no_grad():
        for inputs, targets, lengths, _ in val_loader: # Ignore tickers during validation loss calc
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, lengths) # Normalized predictions (batch, max_len, 1)

            # --- Create mask for ALL valid steps (same logic as training) ---
            max_len = targets.size(1)
            indices = torch.arange(max_len).unsqueeze(0).to(device) # (1, max_len)
            eval_mask = indices < lengths.unsqueeze(1).to(device) # (batch, max_len)
            eval_mask = eval_mask.unsqueeze(2) # (batch, max_len, 1)
            # --- End Mask Creation ---

            loss_unreduced = criterion(outputs, targets) # MSE on normalized values
            loss_masked = torch.where(eval_mask, loss_unreduced, torch.tensor(0.0, device=device))
            num_valid_eval_steps = eval_mask.sum().item()

            if num_valid_eval_steps > 0:
                batch_loss_sum = loss_masked.sum()
                running_val_loss_full += batch_loss_sum.item()
                total_val_steps_full += num_valid_eval_steps

    # Calculate average validation loss over ALL valid steps across all validation batches
    epoch_val_loss = running_val_loss_full / total_val_steps_full if total_val_steps_full > 0 else float('inf') # Use inf if no valid steps
    val_losses.append(epoch_val_loss)

    print(f"Epoch [{epoch+1}/{config['epochs']}], Train Loss: {epoch_train_loss:.6f}, Val Loss (MSE - Norm - Full Seq): {epoch_val_loss:.6f}")

    # --- Early Stopping (based on full sequence validation loss) ---
    if total_val_steps_full > 0 and epoch_val_loss < best_val_loss: # Only update if validation produced results
        print(f"Validation loss improved from {best_val_loss:.6f} to {epoch_val_loss:.6f}. Saving model state.")
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        # Save the best model state dictionary
        best_model_state = copy.deepcopy(model.state_dict())
        # Save best model weights immediately to file
        best_model_weights_path = os.path.join(output_dir, 'best_model_weights.pth')
        try:
            torch.save(best_model_state, best_model_weights_path)
            # print(f"Best model weights saved to '{best_model_weights_path}'") # Optional: print every time
        except Exception as e:
            print(f"Warning: Failed to save best model weights at epoch {epoch+1}: {e}")
    elif total_val_steps_full > 0: # Only increment if validation produced results
        epochs_no_improve += 1
        print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

    # Trigger early stopping
    if epochs_no_improve >= config['early_stopping_patience']:
        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
        break # Exit training loop


# --- Save Final Model State ---
final_model_weights_path = os.path.join(output_dir, 'final_model_weights.pth')
try:
    torch.save(model.state_dict(), final_model_weights_path)
    print(f"\nFinal model weights (Epoch {last_epoch+1}) saved to '{final_model_weights_path}'")
except Exception as e:
    print(f"Warning: Could not save final model weights: {e}")


# --- Save Loss History ---
loss_history = {'train_loss': train_losses, 'validation_loss': val_losses}
loss_history_path = os.path.join(output_dir, 'loss_history.pkl')
try:
    with open(loss_history_path, 'wb') as f:
        pickle.dump(loss_history, f)
    print(f"Loss history saved to '{loss_history_path}'")
except Exception as e:
    print(f"Warning: Could not save loss history: {e}")


# --- Load the best model state for evaluation ---
if best_model_state:
    print("\nLoading best model state for final evaluation.")
    model.load_state_dict(best_model_state)
else:
    print("\nWarning: No best model state was saved (likely no improvement). Using final model state for evaluation.")
    # The 'model' object already holds the final state in this case


# --- Function to Un-normalize ---
def unnormalize_value(scaled_value, min_val, max_val):
    """Applies inverse min-max scaling."""
    if max_val is None or min_val is None: # Handle cases where scaler info might be missing
        return None
    if max_val == min_val:
        return max_val # Avoid division by zero, return the constant value
    return scaled_value * (max_val - min_val) + min_val

# --- 6. Evaluate the Best Model (on FULL Sequence) ---
print("\n--- Evaluating Best Model on Validation Set (Full Sequence) ---")
model.eval()
final_val_loss_mse_norm = 0.0
all_preds_flat_unnorm = []
all_labels_flat_unnorm = []
# Store predictions and labels per company for plotting
results_by_company = {} # {ticker: {'preds_norm': [], 'actual_norm': [], 'preds_unnorm': [], 'actual_unnorm': []}}

total_eval_steps = 0
with torch.no_grad():
    for inputs, targets_norm, lengths, company_tickers in val_loader: # Get tickers
        inputs, targets_norm = inputs.to(device), targets_norm.to(device)
        outputs_norm = model(inputs, lengths) # Normalized predictions (batch, max_len, 1)

        # --- Create mask for ALL valid steps (same logic as before) ---
        max_len = targets_norm.size(1)
        indices = torch.arange(max_len).unsqueeze(0).to(device)
        eval_mask = indices < lengths.unsqueeze(1).to(device) # (batch, max_len)
        eval_mask_unsqueezed = eval_mask.unsqueeze(2) # (batch, max_len, 1)
        # --- End Mask Creation ---

        # Calculate loss for this batch
        loss_unreduced = criterion(outputs_norm, targets_norm) # MSE loss on normalized
        loss_masked = torch.where(eval_mask_unsqueezed, loss_unreduced, torch.tensor(0.0, device=device))
        batch_loss_sum = loss_masked.sum()
        num_valid_eval_steps = eval_mask_unsqueezed.sum().item()

        if num_valid_eval_steps > 0:
            final_val_loss_mse_norm += batch_loss_sum.item() # Accumulate sum of losses
            total_eval_steps += num_valid_eval_steps      # Accumulate count of valid steps

            # Process results batch by batch, company by company for storage and un-normalization
            for i in range(len(lengths)): # Iterate through sequences in the batch
                ticker = company_tickers[i]
                if ticker is None: # Handle potential missing tickers
                    ticker = f"Unknown_{i}" # Assign a placeholder name

                length = lengths[i].item()
                # Get predictions and labels for the valid length of this sequence
                # Ensure slicing is correct: [:length] gives elements from 0 to length-1
                preds_norm_seq = outputs_norm[i, :length, 0].cpu().numpy()
                labels_norm_seq = targets_norm[i, :length, 0].cpu().numpy()

                # Initialize storage for this company if not seen before
                if ticker not in results_by_company:
                    results_by_company[ticker] = {
                        'preds_norm': [], 'actual_norm': [],
                        'preds_unnorm': [], 'actual_unnorm': []
                    }

                results_by_company[ticker]['preds_norm'].extend(preds_norm_seq.tolist())
                results_by_company[ticker]['actual_norm'].extend(labels_norm_seq.tolist())

                # Un-normalize if possible
                preds_unnorm_seq = []
                labels_unnorm_seq = []
                can_unnormalize = False
                if scaler_info_dict and ticker is not None and ticker in scaler_info_dict:
                    scaler_params = scaler_info_dict[ticker]
                    # Check if 'min' and 'max' exist and are not None
                    min_v = scaler_params.get('min')
                    max_v = scaler_params.get('max')
                    if min_v is not None and max_v is not None:
                         can_unnormalize = True
                         for p_norm, l_norm in zip(preds_norm_seq, labels_norm_seq):
                             p_unnorm = unnormalize_value(p_norm, min_v, max_v)
                             l_unnorm = unnormalize_value(l_norm, min_v, max_v)
                             # Only append if unnormalization was successful (returned a number)
                             if p_unnorm is not None: preds_unnorm_seq.append(p_unnorm)
                             if l_unnorm is not None: labels_unnorm_seq.append(l_unnorm)
                         # Ensure we added the same number of unnormalized points as normalized ones
                         if len(preds_unnorm_seq) != len(preds_norm_seq) or len(labels_unnorm_seq) != len(labels_norm_seq):
                             can_unnormalize = False # Mark as failed if lengths mismatch
                             print(f"Warning: Unnormalization length mismatch for {ticker}. Skipping unnormalized storage for this sequence.")


                # Store unnormalized results if successful
                if can_unnormalize:
                    results_by_company[ticker]['preds_unnorm'].extend(preds_unnorm_seq)
                    results_by_company[ticker]['actual_unnorm'].extend(labels_unnorm_seq)


# --- Calculate final metrics ---
final_metrics = {}
if total_eval_steps > 0:
    final_val_loss_mse_norm /= total_eval_steps # Average MSE over all valid steps
    print(f"\n--- Metrics Calculated on Full Sequence ---")
    print(f"Validation Loss (MSE - Normalized - Full Seq): {final_val_loss_mse_norm:.6f}")
    final_metrics['val_loss_mse_normalized'] = final_val_loss_mse_norm

    # Flatten results across all companies for overall unnormalized metrics
    all_preds_flat_unnorm = [p for ticker_data in results_by_company.values() for p in ticker_data.get('preds_unnorm', [])]
    all_labels_flat_unnorm = [l for ticker_data in results_by_company.values() for l in ticker_data.get('actual_unnorm', [])]

    # Ensure we have valid, non-empty, equal-length lists for unnormalized metrics
    if len(all_labels_flat_unnorm) > 0 and len(all_labels_flat_unnorm) == len(all_preds_flat_unnorm):
        all_preds_flat_unnorm = np.array(all_preds_flat_unnorm)
        all_labels_flat_unnorm = np.array(all_labels_flat_unnorm)

        final_val_mae_unnorm = mean_absolute_error(all_labels_flat_unnorm, all_preds_flat_unnorm)
        final_val_mse_unnorm = mean_squared_error(all_labels_flat_unnorm, all_preds_flat_unnorm)
        final_val_rmse_unnorm = np.sqrt(final_val_mse_unnorm)
        print(f"Validation MAE (Original Scale - Full Seq): {final_val_mae_unnorm:.4f}")
        print(f"Validation RMSE (Original Scale - Full Seq): {final_val_rmse_unnorm:.4f}")
        final_metrics['val_mae_original_scale'] = final_val_mae_unnorm
        final_metrics['val_rmse_original_scale'] = final_val_rmse_unnorm
        final_metrics['val_mse_original_scale'] = final_val_mse_unnorm

    else:
        print("Could not calculate metrics on original scale (un-normalization likely failed for some/all samples or no samples available).")
        final_metrics['val_mae_original_scale'] = None
        final_metrics['val_rmse_original_scale'] = None
        final_metrics['val_mse_original_scale'] = None

    # --- Save Evaluation Metrics ---
    metrics_save_path = os.path.join(output_dir, 'evaluation_metrics.pkl')
    try:
        with open(metrics_save_path, 'wb') as f:
            pickle.dump(final_metrics, f)
        print(f"Evaluation metrics saved to '{metrics_save_path}'")
    except Exception as e:
        print(f"Warning: Could not save evaluation metrics: {e}")


    # --- 7. Visualize Results (Plot ALL Validation Companies) ---
    print("\n--- Plotting Results for ALL Validation Companies ---")
    # Plot training & validation loss (normalized scale)
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history['train_loss'], label='Train Loss (Full Seq)')
    plt.plot(loss_history['validation_loss'], label='Validation Loss (Full Seq)')
    plt.title('Model Loss During Training (MSE - Normalized)')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    # Save the plot
    loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
    try:
        plt.savefig(loss_plot_path)
        print(f"Loss plot saved to '{loss_plot_path}'")
    except Exception as e:
        print(f"Warning: Could not save loss plot: {e}")
    plt.show() # Display the plot
    plt.close() # Close the figure


    # Plot Predictions vs Actual for each validation company
    print("\nPlotting validation sequences...")
    if not results_by_company:
         print("No validation results available to plot.")

    for ticker, results in results_by_company.items():
        actual_norm = np.array(results.get('actual_norm', []))
        preds_norm = np.array(results.get('preds_norm', []))
        actual_unnorm = np.array(results.get('actual_unnorm', []))
        preds_unnorm = np.array(results.get('preds_unnorm', []))

        # Check if unnormalized results are available and have the same length as normalized
        use_unnormalized = (len(actual_unnorm) > 0 and len(preds_unnorm) > 0 and
                           len(actual_unnorm) == len(actual_norm) and len(preds_unnorm) == len(preds_norm))

        if use_unnormalized:
            plot_actual = actual_unnorm
            plot_preds = preds_unnorm
            plot_title_suffix = "(Original Scale)"
            y_label = "Original Stock Price"
            y_limits = None # Auto-scale for original prices
        elif len(actual_norm) > 0 and len(preds_norm) > 0: # Fallback to normalized if possible
             plot_actual = actual_norm
             plot_preds = preds_norm
             plot_title_suffix = "(Normalized Scale)"
             y_label = "Normalized Stock Price (0-1)"
             y_limits = (-0.1, 1.1) # Fixed scale for normalized
        else:
            print(f"Warning: No valid data (normalized or unnormalized) to plot for Ticker: {ticker}")
            continue # Skip plotting for this ticker

        time_indices = np.arange(len(plot_actual)) # Simple time step index

        # Ensure lengths match before plotting (should be guaranteed by checks above, but safer)
        if len(time_indices) == len(plot_actual) == len(plot_preds):
            plt.figure(figsize=(12, 6))
            plt.plot(time_indices, plot_actual, label=f'Actual Price', color='blue', marker='.')
            plt.plot(time_indices, plot_preds, label=f'Predicted Price', color='red', linestyle='--', marker='.')
            plt.title(f'Validation Prediction vs Actual {plot_title_suffix} - Ticker: {ticker}')
            plt.xlabel('Time Step in Sequence')
            plt.ylabel(y_label)
            if y_limits: plt.ylim(y_limits)
            plt.legend()
            plt.grid(True)

            # Save the plot
            plot_filename = f'validation_plot_{ticker}.png'
            plot_save_path = os.path.join(output_dir, plot_filename)
            try:
                plt.savefig(plot_save_path)
                # print(f"Validation plot for {ticker} saved to '{plot_save_path}'") # Can be verbose
            except Exception as e:
                print(f"Warning: Could not save validation plot for {ticker}: {e}")
            plt.show() # Display the plot
            plt.close() # Close the figure to avoid displaying too many plots at once later

        else:
            # This case should ideally not be reached due to prior checks
            print(f"Warning: Final length mismatch prevented plotting for Ticker: {ticker}. Indices: {len(time_indices)}, Actual: {len(plot_actual)}, Preds: {len(plot_preds)}")


else:
    print("\nEvaluation could not be performed: No valid evaluation steps found across the validation set.")
    # Save empty metrics if evaluation failed
    metrics_save_path = os.path.join(output_dir, 'evaluation_metrics.pkl')
    try:
        with open(metrics_save_path, 'wb') as f:
            pickle.dump({'error': 'No valid evaluation steps found'}, f)
        print(f"Evaluation metrics file saved (indicating evaluation failure) to '{metrics_save_path}'")
    except Exception as e:
        print(f"Warning: Could not save empty evaluation metrics file: {e}")


print(f"\n--- PyTorch BiLSTM Regressor Script Finished ---")
print(f"Output artifacts saved in: '{output_dir}'")