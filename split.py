import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import io
import random # For shuffling companies
import math # For ceiling division in split calculation
import ast # For potentially reading back stringified lists
import pickle # To save scaler info

# --- Configuration ---
csv_file_path = 'cleaned_file2.csv' # Input CSV file path
# --- Output Paths ---
train_output_path = 'train_agg_regression_60_40.csv' # Updated filename
val_output_path = 'validation_agg_regression_60_40.csv' # Updated filename
# --- Path for scaler info ---
scaler_info_output_path = 'stock_price_scaler_info.pkl'
# --- Updated: Split Ratio ---
validation_split_ratio = 0.4 # Target ratio for validation companies (40%)
# Removed num_val_companies
company_column_name = 'Ticker' # Column name for the company identifier
cik_column_name = 'CIK' # CIK column name
sort_columns = ['Year', 'Quarter'] # Columns to sort by *within each company*

# Columns for input feature sequence (EXCLUDES Year/Quarter and Stock Price)
sequence_feature_columns = ['Total Revenue', 'Net Income', 'Operating Margin', 'EPS']

# Columns to normalize (includes Stock Price for normalization step, but NOT Year/Quarter)
columns_to_normalize = ['Total Revenue', 'Net Income', 'Operating Margin', 'EPS', 'Stock Price']

# Target column name (the actual normalized price)
target_column_name = 'Stock Price' # The column containing the value to predict

# Columns used for initial sorting
primary_sort_cols = [company_column_name] + sort_columns
year_offset = 2000 # Value to subtract from the Year column
random_seed = 42 # For reproducible company shuffling

# --- Load Data ---
try:
    my_df = pd.read_csv(csv_file_path)
    print(f"Successfully loaded data from {csv_file_path}")
    print(f"Total rows loaded: {len(my_df)}")
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading the CSV file: {e}")
    sys.exit(1)

# --- Adjust Year Column ---
if 'Year' in my_df.columns:
    print(f"Subtracting {year_offset} from the 'Year' column...")
    my_df['Year'] = my_df['Year'] - year_offset
    print("'Year' column adjusted.")
else:
    print("Error: 'Year' column not found, cannot adjust.")
    sys.exit(1)

# --- Validate Columns ---
required_cols_initial = list(set([company_column_name, cik_column_name] + sort_columns + ['Stock Price'] + sequence_feature_columns + columns_to_normalize))
if not all(col in my_df.columns for col in required_cols_initial):
    missing = [col for col in required_cols_initial if col not in my_df.columns]
    print(f"Error: Initial required columns missing: {missing}")
    sys.exit(1)

# --- Sort Data ---
print(f"\nSorting data by {primary_sort_cols}...")
my_df_sorted = my_df.sort_values(by=primary_sort_cols).reset_index(drop=True)
print("Data sorted.")
print("-" * 30)

# --- Define Per-Group Normalization Function ---
def normalize_group(group, columns_to_norm):
    scaler = MinMaxScaler()
    group_normalized = group.copy()
    scaler_params = {} # Store params for this group
    for col in columns_to_norm:
        if col in group.columns:
            data_to_scale = group[[col]].values.astype(float)
            if np.isnan(data_to_scale).any() or np.isinf(data_to_scale).any():
                 continue
            valid_data = data_to_scale[~np.isnan(data_to_scale)]
            if len(np.unique(valid_data)) > 1:
                try:
                    scaled_values = scaler.fit_transform(data_to_scale)
                    group_normalized[col] = scaled_values
                    # Store min/max if it's the target column
                    if col == target_column_name:
                         scaler_params = {'min': scaler.data_min_[0], 'max': scaler.data_max_[0]}
                except Exception:
                     pass
            else:
                 group_normalized[col] = 0.0
                 # Store constant value if it's the target column
                 if col == target_column_name and len(valid_data) > 0:
                      constant_val = valid_data[0]
                      scaler_params = {'min': constant_val, 'max': constant_val}

    return group_normalized, scaler_params # Return params along with data

# --- Apply Normalization Per Company & Store Scaler Info ---
print("\nNormalizing specified financial data within each company group and storing scaler info...")
processed_groups = []
stock_price_scaler_info = {} # Dictionary to store {ticker: {'min': min_val, 'max': max_val}}

grouped = my_df_sorted.groupby(company_column_name)

for ticker, group in grouped:
    group_normalized, scaler_params = normalize_group(group, columns_to_normalize)
    processed_groups.append(group_normalized)
    # Store the scaler parameters if they were generated
    if scaler_params:
        stock_price_scaler_info[ticker] = scaler_params

if not processed_groups:
     print("Error: No company groups were processed successfully.")
     sys.exit(1)
df_processed = pd.concat(processed_groups, ignore_index=True)

print("Normalization complete.")
print(f"Scaler info for '{target_column_name}' stored for {len(stock_price_scaler_info)} companies.")
print("-" * 30)

# --- Save Scaler Information ---
try:
    with open(scaler_info_output_path, 'wb') as f:
        pickle.dump(stock_price_scaler_info, f)
    print(f"Stock price scaler information saved to '{scaler_info_output_path}'")
except Exception as e:
    print(f"Error saving scaler information: {e}")
print("-" * 30)


# --- Aggregate Data into Sequences per Company ---
print("Aggregating data into sequences per company...")
aggregated_data = []
cols_for_agg = list(set(sequence_feature_columns + [target_column_name, cik_column_name, company_column_name] + sort_columns))
if not all(col in df_processed.columns for col in cols_for_agg):
     missing = [col for col in cols_for_agg if col not in df_processed.columns]
     print(f"Error: Columns needed for aggregation missing: {missing}")
     sys.exit(1)

grouped_processed = df_processed.groupby([cik_column_name, company_column_name])

for name, group in grouped_processed:
    cik, ticker = name
    group_sorted = group.sort_values(by=sort_columns)
    data_sequence = group_sorted[sequence_feature_columns].values.tolist()
    target_sequence = group_sorted[target_column_name].values.tolist()
    aggregated_data.append({
        cik_column_name: cik,
        company_column_name: ticker,
        'Data_Sequence': str(data_sequence),
        'Target_Sequence': str(target_sequence)
    })

df_aggregated = pd.DataFrame(aggregated_data)
print(f"Aggregation complete. {len(df_aggregated)} companies aggregated.")
print("Aggregated DataFrame head:")
print(df_aggregated.head())
print("-" * 30)

# --- Split Companies into Train/Validation (60/40 Ratio) ---
print(f"\nSplitting companies into training ({1-validation_split_ratio:.0%}) and validation ({validation_split_ratio:.0%}) sets...")
all_companies = df_aggregated[company_column_name].unique().tolist()
num_total_companies = len(all_companies)

# Calculate number of validation companies based on ratio
num_val_companies_calc = math.ceil(num_total_companies * validation_split_ratio)

# Ensure at least 1 company in validation and 1 in training if possible
if num_val_companies_calc >= num_total_companies:
    num_val_companies_calc = num_total_companies - 1 # Keep at least one for training
elif num_val_companies_calc == 0:
     num_val_companies_calc = 1 # Keep at least one for validation

if num_total_companies < 2:
    print(f"Error: Only {num_total_companies} company found. Cannot perform train/validation split.")
    sys.exit(1)
if num_val_companies_calc <= 0 or num_val_companies_calc >= num_total_companies:
     print(f"Error: Calculated number of validation companies ({num_val_companies_calc}) is invalid for total companies ({num_total_companies}).")
     sys.exit(1)


# Shuffle companies
random.seed(random_seed)
random.shuffle(all_companies)

# Select validation and training companies
val_company_list = all_companies[:num_val_companies_calc]
train_company_list = all_companies[num_val_companies_calc:]

print(f"Total companies: {num_total_companies}")
print(f"Training companies ({len(train_company_list)}): {sorted(train_company_list)}")
print(f"Validation companies ({len(val_company_list)}): {sorted(val_company_list)}")

train_data_agg = df_aggregated[df_aggregated[company_column_name].isin(train_company_list)].copy()
val_data_agg = df_aggregated[df_aggregated[company_column_name].isin(val_company_list)].copy()

print("Company splitting complete.")

# --- Output Summary ---
print("\n--- Final Split Summary (Aggregated Company Sequences for Regression - 60/40 Split) ---")
print(f"Rows in final training set (1 per company): {len(train_data_agg)}")
print(f"Rows in final validation set (1 per company): {len(val_data_agg)}")
print("-" * 30)
print(f"Unique companies in training: {train_data_agg[company_column_name].nunique()}")
print(f"Unique companies in validation: {val_data_agg[company_column_name].nunique()}")
print("-" * 30)

# --- Save the aggregated and split data ---
try:
    train_data_agg.to_csv(train_output_path, index=False)
    val_data_agg.to_csv(val_output_path, index=False)
    print(f"\nAggregated training data saved to '{train_output_path}'")
    print(f"Aggregated validation data saved to '{val_output_path}'")
    print("\nWARNING: Sequences are stored as strings in the CSV.")
except Exception as e:
    print(f"\nAn error occurred while saving the aggregated split files: {e}")

print("\n--- Script Finished ---")
