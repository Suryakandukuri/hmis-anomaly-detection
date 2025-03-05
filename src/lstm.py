#%%
# # import libraries
import pandas as pd
import polars as pl
import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
from pathlib import Path
from sklearn.model_selection import train_test_split

# define data paths
RAW_DATA_PATH = Path(__file__).parent.parent / "data"/ "raw"
INTERIM_DATA_PATH = Path(__file__).parent.parent / "data"/ "interim"
REPORTS_PATH = Path(__file__).parent.parent / "reports"/ "visualisations"
# %%
# hmis_data
hmis_data = pl.read_csv(str(INTERIM_DATA_PATH) + "/hmis-filtered.csv")
# selected features
feature_columns = pl.read_csv(str(INTERIM_DATA_PATH) + "/feature_importance.csv")

# demographic columns
demographic_columns = [
    "state_name", "district_name", "subdistrict_name", "sector", "date",
    "state_code", "district_code", "subdistrict_code"
]

# create model_data with demographic columns and selected features 
model_data = hmis_data.select(demographic_columns + feature_columns["Feature"].to_list())
model_data = model_data.with_columns(
    pl.col("date").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f").dt.date()
)


#%%
# Define split years
train_end = "2019-12-31"
val_end = "2020-12-31"
# Convert string date thresholds to Polars Date type
train_end_date = pl.lit(train_end).str.to_date("%Y-%m-%d")
val_end_date = pl.lit(val_end).str.to_date("%Y-%m-%d")

# Skip conversion if already in Date format!
if model_data.schema["date"] != pl.Date:
    model_data = model_data.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))

# Split the dataset
train_data = model_data.filter(pl.col("date") <= train_end_date)
val_data = model_data.filter((pl.col("date") > train_end_date) & (pl.col("date") <= val_end_date))
test_data = model_data.filter(pl.col("date") > val_end_date)

# Print dataset sizes
print(f"Train: {train_data.shape}, Validation: {val_data.shape}, Test: {test_data.shape}")

# %%
from sklearn.preprocessing import MinMaxScaler

# Select only numeric feature columns
numeric_columns = model_data.select(pl.exclude(["date", "state_name", "district_name", "subdistrict_name", "sector", "state_code", "district_code", "subdistrict_code"])).columns

# Initialize scaler and fit on training data only
scaler = MinMaxScaler()
scaler.fit(train_data[numeric_columns].to_numpy())

# Apply transformation
train_scaled = train_data.with_columns(pl.DataFrame(scaler.transform(train_data[numeric_columns].to_numpy()), schema=numeric_columns))
val_scaled = val_data.with_columns(pl.DataFrame(scaler.transform(val_data[numeric_columns].to_numpy()), schema=numeric_columns))
test_scaled = test_data.with_columns(pl.DataFrame(scaler.transform(test_data[numeric_columns].to_numpy()), schema=numeric_columns))

# %%
import numpy as np

def create_sequences(data, sequence_length=12):
    X, y = [], []
    data_array = data.to_numpy()  # Convert to NumPy

    for i in range(len(data) - sequence_length):
        X.append(data_array[i:i + sequence_length, :])  # Input: past `sequence_length` months
        y.append(data_array[i + 1:i + sequence_length + 1, :])  # Output: next `sequence_length` months

    return np.array(X), np.array(y)

# Create sequences
X_train, y_train = create_sequences(train_scaled[numeric_columns])
X_val, y_val = create_sequences(val_scaled[numeric_columns])
X_test, y_test = create_sequences(test_scaled[numeric_columns])

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

# %%
import torch

# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
X_val_torch = torch.tensor(X_val, dtype=torch.float32)
y_val_torch = torch.tensor(y_val, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32)

print(f"PyTorch tensors: {X_train_torch.shape}, {y_train_torch.shape}")

# %%
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, num_layers=2):
        super(LSTMAutoencoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Decoder LSTM
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer to reconstruct original feature size (64 → 20)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        print(f"Input to LSTM Shape: {x.shape}")  # Debugging line

        # Encoding
        _, (hidden, cell) = self.encoder(x)

        # Decoder input (using zero tensor with correct shape)
        decoder_input = torch.zeros((x.shape[0], x.shape[1], self.hidden_size), device=x.device)

        # Decoding
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        print(f"Output from LSTM Shape: {decoder_output.shape}")  # Debugging line

        # Final FC layer
        out = self.fc(decoder_output)
        print(f"Output from Decoder Shape: {out.shape}")  # Debugging line

        return out

# Model Parameters
input_size = 20  # Number of features
hidden_size = 64  # LSTM hidden units
num_layers = 2  # LSTM layers

# Initialize Model
model = LSTMAutoencoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

# Loss Function & Optimizer
criterion = nn.MSELoss()  # Measures reconstruction error
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print Model Summary
print(model)


#%%
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Define the sequence length (e.g., using 12 months for time-series modeling)
sequence_length = 12  # Adjust based on use case
# Select only numeric features
numeric_columns = train_data.select(pl.exclude(["date", "state_name", "district_name", "subdistrict_name", "sector", "state_code", "district_code", "subdistrict_code"])).columns

# Convert to NumPy (excluding non-numeric columns)
data_numpy = train_data[numeric_columns].to_numpy().astype(np.float32)

# Define num_features (number of selected indicators)
num_features = data_numpy.shape[1]  # Number of columns (excluding date & IDs)

# Ensure correct shape: (samples, sequence_length, num_features)
num_samples = data_numpy.shape[0] - sequence_length
X_sequences = np.array([
    data_numpy[i : i + sequence_length] for i in range(num_samples)
])

# Convert to PyTorch tensor
train_data_tensor = torch.tensor(X_sequences, dtype=torch.float32)

# Define batch size and create DataLoader
batch_size = 32
train_loader = DataLoader(TensorDataset(train_data_tensor), batch_size=batch_size, shuffle=True)

# Print Shapes
print(f"Train Data Shape: {train_data_tensor.shape}")  # Should be (samples, sequence_length, num_features)


# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for X_batch in train_loader:
        X_batch = X_batch[0].to(device)  # Extract tensor from tuple
        optimizer.zero_grad()

        y_pred = model(X_batch)
        loss = criterion(y_pred, X_batch)  # Reconstruction loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")




# %%
