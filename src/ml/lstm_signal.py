import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.ml.lgbm_signal import build_features

SEQ_LEN = 60  # Look back 60 trading days


class ReturnSequenceDataset(Dataset):
    """PyTorch dataset: for each (date, ticker), return (60-day feature sequence, target)."""

    def __init__(self, feature_tensor, target_series, seq_len=SEQ_LEN):
        self.X = feature_tensor       # shape: (dates, tickers, features)
        self.y = target_series        # shape: (dates, tickers)
        self.seq_len = seq_len
        self.valid_indices = []

        T, N, F = self.X.shape
        for t in range(seq_len, T):
            for n in range(N):
                if (not np.isnan(self.y[t, n])
                        and not np.any(np.isnan(self.X[t - seq_len:t, n]))):
                    self.valid_indices.append((t, n))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t, n = self.valid_indices[idx]
        x = torch.tensor(self.X[t - self.seq_len:t, n], dtype=torch.float32)
        y = torch.tensor(self.y[t, n], dtype=torch.float32)
        return x, y


class LSTMSignalModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)  # last timestep output


def train_lstm_signal(ret_wide, horizon=21, epochs=30, batch_size=256):
    """
    Train LSTM and return predicted signal (wide format).
    Uses GPU if available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    features = build_features(ret_wide)
    feature_names = list(features.keys())

    # Stack features into 3D tensor: (dates, tickers, features)
    tickers = ret_wide.columns.tolist()
    dates = ret_wide.index

    feat_array = np.stack(
        [features[f].reindex(index=dates, columns=tickers).values
         for f in feature_names],
        axis=-1,
    )  # shape: (T, N, F)

    target = ret_wide.shift(-horizon).reindex(index=dates, columns=tickers).values

    # Train/val split (chronological — no shuffling)
    T = len(dates)
    split = int(T * 0.8)

    train_ds = ReturnSequenceDataset(feat_array[:split], target[:split])
    val_ds = ReturnSequenceDataset(feat_array[split:], target[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LSTMSignalModel(input_size=len(feature_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience, patience_counter = 5, 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()

        print(f"Epoch {epoch+1:02d} | train loss: {train_loss/len(train_loader):.6f} "
              f"| val loss: {val_loss/len(val_loader):.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "data/lstm_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Generate predictions on all data
    model.load_state_dict(torch.load("data/lstm_best.pt"))
    model.eval()

    all_ds = ReturnSequenceDataset(feat_array, target)
    loader = DataLoader(all_ds, batch_size=512, shuffle=False)

    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            preds.extend(model(X_batch.to(device)).cpu().numpy())

    # Rebuild wide format from valid_indices
    pred_df_rows = []
    for i, (t, n) in enumerate(all_ds.valid_indices):
        pred_df_rows.append({
            "date": dates[t], "ticker": tickers[n], "lstm_raw": preds[i],
        })

    pred_df = pd.DataFrame(pred_df_rows)
    signal_wide = pred_df.pivot(index="date", columns="ticker", values="lstm_raw")
    signal_wide.index = pd.to_datetime(signal_wide.index)
    return signal_wide