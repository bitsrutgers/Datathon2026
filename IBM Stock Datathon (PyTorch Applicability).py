import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.preprocessing import MinMaxScaler

# 1. LOAD DATA
# Ensure your IBM.csv has columns: Date, Open, High, Low, Close
df = pd.read_csv('IBM.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 2. PREPROCESSING
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 10
X, y = create_sequences(scaled_data, SEQ_LENGTH)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

# 3. DEFINE PYTORCH LSTM MODEL
class IBM_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(IBM_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64)
        c0 = torch.zeros(2, x.size(0), 64)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

model = IBM_LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. TRAINING LOOP
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 5. PREDICTIONS
model.eval()
with torch.no_grad():
    predictions = model(X).numpy()
    predictions = scaler.inverse_transform(predictions)

# 6. CANDLESTICK VISUALIZATION
def plot_candlestick(df, predictions):
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # We plot the last 50 days for clarity
    df_plot = df.iloc[-50:].reset_index()
    preds_plot = predictions[-50:]
    
    for i in range(len(df_plot)):
        # Determine color
        color = 'green' if df_plot.loc[i, 'Close'] >= df_plot.loc[i, 'Open'] else 'red'
        
        # Draw the wick (High/Low)
        ax.vlines(x=i, ymin=df_plot.loc[i, 'Low'], ymax=df_plot.loc[i, 'High'], color='black', linewidth=1)
        
        # Draw the body (Open/Close)
        body_bottom = min(df_plot.loc[i, 'Open'], df_plot.loc[i, 'Close'])
        body_top = max(df_plot.loc[i, 'Open'], df_plot.loc[i, 'Close'])
        rect = Rectangle((i - 0.3, body_bottom), 0.6, body_top - body_bottom, color=color, alpha=0.8)
        ax.add_patch(rect)

    # Plot the LSTM prediction line
    ax.plot(range(len(df_plot)), preds_plot, color='blue', label='AI Predicted Price', linewidth=2)
    
    ax.set_title("IBM Stock Price: Actual Candlesticks vs. PyTorch AI Predictions", fontsize=16)
    ax.set_xticks(range(0, len(df_plot), 5))
    ax.set_xticklabels(df_plot['Date'].dt.strftime('%Y-%m-%d').iloc[::5], rotation=45)
    ax.legend()
    plt.grid(alpha=0.3)
    plt.savefig('ibm_prediction_candlestick.png')
    plt.show()

plot_candlestick(df, predictions)
