import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Dataset (Expanded for better visualization)
# Training Data: The "History" the AI learns from
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
y_train = torch.tensor([[60.0], [65.0], [75.0], [82.0], [90.0], [93.0]])

# Test Data: "Future" points the AI hasn't seen yet
x_test = torch.tensor([[7.0], [8.0]])
y_test = torch.tensor([[96.0], [99.0]])

# 2. Model, Loss, and Optimizer
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3. Training Loop
for epoch in range(500):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. PREDICTION PHASE
model.eval()
with torch.no_grad():
    # Predict for the whole range to draw the line
    x_range = torch.linspace(0, 10, 100).reshape(-1, 1)
    predictions = model(x_range)

# 5. VISUALIZATION
plt.figure(figsize=(10, 6))

# Plot the historical training data
plt.scatter(x_train, y_train, color='blue', label='Historical Data (Train)')

# Plot the "Future" test data
plt.scatter(x_test, y_test, color='green', marker='s', label='Unseen Data (Test)')

# Plot the AI's predictive trend line
plt.plot(x_range, predictions, color='red', linestyle='--', label='AI Predictive Trend')

plt.title('Predictive Analytics: Hours vs. Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
