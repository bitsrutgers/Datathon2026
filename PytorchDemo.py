# Simple PyTorch demo: learn a line (hours studied -> exam score)

import torch
import torch.nn as nn

# 1) Fake dataset (small, easy to understand)
# x = hours studied
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
# y = exam score (roughly 10*x + 50, with tiny noise)
y = torch.tensor([[60.0], [70.0], [79.0], [90.0], [98.0]])

# 2) Define a tiny model: a single linear layer y = w*x + b
model = nn.Linear(in_features=1, out_features=1)

# 3) Define "how wrong we are" (loss) and how we improve (optimizer)
loss_fn = nn.MSELoss()  # mean squared error
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4) Training loop
for epoch in range(1, 501):
    # Forward pass: predict y from x
    y_pred = model(x)

    # Compute loss
    loss = loss_fn(y_pred, y)

    # Backprop: compute gradients
    optimizer.zero_grad()
    loss.backward()

    # Update weights
    optimizer.step()

    # Print progress occasionally
    if epoch % 100 == 0:
        w = model.weight.item()
        b = model.bias.item()
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.2f} | w: {w:.2f} | b: {b:.2f}")

# 5) Make a simple prediction
new_hours = torch.tensor([[6.0]])
pred_score = model(new_hours).item()
print(f"\nPrediction: If someone studies 6 hours, predicted score ≈ {pred_score:.1f}")
