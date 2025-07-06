import torch
import torch.nn as nn
import torch.optim as optim

# Sample data: 4 samples, 2 features each
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])

# Labels: binary output
y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])  # XOR Problem

class SimpleNN(nn.Module):
    def __init__(self) -> None:
        super(SimpleNN, self).__init__()
        self.forwardconnection1 = nn.Linear(2,4)
        self.forwardconnection2 = nn.Linear(4,1)
        self.sigmoid = nn.Sigmoid() # This line initializes the Sigmoid activation function.
        # The Sigmoid function squashes the output to a value between 0 and 1.
        # It’s perfect for binary classification problems because it outputs probabilities.

    def forward(self, x):
        x = torch.relu(self.forwardconnection1(x))
        x = self.sigmoid(self.forwardconnection2(x))
        return x
    

model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad() #It clears the previous gradients before the new backward pass.
    output = model(X)
    loss = criterion(output, y)
    loss.backward() # It computes gradients of the loss w.r.t all the parameters (weights & biases) using backpropagation.
    optimizer.step() # Updates the model parameters (weights and biases) using the computed gradients.
    # This is where learning happens — the optimizer (like Adam) adjusts the weights so that the next prediction should result in lower loss.

    if(epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


with torch.no_grad(): # It tells PyTorch not to track gradients during inference. We don’t need gradients while predicting (only needed for training)
    preds = model(X)
    predicted_classes = preds.round() # It rounds the output probabilities to either 0 or 1.
    print(predicted_classes)