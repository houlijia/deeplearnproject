import torch
import torch.nn as nn
import torch.optim as optim

# dimensions of the input, hidden, and output layers,anf the batch size
n_in, n_h, n_out, batch_size = 10, 5, 1, 10

x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0],
                  [1.0], [1.0], [1.0],
                  [0.0], [0.0], [1.0], [1.0]])  # 目标输出数据
print("Input data x: ", x)
print("Target output y: ", y)

# define the neural network
module = nn.Sequential(
    nn.Linear(n_in, n_h),
    nn.ReLU(),
    nn.Linear(n_h, n_out),
    nn.Sigmoid()
)

# define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(module.parameters(), lr=0.1)

# train the neural network
for epoch in range(50):
    y_pred = module(x)  # forward pass
    loss = criterion(y_pred, y)  # compute the loss
    print("Epoch: ", epoch, " Loss: ", loss.item())  # print the loss

    optimizer.zero_grad()  # clear the gradients
    loss.backward()  # compute the gradients
    optimizer.step()  # update