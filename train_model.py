import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def load_data(path):
    data = np.load(path)
    X = torch.tensor(data["X_train"], dtype=torch.float32)
    y = torch.tensor(data["y_train"], dtype=torch.float32)
    return X, y

def create_dataloader(X, y, batch_size=256, shuffle=True):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class Model(nn.Module):
    def __init__(self, input_size=3, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze()


def create_model(input_size=3, hidden_size=32):
    return Model(input_size=input_size, hidden_size=hidden_size)


def compute_pos_weight(y):
    return torch.tensor([(1 - y.mean()) / y.mean()])


def train(model, loader, epochs=5, lr=1e-3, pos_weight=None, device="cpu"):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1:2d} | Loss: {total_loss / len(loader):.4f}")


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")


def load_model(path, input_size=3, hidden_size=32, device="cpu"):
    model = Model(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":

    data_path = "data_big.npz"
    model_path = "model_big.pt"

    X, y = load_data(data_path)
    loader = create_dataloader(X, y, batch_size=512)

    model = create_model(input_size=3, hidden_size=32)

    pos_weight = compute_pos_weight(y)

    print("train model...")
    train(
        model,
        loader,
        epochs=5,
        lr=1e-3,
        pos_weight=pos_weight,
        device= "cuda" if torch.cuda.is_available() else "cpu"
    )

    save_model(model, model_path)