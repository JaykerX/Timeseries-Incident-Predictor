import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data = np.load("data_I200.npz")

X_test = torch.tensor(data["X_test"], dtype=torch.float32)
y_test = data["y_test"]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return torch.sigmoid(self.fc(h[-1])).squeeze()

model = Model()
model.load_state_dict(torch.load("model.pt"))
model.eval()

thresh = 0.7

with torch.no_grad():
    preds = model(X_test).numpy()
    preds = (preds > thresh).astype(int)

print(f"Threshold: {thresh}")
print("Accuracy:", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds))
print("Recall:", recall_score(y_test, preds))
print("F1:", f1_score(y_test, preds))

