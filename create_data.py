import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)

T = 50000
I = 200
W = 50
H = 20

t = np.arange(T)

series = np.stack([
    0.5 * np.sin(0.02 * t) + 0.2 * np.random.randn(T),
    0.3 * np.cos(0.015 * t) + 0.2 * np.random.randn(T),
    0.1 * t / T + 0.2 * np.random.randn(T)
], axis=1)
series = (series - series.mean()) / series.std()

labels = np.zeros(T)

for _ in range(I):
    start = np.random.randint(10, T - 50)
    duration = np.random.randint(20, 80)
    end = start + duration
    trend = np.linspace(0, 3, 10)[:, None]
    series[start-10:start] += trend
    series[start:end] += np.random.uniform(3, 5)
    labels[start:end] = 1

X, y = [], []

for i in range(W, T - H):
    X.append(series[i-W:i])
    y.append(1 if labels[i:i+H].any() else 0)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

np.savez(
    "data.npz",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

print("Saved dataset to data.npz")