import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    eps = 1e-9
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

class MLP:
    def __init__(self, layer_sizes, lr=0.01):
        self.lr = lr
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def forward(self, X):
        self.z = []
        self.a = [X]
        
        for i in range(len(self.weights) - 1):
            z = self.a[-1] @ self.weights[i] + self.biases[i]
            self.z.append(z)
            self.a.append(relu(z))
        
        z = self.a[-1] @ self.weights[-1] + self.biases[-1]
        self.z.append(z)
        self.a.append(softmax(z))
        return self.a[-1]

    def backward(self, X, y):
        m = X.shape[0]
        dz = self.a[-1] - y
        
        for i in reversed(range(len(self.weights))):
            dw = self.a[i].T @ dz / m
            db = np.mean(dz, axis=0, keepdims=True)
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db
            if i > 0:
                dz = (dz @ self.weights[i].T) * relu_grad(self.z[i-1])

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = cross_entropy(y, y_pred)
            self.backward(X, y)
            
            if epoch % 10 == 0:
                # print(f"Epoch {epoch}, loss = {loss:.4f}")
                continue

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

data = pd.read_csv('data.csv')
X = data.drop(columns=['quality']).values
y_raw = data['quality'].values

y = np.zeros((len(y_raw), len(np.unique(y_raw))))
y[np.arange(len(y_raw)), y_raw] = 1


epochs_vector = [100, 200, 400, 800, 1600, 3200, 6400, 12800]

for epochs in epochs_vector:
    acc = 0
    for i in range(1, 6):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42+i)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42+i)
        mlp = MLP(layer_sizes=[X.shape[1], 16, 8, y.shape[1]], lr=0.02)
        mlp.train(X_train, y_train, epochs=epochs)
        y_pred_test = mlp.predict(X_test)
        y_true_test = np.argmax(y_test, axis=1)

        acc += accuracy_score(y_true_test, y_pred_test)/5 
    print(f"Accuracy for {epochs} epochs: {acc:.4f}")
