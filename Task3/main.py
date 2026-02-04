import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

def entropy(y):
    counts = Counter(y)
    total = len(y)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * np.log2(p)
    return ent

def information_gain(X_col, y):
    base_entropy = entropy(y)
    values = set(X_col)
    cond_entropy = 0.0
    total = len(y)
    for v in values:
        idx = X_col == v
        cond_entropy += (np.sum(idx) / total) * entropy(y[idx])
    return base_entropy - cond_entropy

class Node:
    def __init__(self, attribute=None, label=None, majority=None):
        self.attribute = attribute
        self.children = {}
        self.label = label
        self.majority = majority

    def is_leaf(self):
        return self.label is not None

def id3(X, y, attributes, max_depth, depth=0):

    if len(set(y)) == 1:
        return Node(label=y.iloc[0])

    if len(attributes) == 0 or depth == max_depth:
        majority = Counter(y).most_common(1)[0][0]
        return Node(label=majority)

    gains = [(information_gain(X[attr], y), attr) for attr in attributes]
    _, best_attr = max(gains, key=lambda x: x[0])

    majority = Counter(y).most_common(1)[0][0]
    node = Node(attribute=best_attr, majority=majority)

    for value in set(X[best_attr]):
        idx = X[best_attr] == value
        if np.sum(idx) == 0:
            node.children[value] = Node(label=majority)
        else:
            new_attrs = [a for a in attributes if a != best_attr]
            node.children[value] = id3(
                X.loc[idx], y.loc[idx],
                new_attrs, max_depth, depth + 1
            )

    return node

def predict_one(node, sample):
    if node.is_leaf():
        return node.label

    value = sample[node.attribute]
    if value in node.children:
        return predict_one(node.children[value], sample)
    else:
        return node.majority


def predict(node, X):
    return np.array([predict_one(node, X.iloc[i]) for i in range(len(X))])

def discretize(d):
    df = d.copy()
    df = df.drop(columns=['id'])
    df['age'] = pd.cut(df['age'], bins=5, labels=False)
    df['height'] = pd.cut(df['height'], bins=5, labels=False)
    df['weight'] = pd.cut(df['weight'], bins=5, labels=False)
    df['ap_hi'] = pd.cut(df['ap_hi'], bins=5, labels=False)
    df['ap_lo'] = pd.cut(df['ap_lo'], bins=5, labels=False)
    df['cholesterol'] = pd.cut(df['cholesterol'], bins=4, labels=False)
    df['gluc'] = pd.cut(df['gluc'], bins=4, labels=False)
    # df['smoke'] = pd.cut(df['smoke'], bins = 2)
    # df['alco'] = pd.cut(df['alco'], bins = 2)
    # df['active'] = pd.cut(df['active'], bins = 2)

    return df

df = pd.read_csv('cardio_train.csv', sep=';')

dff = discretize(df)
X = dff.drop(columns=['cardio'])
y = dff['cardio']

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

attributes = list(X_train.columns)

best_depth = None
best_acc = 0

for depth in range(1, 9):
    tree = id3(X_train, y_train, attributes, max_depth=depth)
    y_pred = predict(tree, X_val)
    acc = np.mean(y_pred == y_val.values)
    print(f"Głębokość {depth}: accuracy walidacyjna = {acc:.4f}")
    print(depth)
    if acc > best_acc:
        best_acc = acc
        best_depth = depth

print(f"\nNajlepsza głębokość: {best_depth}")

tree = id3(X_train, y_train, attributes, max_depth=best_depth)
y_test_pred = predict(tree, X_test)
test_acc = np.mean(y_test_pred == y_test.values)

print(f"Accuracy testowa: {test_acc:.4f}")

