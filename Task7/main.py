import pandas as pd
import numpy as np
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import math

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

class NaiveBayesClassifier:
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.total_words = defaultdict(int)
        self.vocab = set()
        self.class_priors = {}

    def fit(self, X, y):
        for text, label in zip(X, y):
            self.class_counts[label] += 1
            words = [w for w in preprocess(text) if len(w) > 0]

            for word in words:
                self.word_counts[label][word] += 1
                self.total_words[label] += 1
                self.vocab.add(word)

        total_docs = sum(self.class_counts.values())
        for label in self.class_counts:
            self.class_priors[label] = self.class_counts[label] / total_docs

    def predict_one(self, text):
        words = preprocess(text)
        scores = {}

        for label in self.class_counts:
            log_prob = math.log(self.class_priors[label])
            for word in words:
                word_count = self.word_counts[label].get(word, 0)
                prob = (word_count + 1) / (self.total_words[label] + len(self.vocab))
                log_prob += math.log(prob)
            scores[label] = log_prob

        return max(scores, key=scores.get)

    def predict(self, X):
        return [self.predict_one(text) for text in X]

def main():
    df = pd.read_csv("spam.csv", encoding="latin-1")

    X = df["v2"]
    y = df["v1"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=33, stratify=y
    )

    model = NaiveBayesClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision (spam):", precision_score(y_test, y_pred, pos_label="spam"))
    print("Recall (spam):", recall_score(y_test, y_pred, pos_label="spam"))
    print("F1-score (spam):", f1_score(y_test, y_pred, pos_label="spam"))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=["spam", "ham"]))
    counter = 0
    print('Ham most ofter words:')
    for k, v in sorted(model.word_counts['ham'].items(), key = lambda x: int(x[1]), reverse=True):
        print(k, v)
        counter += 1
        if counter > 10:
            break
    counter = 0
    print('Spam most ofter words:')
    for k, v in sorted(model.word_counts['spam'].items(), key = lambda x: int(x[1]), reverse=True):
        print(k, v)
        counter += 1
        if counter > 10:
            break
if __name__ == "__main__":
    main()
