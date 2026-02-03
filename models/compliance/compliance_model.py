from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
import pandas as pd
from typing import List

class ComplianceModel:
    """
    Compliance Risk Model: NLP classifier to detect violations in audit logs or documents.
    """
    def __init__(self, model_dir: str = "models/compliance"):
        self.model_dir = model_dir
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.classifier = LogisticRegression()

    def train(self, texts: List[str], labels: List[int]):
        print(f"Training Compliance Model on {len(texts)} documents...")
        print("Vectorizing text data with TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        X = self.vectorizer.fit_transform(texts)
        print("Training Logistic Regression classifier...")
        self.classifier = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        self.classifier.fit(X, labels)
        self.save_model()
        print("Compliance Model training complete.")

    def predict_risk(self, texts: List[str]) -> List[float]:
        X = self.vectorizer.transform(texts)
        # Probability of class 1 (violation)
        return self.classifier.predict_proba(X)[:, 1]

    def save_model(self):
        joblib.dump(self.vectorizer, os.path.join(self.model_dir, "vectorizer.joblib"))
        joblib.dump(self.classifier, os.path.join(self.model_dir, "classifier.joblib"))

    def load_model(self):
        self.vectorizer = joblib.load(os.path.join(self.model_dir, "vectorizer.joblib"))
        self.classifier = joblib.load(os.path.join(self.model_dir, "classifier.joblib"))
