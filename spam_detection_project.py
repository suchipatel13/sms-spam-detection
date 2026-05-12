import pandas as pd
import string
import re
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Load dataset
# Make sure your CSV file is named spam.csv and is in the same folder
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only needed columns
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

print("First 5 rows:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Preprocessing demo example
sample_message = "FREE entry!!! Win $1000 NOW!!!"
cleaned_sample = clean_text(sample_message)

print("\nPreprocessing Demo:")
print("Original Message:", sample_message)
print("Cleaned Message :", cleaned_sample)

df["cleaned_message"] = df["message"].apply(clean_text)

# Convert labels to numbers
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# Split data
X = df["cleaned_message"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining set size:", len(X_train))
print("Testing set size:", len(X_test))

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Models
models = {
    # Majority-class baseline shows whether real models beat simply predicting "ham".
    "Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent"),
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

# Train and evaluate
results = []

for model_name, model in models.items():
    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# Compare results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-Score", ascending=False)

print("\nFinal Model Comparison:")
print(results_df.to_string(index=False))

# Save the best model (SVM has the highest F1-score)
best_model = models["SVM"]
best_model_name = "SVM"

print("\n" + "=" * 50)
print(f"Saving best model: {best_model_name}")
print("=" * 50)

joblib.dump(best_model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✓ Model saved as: spam_classifier_model.pkl")
print("✓ Vectorizer saved as: tfidf_vectorizer.pkl")

# Predict on new messages
def predict_spam(message):
    """
    Classify a message as spam or not spam.
    
    Args:
        message (str): The SMS message to classify
        
    Returns:
        str: 'spam' or 'not spam'
    """
    # Load model and vectorizer if not already loaded
    if not hasattr(predict_spam, 'model'):
        predict_spam.model = joblib.load("spam_classifier_model.pkl")
        predict_spam.vectorizer = joblib.load("tfidf_vectorizer.pkl")
    
    cleaned = clean_text(message)
    vectorized = predict_spam.vectorizer.transform([cleaned])
    prediction = predict_spam.model.predict(vectorized)[0]
    return "spam" if prediction == 1 else "not spam"

# Demo: Test the predict function
print("\n" + "=" * 50)
print("Testing Predict Function")
print("=" * 50)

test_messages = [
    "Congratulations! You've won a free prize. Claim now!",
    "Hey, how are you doing today?",
    "URGENT: Click here to verify your account or it will be closed!",
    "See you at the meeting tomorrow at 3pm"
]

for msg in test_messages:
    result = predict_spam(msg)
    print(f"\nMessage: {msg}")
    print(f"Classification: {result}")


