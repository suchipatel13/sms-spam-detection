import pandas as pd
import string
import re
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
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
    classification_report,
    make_scorer
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

# Duplicate messages can inflate performance if the same or near-identical SMS
# appears in both training and testing data, so report and test this explicitly.
duplicate_count = df.duplicated(subset=["label", "message"]).sum()
print("\nDuplicate label/message rows:", duplicate_count)

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

def get_models():
    return {
        # Majority-class baseline shows whether real models beat simply predicting "ham".
        "Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": LinearSVC()
    }


def run_experiment(experiment_name, experiment_df):
    print("\n" + "#" * 70)
    print(f"Experiment: {experiment_name}")
    print("#" * 70)
    print("Dataset size:", len(experiment_df))
    print("Class distribution:")
    print(experiment_df["label"].value_counts())

    # Use the same stratified split settings for both datasets so the comparison
    # isolates the effect of removing duplicate label/message rows.
    X = experiment_df["cleaned_message"]
    y = experiment_df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining set size:", len(X_train))
    print("Testing set size:", len(X_test))

    # Fit TF-IDF on the training set only to avoid letting test-set vocabulary
    # influence the model during evaluation.
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    results = []

    for model_name, model in get_models().items():
        print("\n" + "=" * 50)
        print(model_name)
        print("=" * 50)

        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({
            "Experiment": experiment_name,
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
        print(classification_report(
            y_test,
            y_pred,
            target_names=["ham", "spam"],
            zero_division=0
        ))

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
        disp.plot()
        plt.title(f"Confusion Matrix - {experiment_name} - {model_name}")
        plt.show()

    return results


def save_prediction_model(training_df):
    print("\n" + "=" * 50)
    print("Saving prediction model: SVM")
    print("=" * 50)

    # Train the deployable model on deduplicated data so the saved classifier
    # matches the more conservative evaluation used in the report.
    vectorizer = TfidfVectorizer(stop_words="english")
    X_tfidf = vectorizer.fit_transform(training_df["cleaned_message"])
    y = training_df["label_num"]

    model = LinearSVC()
    model.fit(X_tfidf, y)

    joblib.dump(model, "spam_classifier_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    print("Model saved as: spam_classifier_model.pkl")
    print("Vectorizer saved as: tfidf_vectorizer.pkl")


def predict_spam(message):
    """
    Classify a message as spam or not spam using the saved SVM model.
    """
    if not hasattr(predict_spam, "model"):
        predict_spam.model = joblib.load("spam_classifier_model.pkl")
        predict_spam.vectorizer = joblib.load("tfidf_vectorizer.pkl")

    cleaned = clean_text(message)
    vectorized = predict_spam.vectorizer.transform([cleaned])
    prediction = predict_spam.model.predict(vectorized)[0]
    return "spam" if prediction == 1 else "not spam"


def run_cross_validation(experiment_name, experiment_df):
    print("\n" + "#" * 70)
    print(f"5-Fold Cross-Validation: {experiment_name}")
    print("#" * 70)

    X = experiment_df["cleaned_message"]
    y = experiment_df["label_num"]

    # Cross-validation reduces dependence on one train/test split by rotating
    # which rows are used for testing across five stratified folds.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0)
    }

    cv_results = []

    for model_name, model in get_models().items():
        # The pipeline prevents leakage by fitting TF-IDF separately inside
        # each fold rather than on the full dataset before validation.
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("model", model)
        ])

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring
        )

        accuracy_mean = scores["test_accuracy"].mean()
        accuracy_std = scores["test_accuracy"].std()
        recall_mean = scores["test_recall"].mean()
        recall_std = scores["test_recall"].std()
        f1_mean = scores["test_f1"].mean()
        f1_std = scores["test_f1"].std()

        cv_results.append({
            "Experiment": experiment_name,
            "Model": model_name,
            "Accuracy Mean": accuracy_mean,
            "Accuracy Std": accuracy_std,
            "Recall Mean": recall_mean,
            "Recall Std": recall_std,
            "F1 Mean": f1_mean,
            "F1 Std": f1_std
        })

        print(f"\n{model_name}")
        print("Accuracy:", round(accuracy_mean, 4), "+/-", round(accuracy_std, 4))
        print("Recall  :", round(recall_mean, 4), "+/-", round(recall_std, 4))
        print("F1-Score:", round(f1_mean, 4), "+/-", round(f1_std, 4))

    return cv_results


deduplicated_df = df.drop_duplicates(subset=["label", "message"]).copy()
print("\nDeduplicated dataset size:", len(deduplicated_df), "rows")
print("Rows removed by deduplication:", len(df) - len(deduplicated_df))

all_results = []
all_results.extend(run_experiment("Original Dataset", df))
all_results.extend(run_experiment("Deduplicated Dataset", deduplicated_df))

# Compare results
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(by=["Experiment", "F1-Score"], ascending=[True, False])

print("\nFinal Model Comparison:")
print(results_df.to_string(index=False))

outputs_dir = Path("outputs")
outputs_dir.mkdir(exist_ok=True)
comparison_path = outputs_dir / "duplicate_experiment_comparison.csv"
results_df.to_csv(comparison_path, index=False)
print(f"\nSaved duplicate experiment comparison to: {comparison_path}")

save_prediction_model(deduplicated_df)

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

all_cv_results = []
all_cv_results.extend(run_cross_validation("Original Dataset", df))
all_cv_results.extend(run_cross_validation("Deduplicated Dataset", deduplicated_df))

cv_results_df = pd.DataFrame(all_cv_results)
cv_results_df = cv_results_df.sort_values(by=["Experiment", "F1 Mean"], ascending=[True, False])

print("\nCross-Validation Summary:")
print(cv_results_df.to_string(index=False))

cv_path = outputs_dir / "cross_validation_results.csv"
cv_results_df.to_csv(cv_path, index=False)
print(f"\nSaved cross-validation results to: {cv_path}")
