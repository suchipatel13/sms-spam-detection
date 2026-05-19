# SMS Spam Detection using Machine Learning

## Project Overview
This project builds machine learning models to classify SMS messages as either spam or ham. The goal is to compare multiple classification models, test the effect of duplicate messages, and determine which approach performs best for spam detection.

## Dataset
The project uses the SMS Spam Collection Dataset.

- Source: UCI Machine Learning Repository / Kaggle
- Total messages: 5,572
- Ham messages: 4,825
- Spam messages: 747
- Duplicate label/message rows removed: 403
- Deduplicated messages: 5,169
- Deduplicated ham messages: 4,516
- Deduplicated spam messages: 653
- Classes:
  - ham
  - spam

If the dataset is not included in this repository, download it and save it as `spam.csv` in the project folder.

## Models Used
- Baseline Most Frequent Classifier
- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

## Baseline Model
The baseline model uses `DummyClassifier(strategy="most_frequent")`, which always predicts the majority class (`ham`). It provides a reference point for model comparison and shows why accuracy alone can be misleading on an imbalanced dataset.

## Preprocessing
The following preprocessing steps were applied:
- Convert text to lowercase
- Remove punctuation
- Remove numbers
- Remove extra spaces
- Transform text into numerical features using TF-IDF

## Evaluation Metrics
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- 5-fold stratified cross-validation

## Duplicate Removal Experiment
The project evaluates each model on both the original dataset and a deduplicated dataset. Removing 403 duplicate label/message rows reduced the dataset from 5,572 to 5,169 messages and produced a more conservative estimate of spam detection performance.

## Cross-Validation
The project uses 5-fold stratified cross-validation for each model on both the original and deduplicated datasets. TF-IDF vectorization is fit inside each fold through a scikit-learn pipeline to avoid data leakage.

## Generated Outputs
Running `spam_detection_project.py` creates the following outputs:
- `outputs/class_distribution.png`
- `outputs/model_comparison.png`
- `outputs/model_comparison.csv`
- `outputs/duplicate_experiment_comparison.csv`
- `outputs/cross_validation_results.csv`
- Confusion matrix PNG files for each model and experiment
- `spam_classifier_model.pkl`
- `tfidf_vectorizer.pkl`

## Results Summary
The SVM model performed best overall.

- Original dataset SVM accuracy: 98.65%
- Original dataset SVM recall: 91.28%
- Original dataset SVM F1-score: 94.77%
- Deduplicated dataset SVM accuracy: 98.07%
- Deduplicated dataset SVM recall: 85.50%
- Deduplicated dataset SVM F1-score: 91.80%

Naive Bayes and Logistic Regression also performed well, but both had lower recall for spam detection.

## Key Findings
- SVM was the strongest and most stable model across the original dataset, deduplicated dataset, and cross-validation results.
- Removing duplicates lowered spam recall and F1-score, indicating that duplicates can inflate performance estimates.
- The baseline classifier achieved high accuracy by predicting only `ham`, but it had 0 spam recall and 0 F1-score.
- Recall and F1-score are more informative than accuracy for this imbalanced spam detection task.

## How to Run

### 1. Install dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Run the project
```bash
python3 spam_detection_project.py
```
