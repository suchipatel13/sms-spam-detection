# SMS Spam Detection using Machine Learning

## Project Overview
This project builds machine learning models to classify SMS messages as either spam or ham. The goal is to compare multiple classification models and determine which performs best for spam detection.

## Dataset
The project uses the SMS Spam Collection Dataset.

- Source: UCI Machine Learning Repository / Kaggle
- Total messages: 5,572
- Classes:
  - ham
  - spam

If the dataset is not included in this repository, download it and save it as `spam.csv` in the project folder.

## Models Used
- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

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

## Results Summary
The SVM model performed best.

- SVM Accuracy: 98.65%
- SVM Precision: 98.55%
- SVM Recall: 91.28%
- SVM F1-score: 94.77%

Naive Bayes and Logistic Regression also performed well, but both had lower recall for spam detection.

## How to Run

### 1. Install dependencies
```bash
pip3 install -r requirements.txt