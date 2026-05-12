import joblib
import string
import re

# Load the pre-trained model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Clean text function (same as in main script)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_spam(message):
    """
    Classify a message as spam or not spam.
    
    Args:
        message (str): The SMS message to classify
        
    Returns:
        str: 'spam' or 'not spam'
    """
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "spam" if prediction == 1 else "not spam"

# Interactive testing
if __name__ == "__main__":
    print("=" * 60)
    print("SMS Spam Detector - Message Classification Tool")
    print("=" * 60)
    print("Type 'quit' to exit\n")
    
    while True:
        message = input("Enter message to classify: ").strip()
        
        if message.lower() == "quit":
            print("Goodbye!")
            break
        
        if not message:
            print("Please enter a message.\n")
            continue
        
        result = predict_spam(message)
        print(f"Classification: {result.upper()}\n")
