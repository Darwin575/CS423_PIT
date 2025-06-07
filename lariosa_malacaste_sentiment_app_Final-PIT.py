# Import necessary libraries
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# Define stopwords and preserve intensifiers
default_stopwords = set(stopwords.words('english'))
intensifiers = {"very", "really", "so", "too", "extremely", "incredibly", 
                "absolutely", "completely", "utterly", "highly", "remarkably", "awfully", "not"}
updated_stopwords = default_stopwords - intensifiers

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def custom_analyzer(text):
    """
    Preprocesses the input text by lowercasing, removing punctuation (except digits),
    tokenizing, filtering stopwords (while preserving intensifiers), and lemmatizing.
    Returns a list of tokens.
    """
    text = text.lower()
    # Remove punctuation but keep numbers and whitespace
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in updated_stopwords]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def load_model_and_vectorizer(model_path="sentiment_best_model.pkl", vectorizer_path="vectorizer.pkl"):
    """
    Loads the pickled model and vectorizer (which includes the custom analyzer).
    The presence of custom_analyzer in this file ensures successful unpickling.
    """
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    """
    Uses the loaded vectorizer to transform the input text and the model to predict its sentiment.
    """
    # The vectorizer already integrates the custom_analyzer for preprocessing.
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)
    return prediction[0]

def main():
    model, vectorizer = load_model_and_vectorizer()
    print("Sentiment Prediction CLI - Type 'exit' to quit.")
    
    while True:
        user_input = input("Enter text: ")
        if user_input.strip().lower() == 'exit':
            print("Goodbye!")
            break
        sentiment = predict_sentiment(user_input, model, vectorizer)
        print("Predicted Sentiment:", sentiment)

if __name__ == "__main__":
    main()

