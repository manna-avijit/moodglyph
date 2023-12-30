from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Define a dictionary to map sentiment labels to emojis
EMOJI_DICT = {
    'positive': '😃',
    'negative': '😢',
    'neutral': '😐',
}

def load_sentiment_model(model_name='nlptown/bert-base-multilingual-uncased-sentiment'):
    """
    Load the pre-trained sentiment analysis model and tokenizer from Hugging Face Transformers.

    Args:
        model_name (str): The name or path of the pre-trained model.

    Returns:
        model: The loaded sentiment analysis model.
        tokenizer: The loaded tokenizer.
    """
    try:
        # Load the sentiment analysis model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        return None, None

def predict_sentiment(model, tokenizer, text):
    """
    Predict sentiment for the given text using the provided sentiment analysis model and tokenizer.

    Args:
        model (AutoModelForSequenceClassification): The sentiment analysis model.
        tokenizer (AutoTokenizer): The tokenizer.
        text (str): The input text for sentiment analysis.

    Returns:
        str: Predicted sentiment label ('positive', 'negative', 'neutral', or 'error').
        float: Confidence score for the predicted sentiment.
    """
    try:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt")

        # Get the predicted sentiment
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits).item()

        # Convert predicted class to sentiment label
        sentiment_labels = ['negative', 'neutral', 'positive']
        predicted_sentiment = sentiment_labels[predicted_class]

        # Get the confidence score
        confidence_score = torch.nn.functional.softmax(outputs.logits, dim=1)[0][predicted_class].item()

        return predicted_sentiment, confidence_score
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return 'error', 0.0

def display_emoji(sentiment, confidence_score):
    """
    Display the sentiment label, corresponding emoji, and confidence score.

    Args:
        sentiment (str): Predicted sentiment label ('positive', 'negative', 'neutral', or 'error').
        confidence_score (float): Confidence score for the predicted sentiment.
    """
    emoji = EMOJI_DICT.get(sentiment, '🤷‍♂️')
    print(f'Sentiment: {sentiment.capitalize()} {emoji} (Confidence: {confidence_score:.2%})')

def main():
    """
    Main function to execute the Sentiment Analysis and Emoji Display Program.
    """
    print("Welcome to the Sentiment Analysis and Emoji Display Program!")

    # Load the sentiment analysis model and tokenizer
    model, tokenizer = load_sentiment_model()

    if model is None or tokenizer is None:
        print("Exiting due to model loading error.")
        return

    while True:
        user_input = input("Enter text or type 'exit' to end: ")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Predict sentiment and display results
        sentiment, confidence_score = predict_sentiment(model, tokenizer, user_input)

        if sentiment == 'error':
            print("Unable to determine sentiment. Please try again.")
        else:
            display_emoji(sentiment, confidence_score)

if __name__ == "__main__":
    main()
    
