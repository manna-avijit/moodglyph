from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

# Define a dictionary to map sentiment labels to emojis
EMOJI_DICT = {
    'positive': '😃',
    'negative': '😢',
    'neutral': '😐',
}

def predict_sentiment(text, model_name='nlptown/bert-base-multilingual-uncased-sentiment'):
    try:
        # Load the sentiment analysis model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt")

        # Get the predicted sentiment
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()

        # Convert predicted class to sentiment label
        sentiment_labels = ['negative', 'neutral', 'positive']
        predicted_sentiment = sentiment_labels[predicted_class]

        # Get the confidence score
        confidence_score = torch.nn.functional.softmax(logits, dim=1)[0][predicted_class].item()

        return predicted_sentiment, confidence_score

    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return 'error', 0.0

def display_emoji(sentiment, confidence_score):
    emoji = EMOJI_DICT.get(sentiment, '🤷‍♂️')
    print(f'Sentiment: {sentiment.capitalize()} {emoji} (Confidence: {confidence_score:.2%})')

def main():
    print("Welcome to the Sentiment Analysis and Emoji Display Program!")

    while True:
        user_input = input("Enter text or type 'exit' to end: ")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        sentiment, confidence_score = predict_sentiment(user_input)

        if sentiment == 'error':
            print("Unable to determine sentiment. Please try again.")
        else:
            display_emoji(sentiment, confidence_score)

if __name__ == "__main__":
    main()
