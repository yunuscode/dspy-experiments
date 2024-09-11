import os
from dotenv import load_dotenv
import dspy
from dspy.evaluate import Accuracy
from dspy.teleprompt import BootstrapFewShot

# Load environment variables from .env file
load_dotenv()

# Set up a language model
api_key = os.getenv("OPENAI_API_KEY")
lm = dspy.OpenAI(api_key=api_key, model="gpt-4")
dspy.configure(lm=lm)

# Define a simple classifier module
class SimpleClassifier(dspy.Signature):
    """Classify the sentiment or topic of a given text."""

    text = dspy.InputField()
    label = dspy.OutputField(desc="The classification label")

    def forward(self, text):
        prediction = dspy.Predict("text -> label")(text=text)
        return dspy.Prediction(label=prediction.label)

# Function to classify text
def classify_text(text, labels, dataset):
    # Create a basic compiler with Accuracy metric
    compiler = BootstrapFewShot(metric=Accuracy())

    # Create an instance of SimpleClassifier
    classifier_instance = SimpleClassifier()

    # Compile the model
    compiled_model = compiler.compile(classifier_instance, trainset=dataset)

    # Classify the input text
    classification = compiled_model(text=text)
    return classification.label

# Example usage
if __name__ == "__main__":
    # Sentiment classification example
    sentiment_labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    sentiment_dataset = [
        {"text": "This movie was fantastic! I loved every minute of it.", "label": "POSITIVE"},
        {"text": "The food was terrible and the service was even worse.", "label": "NEGATIVE"},
        {"text": "I'm not sure how I feel about this product.", "label": "NEUTRAL"}
    ]
    sentiment_text = "The concert last night was amazing!"
    sentiment_result = classify_text(sentiment_text, sentiment_labels, sentiment_dataset)
    print(f"Sentiment Classification: {sentiment_result}")

    # Topic classification example
    topic_labels = ["TECHNOLOGY", "SPORTS", "POLITICS", "ENTERTAINMENT"]
    topic_dataset = [
        {"text": "The new iPhone was unveiled yesterday.", "label": "TECHNOLOGY"},
        {"text": "The team won the championship after a thrilling match.", "label": "SPORTS"},
        {"text": "The president signed a new bill into law today.", "label": "POLITICS"},
        {"text": "The award-winning movie premiered at the film festival.", "label": "ENTERTAINMENT"}
    ]
    topic_text = "A new AI model has been developed that can generate realistic images."
    topic_result = classify_text(topic_text, topic_labels, topic_dataset)
    print(f"Topic Classification: {topic_result}")
