import os
from dotenv import load_dotenv
import dspy
from dspy.teleprompt import BootstrapFewShot

# Load environment variables from .env file
load_dotenv()

# Set up a language model
api_key = os.getenv("OPENAI_API_KEY")
lm = dspy.OpenAI(api_key=api_key, model="gpt-4")  # Changed to GPT-4
dspy.configure(lm=lm)

# Define a simple classifier module
class SimpleClassifier(dspy.Module):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        self.classify = dspy.Predict("text -> label")

    def forward(self, text):
        return self.classify(text=text)

# Define a simple accuracy metric
def accuracy_metric(example, pred):
    print(f"Example: {example}, Prediction: {pred}")
    return example['label'] == pred.label

# Function to classify text
def classify_text(text, labels, dataset):
    # Create a basic compiler
    compiler = BootstrapFewShot()

    # Create an instance of SimpleClassifier with the given labels
    classifier_instance = SimpleClassifier(labels)

    # Compile the model
    compiled_model = compiler.compile(classifier_instance, trainset=dataset)

    # Classify the input text
    classification = compiled_model(text=text)
    return classification.label

# Example usage
if __name__ == "__main__":
    # Example: Sentiment classification


    # Example: Topic classification
    topic_labels = ["TECHNOLOGY", "SPORTS", "POLITICS", "ENTERTAINMENT"]

    topic_dataset = [
        {"text": "The new iPhone was unveiled yesterday.", "label": "TECHNOLOGY"},
        {"text": "The team won the championship after a thrilling match.", "label": "SPORTS"},
        {"text": "The president signed a new bill into law today.", "label": "POLITICS"},
        {"text": "The award-winning movie premiered at the film festival.", "label": "ENTERTAINMENT"}
    ]

    # Test topic classifier
    sample_text = "Ronaldo gifted a new Iphone for Trumps birthday."
    topic = classify_text(sample_text, topic_labels, topic_dataset)
    print(f"Topic Classification: {topic}")
