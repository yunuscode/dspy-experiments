import os
from dotenv import load_dotenv
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.primitives import Example

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
        self.classify = dspy.ChainOfThoughtWithHint("text, hint -> label")

    def forward(self, text, hint):
        return self.classify(text=text, hint=hint)

# Define a simple accuracy metric
def accuracy_metric(example, pred, trace=None):
    return example.label == pred.label

# Function to classify text
def classify_text(text, labels, dataset, hint):
    # Create a basic compiler with the accuracy metric
    compiler = BootstrapFewShot(metric=accuracy_metric)

    # Create an instance of SimpleClassifier with the given labels
    classifier_instance = SimpleClassifier(labels)

    # Convert dataset to proper Example objects with inputs specified
    proper_dataset = [
        Example(text=item['text'], label=item['label'], hint=hint).with_inputs('text', 'hint')
        for item in dataset
    ]

    # Compile the model
    compiled_model = compiler.compile(classifier_instance, trainset=proper_dataset)

    # Classify the input text
    classification = compiled_model(text=text, hint=hint)
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
    sample_text = "A new Iphone was unveiled yesterday."
    hint = "Only output the label. If there is no clear label, output 'OTHER'. Don't output anything else."
    topic = classify_text(sample_text, topic_labels, topic_dataset, hint)
    print(f"Topic Classification: {topic}")
