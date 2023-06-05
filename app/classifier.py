import tensorflow_hub as hub
import tensorflow as tf
from tensorflow_text import SentencepieceTokenizer
import joblib

# Load the pre-trained model
model = joblib.load("../NLP_AF_A.joblib")

# Create an instance of the LabelEncoder and fit it on training labels

# Load the Universal Sentence Encoded

def classify_prompt(prompt):
    # Perform the classification using the loaded model
    prediction = list(model.predict_proba([prompt])[0])
    classification = prediction.index(max(prediction))
    return_class = model.classes_[classification]
    return return_class