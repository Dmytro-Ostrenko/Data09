import os
from dotenv import load_dotenv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from utils.py_logger import get_logger
load_dotenv()
logger = get_logger(__name__)
model_test = load_model("my_cnn_model.keras")
model_lenet = load_model("lenet_best_model.keras")
model_lenettune_best = load_model("lenettune_best_model.keras")
model_cnn = load_model("cnn_model1_r1.keras")
model_vgg16 = load_model("model2_vgg16.keras")


def preprocess_image(img):
    img = img.resize((32, 32))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_model(model_type):
    """Get the model based on the selected model type."""
    if model_type == "lenet":
        return model_lenet
    elif model_type == "lenettune":
        return model_lenettune_best
    elif model_type == "vgg16":
        return model_vgg16
    elif model_type == "cnn":
        return model_cnn
    else:
        logger.error("Invalid model type selected")
        return None


def validate_confidence_threshold(threshold):
    """Validate the confidence threshold value."""
    try:
        threshold = float(threshold)
        if threshold < 0 or threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        return threshold
    except ValueError as e:
        logger.error(f"Invalid confidence threshold value: {e}")
        threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.70))
        return threshold


def make_prediction(model, img_array, confidence_threshold):
    """Make a prediction using the model and return the result."""
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    class_labels = os.getenv('MODEL_CLASSES', '').split(',')

    if confidence >= confidence_threshold:
        result_text = f"На картинці зображено {class_labels[predicted_class]} із вірогідністю у {confidence * 100:.2f}%"
    else:
        result_text = (
            f"Поточне зображення не підходить для класифікації. Впевненість моделі становить:"
            f" {confidence * 100:.2f}%. Завантажте, будь ласка, інше зображення."
        )

    return result_text, confidence
