import os

import numpy as np
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.image import img_to_array
from utils.get_models import get_lenet_model
from utils.get_models import get_lenet_tune_model
from utils.get_models import get_snn_model
from utils.get_models import get_vgg16_model
from utils.py_logger import get_logger

load_dotenv()
logger = get_logger(__name__)
model_lenet = get_lenet_model()
model_lenettune_best = get_lenet_tune_model()
model_cnn = get_snn_model()
model_vgg16 = get_vgg16_model()



def preprocess_image(img):
    """Перевірка на ргб та створення масиву зображення."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
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
    # elif model_type == "vgg16":
    #     return model_vgg16
    elif model_type == "cnn":
        return model_cnn
    else:
        logger.error("Invalid model type selected")
        return model_lenet


def validate_confidence_threshold(threshold):
    """Validate the confidence threshold value."""
    try:
        threshold = float(threshold)
        if threshold < 0 or threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        return threshold
    except ValueError as e:
        logger.error(f"Invalid confidence threshold value: {e}")
        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.70))
        return threshold


def make_prediction(model, img_array, confidence_threshold):
    """Make a prediction using the model and return the result."""
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    class_labels = os.getenv("MODEL_CLASSES", "").split(",")

    if confidence >= confidence_threshold:
        result_text = f"На картинці зображено {class_labels[predicted_class]} із вірогідністю у {confidence * 100:.2f}%"
    else:
        result_text = (
            f"Поточне зображення не підходить для класифікації. Впевненість моделі становить:"
            f" {confidence * 100:.2f}%. Завантажте, будь ласка, інше зображення."
        )

    return result_text, confidence
