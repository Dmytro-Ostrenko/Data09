import random

import cloudinary
import cloudinary.api
import cloudinary.uploader
import numpy as np
from django.shortcuts import render
from PIL import Image
from tensorflow.keras.models import load_model
from utils.py_logger import get_logger
from utils.preprocess_image import preprocess_image

logger = get_logger(__name__)

model_test = load_model("my_cnn_model.keras")
model_lenet = load_model("lenet_best_model.keras")
model_lenettune_best = load_model("lenettune_best_model.keras")
model_cnn = load_model("cnn_model1_r1.keras")
model_vgg16 = load_model("model2_vgg16.keras")


def index(request):
    logger.info("Started index")
    return render(request, "index.html")


def process_file(request):
    logger.info("Started process file")
    """Функція яка обробляє файл та робить предікт на нього."""
    show_upload_button = False
    results = []
    if request.method == "POST" and request.FILES.getlist("uploaded_files"):
        model_type = request.POST.get("model_type")
        logger.info(f"Selected model type: {model_type}")
        try:
            confidence_threshold = float(request.POST.get("confidence_threshold", 0.70))
            logger.info(f"Selected confidence threshold: {confidence_threshold}")
            if confidence_threshold < 0 or confidence_threshold > 1:
                raise ValueError("Confidence threshold must be between 0 and 1.")
        except ValueError as e:
            logger.error(f"Invalid confidence threshold value: {e}")
            confidence_threshold = 0.70
        if model_type == "lenet":
            model = model_lenet
        elif model_type == "lenettune":
            model = model_lenettune_best
        elif model_type == "vgg16":
            model = model_vgg16
        elif model_type == "cnn":
            model = model_cnn
        else:
            model = None
            logger.error("Invalid model type selected")
        uploaded_files = request.FILES.getlist("uploaded_files")

        for uploaded_file in uploaded_files:
            public_id = f"PhotoClassifier/{random.randint(1, 1000000)}"
            upload_result = cloudinary.uploader.upload(
                uploaded_file, public_id=public_id, overwrite=True
            )
            uploaded_image_url = upload_result["url"]

            img = Image.open(uploaded_file)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_array = preprocess_image(img)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]

            class_labels = [
                "літак",
                "автомобіль",
                "птах",
                "кішка",
                "олень",
                "собака",
                "жаба",
                "кінь",
                "корабель",
                "вантажівка",
            ]

            if confidence >= confidence_threshold:
                logger.info(
                    f"Predicted class: {class_labels[predicted_class]} with confidence: {confidence * 100:.2f}%"
                )
                result_text = (f"На картинці зображено {class_labels[predicted_class]} із вірогідністю у "
                               f"{confidence * 100:.2f}%")
            else:
                logger.info(
                    f"Predicted class: {class_labels[predicted_class]} with confidence: {confidence * 100:.2f}%"
                )
                result_text = (f"Поточне зображення не підходить для класифікації. Впевненість моделі становить: "
                               f"{confidence * 100:.2f}%. Завантажте, будь ласка, інше зображення.")

            results.append(
                {"uploaded_image_url": uploaded_image_url, "result_text": result_text}
            )

        show_upload_button = True

    return render(
        request,
        "result.html",
        {"results": results, "show_upload_button": show_upload_button},
    )
