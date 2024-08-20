import random
import os

import cloudinary
import cloudinary.api
import cloudinary.uploader
from django.core.files.storage import FileSystemStorage
from django.http import FileResponse
from django.shortcuts import render, redirect
from PIL import Image
from utils.preprocess_image import get_model
from utils.preprocess_image import make_prediction
from utils.preprocess_image import preprocess_image
from utils.preprocess_image import validate_confidence_threshold
from utils.py_logger import get_logger
from data9.settings import BASE_DIR

logger = get_logger(__name__)


def index(request):
    logger.info("Started index")
    return render(request, "index.html")


def process_file(request):
    """Process uploaded files and make predictions."""
    logger.info("Started process file")
    show_upload_button = False
    results = []

    if request.method == "POST" and request.FILES.getlist("uploaded_files"):
        model_type = request.POST.get("model_type")
        logger.info(f"Selected model type: {model_type}")

        confidence_threshold = validate_confidence_threshold(
            request.POST.get("confidence_threshold", 0.70)
        )
        logger.info(f"Selected confidence threshold: {confidence_threshold}")

        model = get_model(model_type)
        if not model:
            return render(
                request,
                "result.html",
                {"results": results, "show_upload_button": show_upload_button},
            )

        uploaded_files = request.FILES.getlist("uploaded_files")

        for uploaded_file in uploaded_files:
            public_id = f"PhotoClassifier/{random.randint(1, 1000000)}"
            upload_result = cloudinary.uploader.upload(
                uploaded_file, public_id=public_id, overwrite=True
            )
            uploaded_image_url = upload_result["url"]

            img = Image.open(uploaded_file)
            img_array = preprocess_image(img)
            result_text, confidence = make_prediction(
                model, img_array, confidence_threshold
            )

            logger.info(f"Predicted class with confidence: {confidence * 100:.2f}%")
            results.append(
                {"uploaded_image_url": uploaded_image_url, "result_text": result_text}
            )

        show_upload_button = True

    return render(
        request,
        "result.html",
        {"results": results, "show_upload_button": show_upload_button},
    )

def model_LeNet(request):
    return render(request, 'model_LeNet.html')

def model_LeNet_tuned(request):
    return render(request, 'model_LeNet_tuned.html')

def model_VGG16(request):
    return render(request, 'model_VGG16.html')

def model_CNN(request):
    return render(request, 'model_CNN.html')


def download_model(request, model_type):
    model_path = os.path.join(BASE_DIR, f"{model_type}_model.keras")

    if os.path.exists(model_path):
        response = FileResponse(open(model_path, 'rb'), as_attachment=True, filename=f"{model_type}_model.keras")
        return response
    else:
        return render(request, 'error.html', {'message': 'Model file not found'})
