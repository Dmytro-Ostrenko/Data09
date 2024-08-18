import os
import random

import cloudinary
import cloudinary.uploader
import cloudinary.api
import numpy as np
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from PIL import Image
from tensorflow.keras.models import load_model
from utils.py_logger import get_logger

logger = get_logger(__name__)

model = load_model('my_cnn_model.keras')
# upload_dir = os.path.join('data9', 'image_analysis', '../media')


def index(request):
    logger.info("Started index")
    return render(request, 'index.html')


def process_file(request):
    logger.info("Started process file")
    """Функція яка обробляє файл та робить предікт на нього."""
    show_upload_button = False
    results = []
    # result_text = None
    # uploaded_image_url = None

    if request.method == 'POST' and request.FILES.getlist('uploaded_files'):
        uploaded_files = request.FILES.getlist('uploaded_files')
        # fs = FileSystemStorage()

        for uploaded_file in uploaded_files:
            # filename = fs.save(uploaded_file.name, uploaded_file)
            # uploaded_image_url = fs.url(filename)
            public_id = f'PhotoClassifier/{random.randint(1, 1000000)}'
            upload_result = cloudinary.uploader.upload(uploaded_file, public_id=public_id, overwrite=True)
            uploaded_image_url = upload_result['url']

            img = Image.open(uploaded_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((32, 32))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]

            confidence_threshold = 0.70
            class_labels = ['літак', 'автомобіль', 'птах', 'кішка', 'олень', 'собака', 'жаба', 'кінь', 'корабель',
                            'вантажівка']

            if confidence >= confidence_threshold:
                logger.info(f"Predicted class: {class_labels[predicted_class]} with confidence: {confidence * 100:.2f}%")
                result_text = f'На картинці зображено {class_labels[predicted_class]} із вірогідністю у {confidence * 100:.2f}%'
            else:
                logger.info(f"Predicted class: {class_labels[predicted_class]} with confidence: {confidence * 100:.2f}%")
                result_text = f"Поточне зображення не підходить для класифікації. Впевненість моделі становить: {confidence * 100:.2f}%. Завантажте, будь ласка, інше зображення."

            # results.append({'result_text': result_text, 'uploaded_image_url': uploaded_image_url})
            results.append({'uploaded_image_url': uploaded_image_url, 'result_text': result_text})

        show_upload_button = True

    # return render(request, 'result.html', {
    #     'result_text': result_text,
    #     'uploaded_image_url': uploaded_image_url,
    #     'show_upload_button': show_upload_button
    # })

    return render(request, 'result.html', {
        'results': results,
        'show_upload_button': show_upload_button
    })
