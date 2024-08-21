import os
from functools import lru_cache
from os.path import exists
from pathlib import Path

import gdown
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_ROOT = os.path.join(BASE_DIR, "models")


def get_vgg16_model():
    if not exists("models/vgg16tune_best_model.keras"):
        url = "https://drive.google.com/file/d/1Cf9iNG8XqXjVJhszMkm47njDZL35ETTh/view"
        output = "vgg16tune_best_model.keras"
        gdown.download(url, output, quiet=False, fuzzy=True)
        model_filename = "vgg16tune_best_model.keras"
        model_vgg16 = tf.keras.models.load_model(model_filename)
        return model_vgg16
    else:
        model_vgg16 = tf.keras.models.load_model("models/vgg16tune_best_model.keras")
        return model_vgg16


def get_cnn_model():
    if not exists("models/cnn_model1_r1.keras"):
        url = "https://drive.google.com/file/d/1Cf9iNG8XqXjVJhszMkm47njDZL35ETTh/view?usp=drive_link"
        output = "cnn_model1_r1.keras"
        gdown.download(url, output, quiet=False, fuzzy=True)
        model_filename = "cnn_model1_r1.keras"
        model_cnn = tf.keras.models.load_model(model_filename)
        return model_cnn
    else:
        model_cnn = tf.keras.models.load_model("models/cnn_model1_r1.keras")
        return model_cnn


def get_lenet_model():
    if not exists("models/lenet_best_model.keras"):
        url = "https://drive.google.com/file/d/1Cf9iNG8XqXjVJhszMkm47njDZL35ETTh/view?usp=drive_link"
        output = "lenet_best_model.keras"
        gdown.download(url, output, quiet=False, fuzzy=True)
        model_filename = "lenet_best_model.keras"
        model_lenet = tf.keras.models.load_model(model_filename)
        return model_lenet
    else:
        model_lenet = tf.keras.models.load_model("models/lenet_best_model.keras")
        return model_lenet


def get_lenet_tune_model():
    if not exists("models/lenettune_best_model.keras"):
        url = "https://drive.google.com/file/d/1Cf9iNG8XqXjVJhszMkm47njDZL35ETTh/view?usp=drive_link"
        output = "lenettune_best_model.keras"
        gdown.download(url, output, quiet=False, fuzzy=True)
        model_filename = "lenettune_best_model.keras"
        model_lenettune = tf.keras.models.load_model(model_filename)
        return model_lenettune
    else:
        model_lenettune = tf.keras.models.load_model(
            "models/lenettune_best_model.keras"
        )
        return model_lenettune


def get_mobnet_model():
    if not exists("models/mobilenet_best_model.keras"):
        url = "https://drive.google.com/file/d/1Cf9iNG8XqXjVJhszMkm47njDZL35ETTh/view?usp=drive_link"
        output = "mobilenet_best_model.keras"
        gdown.download(url, output, quiet=False, fuzzy=True)
        model_filename = "mobnet_best_model.keras"
        model_mobilenet = tf.keras.models.load_model(model_filename)
        return model_mobilenet
    else:
        model_mobilenet = tf.keras.models.load_model(
            "models/mobilenet_best_model.keras"
        )
        return model_mobilenet


@lru_cache(maxsize=None)
def get_model_by_type(model_type: str):
    if model_type == "model_VGG16":
        return get_vgg16_model()
    elif model_type == "model_LeNet":
        return get_lenet_model()
    elif model_type == "model_CNN":
        return get_cnn_model()
    elif model_type == "model_LenetTune":
        return get_lenet_tune_model()
    elif model_type == "model_MobileNet":
        return get_mobnet_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
