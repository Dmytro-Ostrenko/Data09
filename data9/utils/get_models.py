import os
from os.path import exists

import gdown
import tensorflow as tf


def get_vgg16_model():
    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = "models/vgg16.keras"
    if not exists(model_path):
        url = "https://drive.google.com/file/d/19fUTQSt9hYdln8qvGl4k1uIWPuWzYkYz/view?usp=drive_link"
        output = model_path
        gdown.download(url, output, quiet=False, fuzzy=True)
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        model = tf.keras.models.load_model(model_path)
        return model


def get_mobnet_model():
    if not os.path.exists("models"):
        os.makedirs("models")
    model_path = "models/mobnet.keras"

    if not exists(model_path):
        url = "https://drive.google.com/file/d/1CO8Yo3nmJpreYb96aGp-fOMvqgDCyq77/view?usp=drive_link"
        output = model_path
        gdown.download(url, output, quiet=False, fuzzy=True)
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        model = tf.keras.models.load_model(model_path)
        return model

def get_lenet_model():
    if not os.path.exists("models"):
        os.makedirs("models")
    model_path = "models/lenet.keras"

    if not exists(model_path):
        url = "https://drive.google.com/file/d/1KYD39NJh-yRnfhuorIUCR2WsAR3GS6PD/view?usp=drive_link"
        output = model_path
        gdown.download(url, output, quiet=False, fuzzy=True)
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        model = tf.keras.models.load_model(model_path)
        return model


def get_alexnet_model():
    if not os.path.exists("models"):
        os.makedirs("models")
    model_path = "models/alexnet.keras"
    if not exists(model_path):
        url = "https://drive.google.com/file/d/1DBTDC2c8k4lgwjyCgdBVObdO9r9CJ_Th/view?usp=drive_link"
        output = model_path
        gdown.download(url, output, quiet=False, fuzzy=True)
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        model = tf.keras.models.load_model(model_path)
        return model
