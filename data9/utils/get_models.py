from os.path import exists

import gdown
import tensorflow as tf


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


def get_snn_model():
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
        model_lenettune = tf.keras.models.load_model("models/lenettune_best_model.keras")
        return model_lenettune
