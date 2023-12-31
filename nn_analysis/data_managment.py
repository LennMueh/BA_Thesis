import tensorflow as tf
import numpy as np
def get_data():
    tf.random.set_seed(42)

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]

    return train_images, train_labels, test_images, test_labels

def get_model(model_name):
    model_path = "/mnt/c/Users/lenna/PycharmProjects/BA_Thesis/models/" + model_name
    model = tf.keras.saving.load_model(model_path)
    return model