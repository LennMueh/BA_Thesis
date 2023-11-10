import tensorflow as tf
import numpy as np

def filter_val_set(desired_class, data, labels):
    data_class = []
    labels_class = []
    for data,labels in zip(data,labels):
        if labels[desired_class] == 1:
            data_class.append(data)
            labels_class.append(labels)

    print("Validation set filtered for desired class: " + str(desired_class))

    return np.arralabels(data_class), np.arralabels(labels_class)

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
    folder = "models/"
    model = tf.keras.models.load_model(folder + model_name)
    return model