import tensorflow as tf
from tensorflow import keras
from plot_keras_history import plot_history

if __name__ == '__main__':
    tf.random.set_seed(42)

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    dnn_model1_gelu = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='gelu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    dnn_model1_gelu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    dnn_model1_gelu.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
    keras.saving.save_model(dnn_model1_gelu, "../models/dnn_hid1_gelu_1epoch")

    dnn_model1_gelu.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(dnn_model1_gelu, "../models/dnn_hid1_gelu_6epoch")

    plot_history(dnn_model1_gelu.history, path="../models/dnn_hid1_gelu.png")

    dnn_model2_gelu = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='gelu'),
        keras.layers.Dense(64, activation='gelu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    dnn_model2_gelu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    dnn_model2_gelu.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
    keras.saving.save_model(dnn_model2_gelu, "../models/dnn_hid2_gelu_1epoch")

    dnn_model2_gelu.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(dnn_model2_gelu, "../models/dnn_hid2_gelu_6epoch")

    plot_history(dnn_model2_gelu.history, path="../models/dnn_hid2_gelu.png")

    cnn_model1_gelu = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='gelu', input_shape=(28,28, 1)),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='gelu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    cnn_model1_gelu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    cnn_model1_gelu.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
    keras.saving.save_model(cnn_model1_gelu, "../models/cnn_conv1_gelu_1epoch")

    cnn_model1_gelu.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(cnn_model1_gelu, "../models/cnn_conv1_gelu_6epoch")
    plot_history(dnn_model2_gelu.history, path="../models/cnn_hid1_gelu.png")
    
    cnn_model2_gelu = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='gelu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='gelu'),  # Additional convolutional layer
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='gelu'),  # Additional fully connected layer
        keras.layers.Dense(10, activation='softmax')
    ])

    cnn_model2_gelu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    cnn_model2_gelu.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
    keras.saving.save_model(cnn_model2_gelu, "../models/cnn_conv2_gelu_1epoch")

    cnn_model2_gelu.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(cnn_model2_gelu, "../models/cnn_conv2_gelu_6epoch")
    plot_history(cnn_model2_gelu.history, path="../models/cnn_hid2_gelu.png")

    dnn_model1_relu = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    dnn_model1_relu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    dnn_model1_relu.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
    keras.saving.save_model(dnn_model1_relu, "../models/dnn_hid1_relu_1epoch")

    dnn_model1_relu.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(dnn_model1_relu, "../models/dnn_hid1_relu_6epoch")

    plot_history(dnn_model1_relu.history, path="../models/dnn_hid1_relu.png")

    dnn_model2_relu = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    dnn_model2_relu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    dnn_model2_relu.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
    keras.saving.save_model(dnn_model2_relu, "../models/dnn_hid2_relu_1epoch")

    dnn_model2_relu.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(dnn_model2_relu, "../models/dnn_hid2_relu_6epoch")

    plot_history(dnn_model2_relu.history, path="../models/dnn_hid2_relu.png")

    cnn_model1_relu = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    cnn_model1_relu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    cnn_model1_relu.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
    keras.saving.save_model(cnn_model1_relu, "../models/cnn_conv1_relu_1epoch")

    cnn_model1_relu.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(cnn_model1_relu, "../models/cnn_conv1_relu_6epoch")
    plot_history(dnn_model2_relu.history, path="../models/cnn_hid1_relu.png")

    cnn_model2_relu = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Additional convolutional layer
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),  # Additional fully connected layer
        keras.layers.Dense(10, activation='softmax')
    ])

    cnn_model2_relu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    cnn_model2_relu.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
    keras.saving.save_model(cnn_model2_relu, "../models/cnn_conv2_relu_1epoch")

    cnn_model2_relu.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(cnn_model2_relu, "../models/cnn_conv2_relu_6epoch")
    plot_history(cnn_model2_relu.history, path="../models/cnn_hid2_relu.png")
