import tensorflow as tf
from tensorflow import keras
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tf.random.set_seed(42)

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    dnn1_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='gelu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    dnn1_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("DNN, 1 Hidden Layer, 5 Epochs")
    dnn1_model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(dnn1_model, "../models/dnn_hid1_gelu_5epoch")

    print("DNN, 1 Hidden Layer, 10 Epochs")
    dnn1_model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(dnn1_model, "../models/dnn_hid1_gelu_10epoch")

    dnn2_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='gelu'),
        keras.layers.Dense(64, activation='gelu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    dnn2_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("DNN, 2 Hidden Layers, 5 Epochs")
    dnn2_model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(dnn2_model, "../models/dnn_hid2_gelu_5epoch")

    print("DNN, 2 Hidden Layers, 10 Epochs")
    dnn2_model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(dnn2_model, "../models/dnn_hid2_gelu_10epoch")

    cnn1_model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='gelu', input_shape=(28,28, 1)),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='gelu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    cnn1_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("CNN, 1 Convolution Layer, 5 Epochs")
    cnn1_model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(cnn1_model, "../models/cnn_conv1_gelu_5epoch")

    print("CNN, 1 Convolution Layer, 10 Epochs")
    cnn1_model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(cnn1_model, "../models/cnn_conv1_gelu_10epoch")
    
    cnn2_model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='gelu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='gelu'),  # Additional convolutional layer
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='gelu'),  # Additional fully connected layer
        keras.layers.Dense(10, activation='softmax')
    ])

    cnn2_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("CNN, 2 Convolution Layer, 5 Epochs")
    cnn1_model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(cnn1_model, "../models/cnn_conv2_gelu_5epoch")

    print("CNN, 2 Convolution Layer, 10 Epochs")
    cnn1_model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(cnn1_model, "../models/cnn_conv2_gelu_10epoch")
