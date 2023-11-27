import tensorflow as tf
from tensorflow import keras
from plot_keras_history import plot_history
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    tf.random.set_seed(42)

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0 # 60000 images of 28x28 pixels with values in the range [0, 255]
    test_images = test_images / 255.0  # 10000 images of 28x28 pixels with values in the range [0, 255]

    half_train_images, _, half_train_labels, _ = train_test_split(train_images, train_labels, test_size=0.5, random_state=42)
    quarter_train_images, _, quarter_train_labels, _ = train_test_split(train_images, train_labels, test_size=0.75, random_state=42)



    dnn_model1_relu = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    dnn_model1_relu_half = keras.models.clone_model(dnn_model1_relu)
    dnn_model1_relu_quarter = keras.models.clone_model(dnn_model1_relu)

    dnn_model1_relu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    dnn_model1_relu_half.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    dnn_model1_relu_quarter.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    dnn_model1_relu.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
    dnn_model1_relu_half.fit(half_train_images, half_train_labels , epochs=1, validation_data=(test_images, test_labels))
    dnn_model1_relu_quarter.fit(quarter_train_images, quarter_train_labels, epochs=1, validation_data=(test_images, test_labels))
    keras.saving.save_model(dnn_model1_relu, "../models/dnn_hid1_relu_1epoch")
    keras.saving.save_model(dnn_model1_relu_half, "../models/dnn_hid1_relu_1epoch_half")
    keras.saving.save_model(dnn_model1_relu_quarter, "../models/dnn_hid1_relu_1epoch_quarter")

    dnn_model1_relu.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    dnn_model1_relu_half.fit(half_train_images, half_train_labels, epochs=5, validation_data=(test_images, test_labels))
    dnn_model1_relu_quarter.fit(quarter_train_images, quarter_train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(dnn_model1_relu, "../models/dnn_hid1_relu_6epoch")
    keras.saving.save_model(dnn_model1_relu_half, "../models/dnn_hid1_relu_6epoch_half")
    keras.saving.save_model(dnn_model1_relu_quarter, "../models/dnn_hid1_relu_6epoch_quarter")

    plot_history(dnn_model1_relu.history, path="../models/dnn_hid1_relu.png")
    plot_history(dnn_model1_relu_half.history, path="../models/dnn_hid1_relu_half.png")
    plot_history(dnn_model1_relu_quarter.history, path="../models/dnn_hid1_relu_quarter.png")

    dnn_model2_relu = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    dnn_model2_relu_half = keras.models.clone_model(dnn_model2_relu)
    dnn_model2_relu_quarter = keras.models.clone_model(dnn_model2_relu)

    dnn_model2_relu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    dnn_model2_relu_half.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    dnn_model2_relu_quarter.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    dnn_model2_relu.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
    dnn_model2_relu_half.fit(half_train_images, half_train_labels, epochs=1, validation_data=(test_images, test_labels))
    dnn_model2_relu_quarter.fit(quarter_train_images, quarter_train_labels, epochs=1, validation_data=(test_images, test_labels))
    keras.saving.save_model(dnn_model2_relu, "../models/dnn_hid2_relu_1epoch")
    keras.saving.save_model(dnn_model2_relu_half, "../models/dnn_hid2_relu_1epoch_half")
    keras.saving.save_model(dnn_model2_relu_quarter, "../models/dnn_hid2_relu_1epoch_quarter")

    dnn_model2_relu.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    dnn_model2_relu_half.fit(half_train_images, half_train_labels, epochs=5, validation_data=(test_images, test_labels))
    dnn_model2_relu_quarter.fit(quarter_train_images, quarter_train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(dnn_model2_relu, "../models/dnn_hid2_relu_6epoch")
    keras.saving.save_model(dnn_model2_relu_half, "../models/dnn_hid2_relu_6epoch_half")
    keras.saving.save_model(dnn_model2_relu_quarter, "../models/dnn_hid2_relu_6epoch_quarter")

    plot_history(dnn_model2_relu.history, path="../models/dnn_hid2_relu.png")
    plot_history(dnn_model2_relu_half.history, path="../models/dnn_hid2_relu_half.png")
    plot_history(dnn_model2_relu_quarter.history, path="../models/dnn_hid2_relu_quarter.png")

    cnn_model1_relu = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    cnn_model1_relu_half = keras.models.clone_model(cnn_model1_relu)
    cnn_model1_relu_quarter = keras.models.clone_model(cnn_model1_relu)

    cnn_model1_relu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model1_relu_half.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model1_relu_quarter.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    cnn_model1_relu.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
    cnn_model1_relu_half.fit(half_train_images, half_train_labels, epochs=1, validation_data=(test_images, test_labels))
    cnn_model1_relu_quarter.fit(quarter_train_images, quarter_train_labels, epochs=1, validation_data=(test_images, test_labels))
    keras.saving.save_model(cnn_model1_relu, "../models/cnn_conv1_relu_1epoch")
    keras.saving.save_model(cnn_model1_relu_half, "../models/cnn_conv1_relu_1epoch_half")
    keras.saving.save_model(cnn_model1_relu_quarter, "../models/cnn_conv1_relu_1epoch_quarter")

    cnn_model1_relu.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    cnn_model1_relu_half.fit(half_train_images, half_train_labels, epochs=5, validation_data=(test_images, test_labels))
    cnn_model1_relu_quarter.fit(quarter_train_images, quarter_train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(cnn_model1_relu, "../models/cnn_conv1_relu_6epoch")
    keras.saving.save_model(cnn_model1_relu_half, "../models/cnn_conv1_relu_6epoch_half")
    keras.saving.save_model(cnn_model1_relu_quarter, "../models/cnn_conv1_relu_6epoch_quarter")
    plot_history(dnn_model2_relu.history, path="../models/cnn_hid1_relu.png")
    plot_history(dnn_model2_relu_half.history, path="../models/cnn_hid1_relu_half.png")
    plot_history(dnn_model2_relu_quarter.history, path="../models/cnn_hid1_relu_quarter.png")

    cnn_model2_relu = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Additional convolutional layer
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),  # Additional fully connected layer
        keras.layers.Dense(10, activation='softmax')
    ])
    cnn_model2_relu_half = keras.models.clone_model(cnn_model2_relu)
    cnn_model2_relu_quarter = keras.models.clone_model(cnn_model2_relu)

    cnn_model2_relu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model2_relu_half.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model2_relu_quarter.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    cnn_model2_relu.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
    cnn_model2_relu_half.fit(half_train_images, half_train_labels, epochs=1, validation_data=(test_images, test_labels))
    cnn_model2_relu_quarter.fit(quarter_train_images, quarter_train_labels, epochs=1, validation_data=(test_images, test_labels))
    keras.saving.save_model(cnn_model2_relu, "../models/cnn_conv2_relu_1epoch")
    keras.saving.save_model(cnn_model2_relu_half, "../models/cnn_conv2_relu_1epoch_half")
    keras.saving.save_model(cnn_model2_relu_quarter, "../models/cnn_conv2_relu_1epoch_quarter")

    cnn_model2_relu.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    cnn_model2_relu_half.fit(half_train_images, half_train_labels, epochs=5, validation_data=(test_images, test_labels))
    cnn_model2_relu_quarter.fit(quarter_train_images, quarter_train_labels, epochs=5, validation_data=(test_images, test_labels))
    keras.saving.save_model(cnn_model2_relu, "../models/cnn_conv2_relu_6epoch")
    plot_history(cnn_model2_relu.history, path="../models/cnn_hid2_relu.png")
