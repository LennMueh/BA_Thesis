import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

tf.random.set_seed(42)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0 # 60000 images of 28x28 pixels with values in the range [0, 255]
test_images = test_images / 255.0  # 10000 images of 28x28 pixels with values in the range [0, 255]

half_train_images, _, half_train_labels, _ = train_test_split(train_images, train_labels, test_size=0.5, random_state=42)
quarter_train_images, _, quarter_train_labels, _ = train_test_split(train_images, train_labels, test_size=0.75, random_state=42)

dnn_model2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
dnn_model3 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
dnn_model4 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
dnn_model5 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
dnn_model2_half = keras.models.clone_model(dnn_model2)
dnn_model3_half = keras.models.clone_model(dnn_model3)
dnn_model4_half = keras.models.clone_model(dnn_model4)
dnn_model5_half = keras.models.clone_model(dnn_model5)
dnn_model2_quarter = keras.models.clone_model(dnn_model2)
dnn_model3_quarter = keras.models.clone_model(dnn_model3)
dnn_model4_quarter = keras.models.clone_model(dnn_model4)
dnn_model5_quarter = keras.models.clone_model(dnn_model5)
dnn_model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model4.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model5.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model2_half.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model3_half.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model4_half.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model5_half.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model2_quarter.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model3_quarter.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model4_quarter.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model5_quarter.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("dnn_model")
dnn_model2.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
print("dnn_model3")
dnn_model3.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
print("dnn_model4")
dnn_model4.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
print("dnn_model5")
dnn_model5.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
print("dnn_model2_half")
dnn_model2_half.fit(half_train_images, half_train_labels, epochs=5, validation_data=(test_images, test_labels))
print("dnn_model3_half")
dnn_model3_half.fit(half_train_images, half_train_labels, epochs=5, validation_data=(test_images, test_labels))
print("dnn_model4_half")
dnn_model4_half.fit(half_train_images, half_train_labels, epochs=5, validation_data=(test_images, test_labels))
print("dnn_model5_half")
dnn_model5_half.fit(half_train_images, half_train_labels, epochs=5, validation_data=(test_images, test_labels))
print("dnn_model2_quarter")
dnn_model2_quarter.fit(quarter_train_images, quarter_train_labels, epochs=5, validation_data=(test_images, test_labels))
print("dnn_model3_quarter")
dnn_model3_quarter.fit(quarter_train_images, quarter_train_labels, epochs=5, validation_data=(test_images, test_labels))
print("dnn_model4_quarter")
dnn_model4_quarter.fit(quarter_train_images, quarter_train_labels, epochs=5, validation_data=(test_images, test_labels))
print("dnn_model5_quarter")
dnn_model5_quarter.fit(quarter_train_images, quarter_train_labels, epochs=5, validation_data=(test_images, test_labels))

cnn_model1 = keras.Sequential([
    keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

cnn_model2 = keras.Sequential([
    keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(8, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
cnn_model1_half = keras.models.clone_model(cnn_model1)
cnn_model2_half = keras.models.clone_model(cnn_model2)
cnn_model1_quarter = keras.models.clone_model(cnn_model1)
cnn_model2_quarter = keras.models.clone_model(cnn_model2)
cnn_model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model1_half.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model2_half.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model1_quarter.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model2_quarter.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("cnn_model1")
cnn_model1.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
print("cnn_model2")
cnn_model2.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
print("cnn_model1_half")
cnn_model1_half.fit(half_train_images, half_train_labels, epochs=5, validation_data=(test_images, test_labels))
print("cnn_model2_half")
cnn_model2_half.fit(half_train_images, half_train_labels, epochs=5, validation_data=(test_images, test_labels))
print("cnn_model1_quarter")
cnn_model1_quarter.fit(quarter_train_images, quarter_train_labels, epochs=5, validation_data=(test_images, test_labels))
print("cnn_model2_quarter")
cnn_model2_quarter.fit(quarter_train_images, quarter_train_labels, epochs=5, validation_data=(test_images, test_labels))