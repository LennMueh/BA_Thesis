import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas

tf.random.set_seed(42)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0 # 60000 images of 28x28 pixels with values in the range [0, 255]
test_images = test_images / 255.0  # 10000 images of 28x28 pixels with values in the range [0, 255]

half_train_images, _, half_train_labels, _ = train_test_split(train_images, train_labels, test_size=0.5, random_state=42)
quarter_train_images, _, quarter_train_labels, _ = train_test_split(train_images, train_labels, test_size=0.75, random_state=42)

dnn_model1 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
dnn_model1_half = keras.models.clone_model(dnn_model1)
dnn_model1_quarter = keras.models.clone_model(dnn_model1)

dnn_model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model1_half.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model1_quarter.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history1_full = dnn_model1.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))
history1_full_df = pandas.DataFrame(history1_full.history)
history1_full_df['epoch'] = history1_full_df.index + 1
history1_full_df['model'] = 'dnn1_full'

history1_half = dnn_model1_half.fit(half_train_images, half_train_labels, epochs=20, validation_data=(test_images, test_labels))
history1_half_df = pandas.DataFrame(history1_half.history)
history1_half_df['epoch'] = history1_half_df.index + 1
history1_half_df['model'] = 'dnn1_half'

history1_quarter = dnn_model1_quarter.fit(quarter_train_images, quarter_train_labels, epochs=20, validation_data=(test_images, test_labels))
history1_quarter_df = pandas.DataFrame(history1_quarter.history)
history1_quarter_df['epoch'] = history1_quarter_df.index + 1
history1_quarter_df['model'] = 'dnn1_quarter'

history1_df = pandas.concat([history1_full_df, history1_half_df, history1_quarter_df])

dnn_model2 = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
])
dnn_model2_half = keras.models.clone_model(dnn_model2)
dnn_model2_quarter = keras.models.clone_model(dnn_model2)

dnn_model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model2_half.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model2_quarter.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history2_full = dnn_model2.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))
history2_full_df = pandas.DataFrame(history2_full.history)
history2_full_df['epoch'] = history2_full_df.index + 1
history2_full_df['model'] = 'dnn2_full'

history2_half = dnn_model2_half.fit(half_train_images, half_train_labels, epochs=20, validation_data=(test_images, test_labels))
history2_half_df = pandas.DataFrame(history2_half.history)
history2_half_df['epoch'] = history2_half_df.index + 1
history2_half_df['model'] = 'dnn2_half'

history2_quarter = dnn_model2_quarter.fit(quarter_train_images, quarter_train_labels, epochs=20, validation_data=(test_images, test_labels))
history2_quarter_df = pandas.DataFrame(history2_quarter.history)
history2_quarter_df['epoch'] = history2_quarter_df.index + 1
history2_quarter_df['model'] = 'dnn2_quarter'

history2_df = pandas.concat([history2_full_df, history2_half_df, history2_quarter_df])

dnn_model3 = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
])
dnn_model3_half = keras.models.clone_model(dnn_model3)
dnn_model3_quarter = keras.models.clone_model(dnn_model3)

dnn_model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model3_half.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn_model3_quarter.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history3_full = dnn_model3.fit(train_images, train_labels, epochs=64, validation_data=(test_images, test_labels))
history3_full_df = pandas.DataFrame(history3_full.history)
history3_full_df['epoch'] = history3_full_df.index + 1
history3_full_df['model'] = 'dnn3_full'

history3_half = dnn_model3_half.fit(half_train_images, half_train_labels, epochs=64, validation_data=(test_images, test_labels))
history3_half_df = pandas.DataFrame(history3_half.history)
history3_half_df['epoch'] = history3_half_df.index + 1
history3_half_df['model'] = 'dnn3_half'

history3_quarter = dnn_model3_quarter.fit(quarter_train_images, quarter_train_labels, epochs=64, validation_data=(test_images, test_labels))
history3_quarter_df = pandas.DataFrame(history3_quarter.history)
history3_quarter_df['epoch'] = history3_quarter_df.index + 1
history3_quarter_df['model'] = 'dnn3_quarter'

history3_df = pandas.concat([history3_full_df, history3_half_df, history3_quarter_df])

cnn_model1 = keras.Sequential([
    keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
cnn_model1_half = keras.models.clone_model(cnn_model1)
cnn_model1_quarter = keras.models.clone_model(cnn_model1)

cnn_model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model1_half.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model1_quarter.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history4_full = cnn_model1.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))
history4_full_df = pandas.DataFrame(history4_full.history)
history4_full_df['epoch'] = history4_full_df.index + 1
history4_full_df['model'] = 'cnn1_full'

history4_half = cnn_model1_half.fit(half_train_images, half_train_labels, epochs=20, validation_data=(test_images, test_labels))
history4_half_df = pandas.DataFrame(history4_half.history)
history4_half_df['epoch'] = history4_half_df.index + 1
history4_half_df['model'] = 'cnn1_half'

history4_quarter = cnn_model1_quarter.fit(quarter_train_images, quarter_train_labels, epochs=20, validation_data=(test_images, test_labels))
history4_quarter_df = pandas.DataFrame(history4_quarter.history)
history4_quarter_df['epoch'] = history4_quarter_df.index + 1
history4_quarter_df['model'] = 'cnn1_quarter'

history4_df = pandas.concat([history4_full_df, history4_half_df, history4_quarter_df])

cnn_model2 = keras.Sequential([
    keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(8, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
cnn_model2_half = keras.models.clone_model(cnn_model2)
cnn_model2_quarter = keras.models.clone_model(cnn_model2)

cnn_model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model2_half.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model2_quarter.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history5_full = cnn_model2.fit(train_images, train_labels, epochs=25, validation_data=(test_images, test_labels))
history5_full_df = pandas.DataFrame(history5_full.history)
history5_full_df['epoch'] = history5_full_df.index + 1
history5_full_df['model'] = 'cnn2_full'

history5_half = cnn_model2_half.fit(half_train_images, half_train_labels, epochs=25, validation_data=(test_images, test_labels))
history5_half_df = pandas.DataFrame(history5_half.history)
history5_half_df['epoch'] = history5_half_df.index + 1
history5_half_df['model'] = 'cnn2_half'

history5_quarter = cnn_model2_quarter.fit(quarter_train_images, quarter_train_labels, epochs=25, validation_data=(test_images, test_labels))
history5_quarter_df = pandas.DataFrame(history5_quarter.history)
history5_quarter_df['epoch'] = history5_quarter_df.index + 1
history5_quarter_df['model'] = 'cnn2_quarter'

history5_df = pandas.concat([history5_full_df, history5_half_df, history5_quarter_df])

fullhistory_df = pandas.concat([history1_df, history2_df, history3_df, history4_df, history5_df])

fullhistory_df.to_excel('/mnt/c/Users/lenna/PycharmProjects/BA_Thesis/data/validation_data.xlsx', index=False)