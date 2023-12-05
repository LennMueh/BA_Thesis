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
history1_full_df['model'] = 'dnn_model1_full'

history1_half = dnn_model1_half.fit(half_train_images, half_train_labels, epochs=20, validation_data=(test_images, test_labels))
history1_half_df = pandas.DataFrame(history1_half.history)
history1_half_df['epoch'] = history1_half_df.index + 1
history1_half_df['model'] = 'dnn_model1_half'

history1_quarter = dnn_model1_quarter.fit(quarter_train_images, quarter_train_labels, epochs=20, validation_data=(test_images, test_labels))
history1_quarter_df = pandas.DataFrame(history1_quarter.history)
history1_quarter_df['epoch'] = history1_quarter_df.index + 1
history1_quarter_df['model'] = 'dnn_model1_quarter'

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
history2_full_df['model'] = 'dnn_model2_full'

history2_half = dnn_model2_half.fit(half_train_images, half_train_labels, epochs=20, validation_data=(test_images, test_labels))
history2_half_df = pandas.DataFrame(history2_half.history)
history2_half_df['epoch'] = history2_half_df.index + 1
history2_half_df['model'] = 'dnn_model2_half'

history2_quarter = dnn_model2_quarter.fit(quarter_train_images, quarter_train_labels, epochs=20, validation_data=(test_images, test_labels))
history2_quarter_df = pandas.DataFrame(history2_quarter.history)
history2_quarter_df['epoch'] = history2_quarter_df.index + 1
history2_quarter_df['model'] = 'dnn_model2_quarter'

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

history3_full = dnn_model3.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))
history3_full_df = pandas.DataFrame(history3_full.history)
history3_full_df['epoch'] = history3_full_df.index + 1
history3_full_df['model'] = 'dnn_model3_full'

history3_half = dnn_model3_half.fit(half_train_images, half_train_labels, epochs=20, validation_data=(test_images, test_labels))
history3_half_df = pandas.DataFrame(history3_half.history)
history3_half_df['epoch'] = history3_half_df.index + 1
history3_half_df['model'] = 'dnn_model3_half'

history3_quarter = dnn_model3_quarter.fit(quarter_train_images, quarter_train_labels, epochs=20, validation_data=(test_images, test_labels))
history3_quarter_df = pandas.DataFrame(history3_quarter.history)
history3_quarter_df['epoch'] = history3_quarter_df.index + 1
history3_quarter_df['model'] = 'dnn_model3_quarter'

history3_df = pandas.concat([history3_full_df, history3_half_df, history3_quarter_df])
fullhistory_df = pandas.concat([history1_df, history2_df, history3_df])

fullhistory_df.to_excel('/mnt/c/Users/lenna/PycharmProjects/BA_Thesis/data/validation_data.xlsx', index=False)