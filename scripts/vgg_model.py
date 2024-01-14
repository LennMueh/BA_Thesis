import tensorflow as tf
from sklearn.model_selection import train_test_split

cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0
half_train_images, _, half_train_labels, _ = train_test_split(train_images, train_labels, test_size=0.5, random_state=42)
quarter_train_images, _, quarter_train_labels, _ = train_test_split(train_images, train_labels, test_size=0.75, random_state=42)

tiny_vgg = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(10, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(10, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(10, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(10, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
tiny_vgg_half = tf.keras.models.clone_model(tiny_vgg)
tiny_vgg_quarter = tf.keras.models.clone_model(tiny_vgg)

tiny_vgg.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tiny_vgg_half.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tiny_vgg_quarter.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

tiny_vgg.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
tiny_vgg_half.fit(half_train_images, half_train_labels, epochs=1, validation_data=(test_images, test_labels))
tiny_vgg_quarter.fit(quarter_train_images, quarter_train_labels, epochs=1, validation_data=(test_images, test_labels))

tf.keras.saving.save_model(tiny_vgg, '../models/tiny_vgg_1epoch_full')
tf.keras.saving.save_model(tiny_vgg_half, '../models/tiny_vgg_1epoch_half')
tf.keras.saving.save_model(tiny_vgg_quarter, '../models/tiny_vgg_1epoch_quarter')


tiny_vgg.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
tiny_vgg_half.fit(half_train_images, half_train_labels, epochs=5, validation_data=(test_images, test_labels))
tiny_vgg_quarter.fit(quarter_train_images, quarter_train_labels, epochs=5, validation_data=(test_images, test_labels))

tf.keras.saving.save_model(tiny_vgg, '../models/tiny_vgg_6epoch_full')
tf.keras.saving.save_model(tiny_vgg_half, '../models/tiny_vgg_6epoch_half')
tf.keras.saving.save_model(tiny_vgg_quarter, '../models/tiny_vgg_6epoch_quarter')