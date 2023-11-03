import tensorflow as tf

(_,_), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
test_images = test_images / 255.0

model1 = tf.keras.saving.load_model("models/dnn_hid1_relu_5epoch")
model2 = tf.keras.saving.load_model("models/dnn_hid1_relu_10epoch")
model3 = tf.keras.saving.load_model("models/dnn_hid2_relu_5epoch")
model4 = tf.keras.saving.load_model("models/dnn_hid2_relu_10epoch")

print(model1.evaluate(test_images, test_labels))
print(model2.evaluate(test_images,test_labels))
print(model3.evaluate(test_images, test_labels))
print(model4.evaluate(test_images,test_labels))
