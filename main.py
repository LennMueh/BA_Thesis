import utilities.data_managment as dm
import utilities.test_network as tn
import numpy as np

model = dm.get_model("dnn_hid2_gelu_6epoch")
train_images, train_labels, test_images, test_labels = dm.get_data()
correct_classifications, misclassifications, layer_outs, predictions = tn.test_model(model, test_images, test_labels)
trainable_layers = tn.get_trainable_layers(model)
scores, num_cf, num_uf, num_cs, num_us = tn.contruct_spectrum_matrices(model, trainable_layers, correct_classifications, misclassifications, layer_outs)