import utilities.data_managment as dm
import utilities.test_network as tn
import utilities.analysis as sa

susp_num = 10
star = 3

model = dm.get_model("cnn_conv2_relu_6epoch")
train_images, train_labels, test_images, test_labels = dm.get_data()
correct_classifications, misclassifications, layer_outs, predictions = tn.test_model(model, test_images, test_labels)
trainable_layers = tn.get_trainable_layers(model)
scores, num_cf, num_uf, num_cs, num_us = tn.contruct_spectrum_matrices(model, trainable_layers, correct_classifications, misclassifications, layer_outs)
random = sa.random_neurons(trainable_layers, scores, 10)
print(random)
"""tarantula = sa.tarantula_analysis(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, susp_num)
ochiai = sa.ochiai_analysis(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, susp_num)
dstar = sa.dstar_analysis(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, susp_num, star)
combine = tarantula + ochiai + dstar
combine = list(set(combine))
print("Tarantula: ", len(tarantula))
print("Ochiai: ", len(ochiai))
print("Dstar: ", len(dstar))
print("Combine: ", len(combine))"""