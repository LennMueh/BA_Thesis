import utilities.data_managment as dm
import utilities.test_network as tn

model = dm.get_model("dnn_hid2_gelu_6epoch")
train_images, train_labels, test_images, test_labels = dm.get_data()
correct_classifications, misclassifications, layer_outs, predictions = tn.test_model(model, test_images, test_labels)

print("Hello man")