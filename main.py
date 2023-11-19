import nn_modification.utilities as nnm
import nn_analysis.data_managment as dm

model = dm.get_model("dnn_hid1_relu_6epoch")
coordinate = (1, 0)
new_model = nnm.modify_all_weights(model, coordinate, 0.0)
print("Stop")