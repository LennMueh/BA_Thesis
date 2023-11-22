import nn_analysis as nn

tarantula_1 = nn.run_analysis("cnn_conv2_relu_1epoch", "tarantula")
tarantula_6 = nn.run_analysis("cnn_conv2_relu_6epoch", "tarantula")
oichai_1 = nn.run_analysis("cnn_conv2_relu_1epoch", "ochiai")
oichai_6 = nn.run_analysis("cnn_conv2_relu_6epoch", "ochiai")
dstar_1 = nn.run_analysis("cnn_conv2_relu_1epoch", "dstar")
dstar_6 = nn.run_analysis("cnn_conv2_relu_6epoch", "dstar")
random_1 = nn.run_analysis("cnn_conv2_relu_1epoch", "random")
random_6 = nn.run_analysis("cnn_conv2_relu_6epoch", "random")
print("Stop here")