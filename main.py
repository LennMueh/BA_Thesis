import utilities as ut

tarantula = ut.run_analysis("cnn_conv2_relu_6epoch", "tarantula")
ochiai = ut.run_analysis("cnn_conv2_relu_6epoch", "ochiai")
dstar = ut.run_analysis("cnn_conv2_relu_6epoch", "dstar")
random = ut.run_analysis("cnn_conv2_relu_6epoch", "random")
print("Stop")