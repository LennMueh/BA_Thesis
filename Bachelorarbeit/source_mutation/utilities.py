import random
import numpy as np

def modify_all_weights(model, coordinate, new_value):
    index_layer, index_neuron = coordinate
    layer = model.get_layer(index=index_layer)
    layer_weights = layer.get_weights()
    if layer.__class__.__name__ == 'Conv2D':
        layer_weights[0][:,:,:,index_neuron] = new_value
    else:
        layer_weights[0][index_neuron, :] = new_value  # Set all weights in the specified neuron to the new value
    layer.set_weights(layer_weights)
    return model

def delete_neuron(model, coordinate):
    return modify_all_weights(model, coordinate, 0)

def modify_weight_one_random_gauss(model, coordinate, sigma=0.5):
    random_value = random.gauss(0, sigma)
    return modify_all_weights(model, coordinate, random_value)

def modify_weight_all_random_gauss(model, coordinate, sigma=0.5):
    index_layer, index_neuron = coordinate
    layer = model.get_layer(index=index_layer)
    layer_weights = layer.get_weights()
    if layer.__class__.__name__ == 'Conv2D':
        random_values = np.random.normal(0, sigma, layer_weights[0][:, :, :, index_neuron].shape)
        layer_weights[0][:, :, :, index_neuron] = random_values
    else:
        random_values = np.random.normal(0, sigma, layer_weights[0][index_neuron, :].shape)
        layer_weights[0][index_neuron, :] = random_values
    layer.set_weights(layer_weights)
    return model

def modify_bias(model, coordinate, new_value):
    index_layer, index_neuron = coordinate
    layer = model.get_layer(index=index_layer)
    layer_weights = layer.get_weights()
    layer_weights[1][index_neuron] = new_value
    layer.set_weights(layer_weights)
    return model

def modify_bias_random_gauss(model, coordinate, sigma=0.5):
    random_value = random.gauss(0, sigma)
    return modify_bias(model, coordinate, random_value)

def modify_weight_one_random_uniform(model, coordinate, area=1):
    random_value = random.uniform(-area, area)
    return modify_all_weights(model, coordinate, random_value)

def modify_weight_all_random_uniform(model, coordinate, area=1):
    index_layer, index_neuron = coordinate
    layer = model.get_layer(index=index_layer)
    layer_weights = layer.get_weights()
    if layer.__class__.__name__ == 'Conv2D':
        random_values = np.random.uniform(-area, area, layer_weights[0][:, :, :, index_neuron].shape)
        layer_weights[0][:, :, :, index_neuron] = random_values
    else:
        random_values = np.random.uniform(-area, area, layer_weights[0][index_neuron, :].shape)
        layer_weights[0][index_neuron, :] = random_values
    layer.set_weights(layer_weights)
    return model

def modify_bias_random_uniform(model, coordinate, area=1):
    random_value = random.uniform(-area, area)
    return modify_bias(model, coordinate, random_value)

def modify_all_weights_by_scalar(model, coordinate, scalar):
    index_layer, index_neuron = coordinate
    layer = model.get_layer(index=index_layer)
    layer_weights = layer.get_weights()
    if layer.__class__.__name__ == 'Conv2D':
        layer_weights[0][:,:,:,index_neuron] *= scalar
    else:
        layer_weights[0][index_neuron, :] *= scalar  # Set all weights in the specified neuron to the new value
    layer.set_weights(layer_weights)
    return model

def modify_all_weights_by_scalar_random_gauss(model, coordinate, sigma=0.5):
    random_value = random.gauss(0, sigma)
    return modify_all_weights_by_scalar(model, coordinate, random_value)

def modify_all_weights_by_scalar_random_uniform(model, coordinate, area=1):
    random_value = random.uniform(-area, area)
    return modify_all_weights_by_scalar(model, coordinate, random_value)

def modify_bias_by_scalar(model, coordinate, scalar):
    index_layer, index_neuron = coordinate
    layer = model.get_layer(index=index_layer)
    layer_weights = layer.get_weights()
    layer_weights[1][index_neuron] *= scalar
    layer.set_weights(layer_weights)
    return model

def modify_bias_by_scalar_random_gauss(model, coordinate, sigma=0.5):
    random_value = random.gauss(0, sigma)
    return modify_bias_by_scalar(model, coordinate, random_value)

def modify_bias_by_scalar_random_uniform(model, coordinate, area=1):
    random_value = random.uniform(-area, area)
    return modify_bias_by_scalar(model, coordinate, random_value)

def modify_weight_all_random_by_scalar_uniform(model, coordinate, area=1):
    index_layer, index_neuron = coordinate
    layer = model.get_layer(index=index_layer)
    layer_weights = layer.get_weights()
    if layer.__class__.__name__ == 'Conv2D':
        random_values = np.random.uniform(-area, area, layer_weights[0][:, :, :, index_neuron].shape)
        layer_weights[0][:, :, :, index_neuron] *= random_values
    else:
        random_values = np.random.uniform(-area, area, layer_weights[0][index_neuron, :].shape)
        layer_weights[0][index_neuron, :] *= random_values
    layer.set_weights(layer_weights)
    return model

def modify_weight_all_random_by_scalar_gauss(model, coordinate, sigma=1):
    index_layer, index_neuron = coordinate
    layer = model.get_layer(index=index_layer)
    layer_weights = layer.get_weights()
    if layer.__class__.__name__ == 'Conv2D':
        random_values = np.random.normal(0, sigma, layer_weights[0][:, :, :, index_neuron].shape)
        layer_weights[0][:, :, :, index_neuron] *= random_values
    else:
        random_values = np.random.normal(0, sigma, layer_weights[0][index_neuron, :].shape)
        layer_weights[0][index_neuron, :] *= random_values
    layer.set_weights(layer_weights)
    return model