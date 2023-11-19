import random

def modify_all_weights(model, coordinate, new_value):
    """
    Modify all weights in a model to a new value.

    Args:
        model: A tf.keras.Model object.
        coordinate: A tuple of 2 integers, representing the coordinate of the weight to be modified.
        new_value: A float, representing the new value of the weight.
    """
    index_layer, index_neuron = coordinate
    layer = model.get_layer(index=index_layer)
    layer_weights = layer.get_weights()
    layer_weights[0][index_neuron][:] = new_value  # Set all weights in the specified neuron to the new value
    layer.set_weights(layer_weights)
    return model

def delete_neuron(model, coordinate):
    """
    Delete a neuron in a model.

    Args:
        model: A tf.keras.Model object.
        coordinate: A tuple of 2 integers, representing the coordinate of the neuron to be deleted.
    """
    return modify_all_weights(model, coordinate, 0)

def modify_weight_one_random(model, coordinate, sigma=0.5):
    """
    Modify a weight of a neuron to a random value.

    Args:
        model: A tf.keras.Model object.
        coordinate: A tuple of 2 integers, representing the coordinate of the weight to be modified.
    """
    random_value = random.gauss(0, sigma)
    return modify_all_weights(model, coordinate, random_value)

def modify_weight_all_random(model, coordinate, sigma=0.5):
    """
    Modify all weights of a neuron to random values.

    Args:
        model: A tf.keras.Model object.
        coordinate: A tuple of 2 integers, representing the coordinate of the weight to be modified.
    """
    index_layer, index_neuron = coordinate
    layer = model.get_layer(index=index_layer)
    layer_weights = layer.get_weights()
    for i in range(len(layer_weights[0][index_neuron])):
        layer_weights[0][index_neuron][i] = random.gauss(0, sigma)
    layer.set_weights(layer_weights)
    return model

def modify_bias(model, coordinate, new_value):
    """
    Modify the bias of a neuron to a new value.

    Args:
        model: A tf.keras.Model object.
        coordinate: A tuple of 2 integers, representing the coordinate of the bias to be modified.
        new_value: A float, representing the new value of the bias.
    """
    index_layer, index_neuron = coordinate
    layer = model.get_layer(index=index_layer)
    layer_weights = layer.get_weights()
    layer_weights[1][index_neuron] = new_value
    layer.set_weights(layer_weights)
    return model

def modify_bias_random(model, coordinate, sigma=0.5):
    """
    Modify the bias of a neuron to a random value.

    Args:
        model: A tf.keras.Model object.
        coordinate: A tuple of 2 integers, representing the coordinate of the bias to be modified.
    """
    random_value = random.gauss(0, sigma)
    return modify_bias(model, coordinate, random_value)