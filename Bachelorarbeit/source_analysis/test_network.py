import numpy as np
import nn_analysis.utilities as an
import time


def test_model(model, X_test, Y_test):
    # Find activations of each neuron in each layer for each input x in X_test
    layer_outs = an.get_layer_outs(model, X_test)

    # Print information about the model
    print(model.summary())

    # Evaluate the model
    score = model.evaluate(X_test, np.argmax(Y_test, axis=1), verbose=0)
    print('[loss, accuracy] -> ' + str(score))

    # Make predictions
    Y_pred = model.predict(X_test)

    # Calculate classification report and confusion matrix
    an.calculate_prediction_metrics(Y_test, Y_pred, score)

    # Find test and prediction classes
    expectations = np.argmax(Y_test, axis=1)
    predictions = np.argmax(Y_pred, axis=1)

    classifications = np.absolute(expectations - predictions)

    # Find correct classifications and misclassifications
    correct_classifications = []
    misclassifications = []
    for i in range(0, len(classifications)):
        if classifications[i] == 0:
            correct_classifications.append(i)
        else:
            misclassifications.append(i)

    print("Testing done!\n")

    return correct_classifications, misclassifications, layer_outs, predictions


def get_trainable_layers(model):
    trainable_layers = []
    for layer in model.layers:
        try:
            weights = layer.get_weights()[0]
            trainable_layers.append(model.layers.index(layer))
        except:
            pass

    trainable_layers = trainable_layers[:-1]  # ignore the output layer

    return trainable_layers

def contruct_spectrum_matrices(model, trainable_layers, correct_classifications, misclassifications,
                               layer_outs, activation_threshold=0):
    start_time = time.time()
    scores = []
    activated_fail = []  # num_cf
    activated_success = []  # num_cs
    not_activated_fail = []  # num_uf
    not_activated_success = []  # num_us

    for layer in trainable_layers:
        shape = model.layers[layer].output_shape[-1]
        activated_fail.append(np.zeros(shape))
        activated_success.append(np.zeros(shape))
        not_activated_fail.append(np.zeros(shape))
        not_activated_success.append(np.zeros(shape))
        scores.append(np.zeros(shape))

    for layer, layer_index in zip(trainable_layers, range(len(trainable_layers))):
        for test_index, outlayer in enumerate(layer_outs[layer][0]):
            for neuron_index in range(model.layers[layer].output_shape[-1]):
                mean_out = np.mean(outlayer[..., neuron_index])
                if test_index in correct_classifications:
                    if mean_out > activation_threshold:
                        activated_success[layer_index][neuron_index] += 1
                    else:
                        not_activated_success[layer_index][neuron_index] += 1
                elif test_index in misclassifications:
                    if mean_out > activation_threshold:
                        activated_fail[layer_index][neuron_index] += 1
                    else:
                        not_activated_fail[layer_index][neuron_index] += 1

    end_time = time.time()
    print("Time to construct spectrum matrices: ", end_time - start_time)
    return scores, activated_fail, not_activated_fail, activated_success, not_activated_success
