import h5py
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from math import ceil


def create_experiment_dir(model_name, approach, star):
    # Create experiment directory
    experiment_path = 'experiments/' + model_name + '_' + approach + '_' + str(star)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    else:
        os.rmdir(experiment_path)
        os.makedirs(experiment_path)
    return experiment_path


def load_classifications(filename, group_index=1):
    filename = filename + '_classifications.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            correct_classifications = group.get('correct_classifications').value
            misclassifications = group.get('misclassifications').value

            print("Classifications loaded from ", filename)
            return correct_classifications, misclassifications
    except (IOError) as error:
        print("Could not open file: ", filename)
        sys.exit(-1)


def save_classifications(correct_classifications, misclassifications, filename, group_index=1):
    filename = filename + '_classifications.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        group.create_dataset("correct_classifications", data=correct_classifications)
        group.create_dataset("misclassifications", data=misclassifications)

    print("Classifications saved in ", filename)
    return


def load_layer_outs(filename, group_index=1):
    filename = filename + '_layer_outs.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            i = 0
            layer_outs = []
            while True:
                layer_outs.append(group.get('layer_outs_' + str(i)).value)
                i += 1

    except (IOError) as error:
        print("Could not open file: ", filename)
        traceback.print_exc()
        sys.exit(-1)
    except (AttributeError) as error:
        # because we don't know the exact dimensions (number of layers of our network)
        # we leave it to iterate until it throws an attribute error, and then return
        # layer outs to the caller function
        print("Layer outs loaded from ", filename)
        return layer_outs


def save_layer_outs(layer_outs, filename, group_index=1):
    filename = filename + '_layer_outs.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        for i in range(len(layer_outs)):
            group.create_dataset("layer_outs_" + str(i), data=layer_outs[i])

    print("Layer outs saved in ", filename)
    return


def get_layer_outs(model, test_input):
    inp = model.input
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [tf.keras.backend.function([inp], [out]) for out in outputs]  # evaluation functions
    layer_outs = [func([test_input]) for func in functors]
    return layer_outs


def calculate_prediction_metrics(test_labels, test_prediction, score):
    Y_test_class = np.argmax(test_labels, axis=1)
    Y_pred_class = np.argmax(test_prediction, axis=1)

    classifications = np.absolute(Y_test_class - Y_pred_class)

    correct_classifications = []
    incorrect_classifications = []
    for i in range(1, len(classifications)):
        if (classifications[i] == 0):
            correct_classifications.append(i)
        else:
            incorrect_classifications.append(i)

    # Accuracy of the predicted values
    print(classification_report(Y_test_class, Y_pred_class))
    print(confusion_matrix(Y_test_class, Y_pred_class))

    acc = sum([np.argmax(test_labels[i]) == np.argmax(test_prediction[i]) for i in range(len(test_labels))]) / len(
        test_labels)
    v1 = ceil(acc * 10000) / 10000
    v2 = ceil(score[1] * 10000) / 10000
    correct_accuracy_calculation = v1 == v2
    try:
        if not correct_accuracy_calculation:
            raise Exception("Accuracy results don't match to score")
    except Exception as error:
        print("Caught this error: " + repr(error))