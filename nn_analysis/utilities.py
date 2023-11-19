import h5py
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from math import ceil
import sys
import traceback


def create_experiment_dir(model_name):
    # Create experiment directory
    experiment_path = 'experiments/' + model_name + '/'
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    return experiment_path + model_name


def load_classifications(filename, group_index=1):
    filename = filename + '_classifications.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            correct_classifications = group['correct_classifications'][:]
            misclassifications = group['misclassifications'][:]

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
    layer_outs = []

    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            if group is not None:
                i = 0
                while f'layer_outs_{i}' in group:
                    layer_outs.append(group[f'layer_outs_{i}'][:])
                    i += 1
            else:
                print(f"Group 'group{group_index}' not found in file: {filename}")
                return None
            return layer_outs
    except IOError as error:
        print("Could not open file: ", filename)
        traceback.print_exc()
        sys.exit(-1)
    except KeyError as error:
        # This exception will be caught when the dataset doesn't exist
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


def save_suspicious_neurons(suspicious_neurons, filename, approach, susp_num, group_index=1):
    filename = filename + '_' + approach + '_SN' + str(susp_num) + '_suspicious_neurons.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        #for i in range(len(suspicious_neurons)):
            #group.create_dataset("suspicious_neurons_" + str(i), data=suspicious_neurons[i])
        for i, tuple_data in enumerate(suspicious_neurons):
            list_data = list(tuple_data)
            group.create_dataset(f'suspicious_neurons_{i}', data=list_data)

    print("Suspicious neurons saved in ", filename)
    return


def load_suspicious_neurons(filename, approach, susp_num, group_index=1):
    filename = filename + '_' + approach + '_SN' + str(susp_num) + '_suspicious_neurons.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            i = 0
            suspicious_neurons = []
            #while f'suspicious_neurons_{i}' in group:
            #    suspicious_neurons.append(group[f'suspicious_neurons_{i}'][:])
            #    i += 1
            while f'suspicious_neurons_{i}' in group:
                suspicious_neurons.append(tuple(group[f'suspicious_neurons_{i}'][:]))
                i += 1
            print("Suspicious neurons loaded from ", filename)
            return suspicious_neurons
    except IOError as error:
        print("Could not open file: ", filename)
        sys.exit(-1)


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


def load_spectrum_matrices(filename, group_index=1):
    filename = filename + '_spectrum_matrices.h5'
    scores, num_cf, num_uf, num_cs, num_us = [], [], [], [], []

    try:
        with h5py.File(filename, 'r') as hf:
            group = hf['group' + str(group_index)]
            if group is not None:
                i = 0
                while f'scores_{i}' in group:
                    scores.append(group[f'scores_{i}'][:])
                    num_cf.append(group[f'num_cf_{i}'][:])
                    num_cs.append(group[f'num_cs_{i}'][:])
                    num_uf.append(group[f'num_uf_{i}'][:])
                    num_us.append(group[f'num_us_{i}'][:])
                    i += 1
            else:
                print(f"Group 'group{group_index}' not found in file: {filename}")
                return None
            return scores, num_cf, num_uf, num_cs, num_us
    except IOError as error:
        print("Could not open file: ", filename)
        traceback.print_exc()
        sys.exit(-1)
    except KeyError as error:
        # This exception will be caught when the dataset doesn't exist
        print("Layer outs loaded from ", filename)
        return scores, num_cf, num_uf, num_cs, num_us


def save_spectrum_matrices(scores, num_cf, num_uf, num_cs, num_us, filename, group_index=1):
    filename = filename + '_spectrum_matrices.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        for i in range(len(scores)):
            group.create_dataset(f'scores_{i}', data=scores[i])
            group.create_dataset(f'num_cf_{i}', data=num_cf[i])
            group.create_dataset(f'num_cs_{i}', data=num_cs[i])
            group.create_dataset(f'num_uf_{i}', data=num_uf[i])
            group.create_dataset(f'num_us_{i}', data=num_us[i])
    print("Spectrum matrices saved in ", filename)
    return
