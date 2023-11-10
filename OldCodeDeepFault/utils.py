from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from keras import backend as K
import sys
from sklearn.metrics import classification_report, confusion_matrix
from math import ceil
import numpy as np
import h5py
from os import path, makedirs
import traceback


def load_CIFAR(one_hot=True):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if one_hot:
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test  = np_utils.to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test


def load_MNIST(one_hot=True, channel_first=True):
    """
    Load MNIST data
    :param one_hot:
    :return:
    """
    #Load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #Preprocess dataset
    #Normalization and reshaping of input.
    if channel_first:
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if one_hot:
        #For output, it is important to change number to one-hot vector.
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test  = np_utils.to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test


def get_layer_outs(model, test_input):
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    layer_outs = [func([test_input]) for func in functors]

    return layer_outs


def calculate_prediction_metrics(Y_test, Y_pred, score):
    """
    Calculate classification report and confusion matrix
    :param Y_test:
    :param Y_pred:
    :param score:
    :return:
    """
    #Find test and prediction classes
    Y_test_class = np.argmax(Y_test, axis=1)
    Y_pred_class = np.argmax(Y_pred, axis=1)

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

    acc = sum([np.argmax(Y_test[i]) == np.argmax(Y_pred[i]) for i in range(len(Y_test))]) / len(Y_test)
    v1 = ceil(acc*10000)/10000
    v2 = ceil(score[1]*10000)/10000
    correct_accuracy_calculation =  v1 == v2
    try:
        if not correct_accuracy_calculation:
            raise Exception("Accuracy results don't match to score")
    except Exception as error:
        print("Caught this error: " + repr(error))


def create_experiment_dir(experiment_path, model_name,
                            selected_class, step_size,
                            approach, susp_num, repeat):

    # define experiment name, create directory experiments directory if it
    # doesnt exist
    experiment_name = model_name + '_C' + str(selected_class) + '_SS' + \
    str(step_size) + '_' + approach + '_SN' + str(susp_num) + '_R' + str(repeat)


    if not path.exists(experiment_path):
        makedirs(experiment_path)

    return experiment_name


def save_classifications(correct_classifications, misclassifications, filename, group_index):
    filename = filename + '_classifications.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group'+str(group_index))
        group.create_dataset("correct_classifications", data=correct_classifications)
        group.create_dataset("misclassifications", data=misclassifications)

    print("Classifications saved in ", filename)
    return


def load_classifications(filename, group_index):
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


def save_layer_outs(layer_outs, filename, group_index):
    filename = filename + '_layer_outs.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group'+str(group_index))
        for i in range(len(layer_outs)):
            group.create_dataset("layer_outs_"+str(i), data=layer_outs[i])

    print("Layer outs saved in ", filename)
    return


def load_layer_outs(filename, group_index):
    filename = filename + '_layer_outs.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            i = 0
            layer_outs = []
            while True:
                layer_outs.append(group.get('layer_outs_'+str(i)).value)
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


def save_suspicious_neurons(suspicious_neurons, filename, group_index):
    filename = filename + '_suspicious_neurons.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group'+str(group_index))
        for i in range(len(suspicious_neurons)):
            group.create_dataset("suspicious_neurons"+str(i), data=suspicious_neurons[i])

    print("Suspicious neurons saved in ", filename)
    return


def load_suspicious_neurons(filename, group_index):
    filename = filename + '_suspicious_neurons.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            i = 0
            suspicious_neurons = []
            while True:
                suspicious_neurons.append(group.get('suspicious_neurons' + str(i)).value)
                i += 1

    except (IOError) as error:
        print("Could not open file: ", filename)
        sys.exit(-1)
    except (AttributeError) as error:
        # because we don't know the exact dimensions (number of layers of our network)
        # we leave it to iterate until it throws an attribute error, and then return
        # layer outs to the caller function
        print("Suspicious neurons  loaded from ", filename)
        return suspicious_neurons


def filter_val_set(desired_class, X, Y):
    X_class = []
    Y_class = []
    for x,y in zip(X,Y):
        if y[desired_class] == 1:
            X_class.append(x)
            Y_class.append(y)

    print("Validation set filtered for desired class: " + str(desired_class))

    return np.array(X_class), np.array(Y_class)


def get_trainable_layers(model):

    trainable_layers = []
    for layer in model.layers:
        try:
            weights = layer.get_weights()[0]
            trainable_layers.append(model.layers.index(layer))
        except:
            pass

    trainable_layers = trainable_layers[:-1]  #ignore the output layer

    return trainable_layers


def construct_spectrum_matrices(model, trainable_layers,
                                correct_classifications, misclassifications,
                                layer_outs, activation_threshold=0):
    scores = []
    num_cf = []
    num_uf = []
    num_cs = []
    num_us = []
    for tl in trainable_layers:
        print(model.layers[tl].output_shape)
        num_cf.append(np.zeros(model.layers[tl].output_shape[-1]))  # covered (activated) and failed
        num_uf.append(np.zeros(model.layers[tl].output_shape[-1]))  # uncovered (not activated) and failed
        num_cs.append(np.zeros(model.layers[tl].output_shape[-1]))  # covered and succeeded
        num_us.append(np.zeros(model.layers[tl].output_shape[-1]))  # uncovered and succeeded
        scores.append(np.zeros(model.layers[tl].output_shape[-1]))


    for tl in trainable_layers:
        layer_idx = trainable_layers.index(tl)
        all_neuron_idx = range(model.layers[tl].output_shape[-1])
        test_idx = 0
        for l in layer_outs[tl][0]:
            for neuron_idx in range(model.layers[tl].output_shape[-1]):
                if test_idx in correct_classifications and np.mean(l[...,neuron_idx]) > activation_threshold:
                    num_cs[layer_idx][neuron_idx] += 1
                elif test_idx in correct_classifications and np.mean(l[...,neuron_idx]) <= activation_threshold:
                    num_us[layer_idx][neuron_idx] += 1
                elif test_idx in misclassifications and np.mean(l[...,neuron_idx]) > activation_threshold:
                    num_cf[layer_idx][neuron_idx] += 1
                else:
                    num_uf[layer_idx][neuron_idx] += 1

            test_idx += 1
            '''
            covered_idx   = list(np.where(l  > 0)[0])
            uncovered_idx = list(set(all_neuron_idx) - set(covered_idx))
            #uncovered_idx = list(np.where(l <= 0)[0])
            if test_idx  in correct_classifications:
                for cov_idx in covered_idx:
                    num_cs[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_us[layer_idx][uncov_idx] += 1
            elif test_idx in misclassifications:
                for cov_idx in covered_idx:
                    num_cf[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_uf[layer_idx][uncov_idx] += 1
            test_idx += 1
            '''

    return scores, num_cf, num_uf, num_cs, num_us

def weight_analysis(model):
    threshold_weight = 0.1
    deactivatables = []
    for i in range(2, target_layer + 1):
        for k in range(model.layers[i - 1].output_shape[1]):
            neuron_weights = model.layers[i].get_weights()[0][k]
            deactivate = True
            for j in range(len(neuron_weights)):
                if neuron_weights[j] > threshold_weight:
                    deactivate = False

            if deactivate:
                deactivatables.append((i,k))

    return deactivatables

