"""
This is the main file that executes the flow of DeepFault
"""
from test_nn import test_model
from spectrum_analysis import *
from utils import load_suspicious_neurons, save_suspicious_neurons
from utils import create_experiment_dir, get_trainable_layers
from utils import load_classifications, save_classifications
from utils import save_layer_outs, load_layer_outs, construct_spectrum_matrices
from utils import load_MNIST, load_CIFAR, filter_val_set
from sklearn.model_selection import train_test_split
import random

if __name__ == "__main__":
    model_name = args['model']
    dataset = 'mnist'
    selected_class = 0
    step_size = 1
    approach = 'random' # analysis formula
    susp_num = 1
    repeat = 1
    seed = 42
    star = 3

    ####################
    # 0) Load MNIST or CIFAR10 data
    if dataset == 'mnist':
        X_train, Y_train, X_test, Y_test = load_MNIST(one_hot=True)
    else:
        X_train, Y_train, X_test, Y_test = load_CIFAR()


    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                      test_size=1/6.0,
                                                      random_state=seed)

    experiment_name = create_experiment_dir(experiment_path, model_name,
                                            selected_class, step_size,
                                            approach, susp_num, repeat)

    #Fault localization is done per class.
    X_val, Y_val = filter_val_set(selected_class, X_test, Y_test)


    ####################
    # 2)test the model and receive the indexes of correct and incorrect classifications
    # Also provide output of each neuron in each layer for test input x.
    filename = experiment_path + '/' + model_name + '_' + str(selected_class)
    try:
        correct_classifications, misclassifications = load_classifications(filename, group_index)
        layer_outs = load_layer_outs(filename, group_index)
    except:
        correct_classifications, misclassifications, layer_outs, predictions =\
                test_model(model, X_val, Y_val)
        save_classifications(correct_classifications, misclassifications,
                             filename, group_index)
        save_layer_outs(layer_outs, filename, group_index)


    ####################
    # 3) Receive the correct classifications  & misclassifications and identify
    # the suspicious neurons per layer

    # Do below.


    trainable_layers = get_trainable_layers(model)
    scores, num_cf, num_uf, num_cs, num_us = construct_spectrum_matrices(model,
                                                                        trainable_layers,
                                                                        correct_classifications,
                                                                        misclassifications,
                                                                        layer_outs)



    filename = experiment_path + '/' + model_name + '_C' + str(selected_class) + '_' +\
    approach +  '_SN' +  str(susp_num)

    if approach == 'tarantula':
        try:
            suspicious_neuron_idx = load_suspicious_neurons(filename, group_index)
        except:
            suspicious_neuron_idx = tarantula_analysis(trainable_layers, scores,
                                                 num_cf, num_uf, num_cs, num_us,
                                                 susp_num)

            save_suspicious_neurons(suspicious_neuron_idx, filename, group_index)

    elif approach == 'ochiai':
        try:
            suspicious_neuron_idx = load_suspicious_neurons(filename, group_index)
        except:
            suspicious_neuron_idx = ochiai_analysis(trainable_layers, scores,
                                                 num_cf, num_uf, num_cs, num_us,
                                                 susp_num)

            save_suspicious_neurons(suspicious_neuron_idx, filename, group_index)

    elif approach == 'dstar':
        try:
            suspicious_neuron_idx = load_suspicious_neurons(filename, group_index)
        except:
            suspicious_neuron_idx = dstar_analysis(trainable_layers, scores,
                                                 num_cf, num_uf, num_cs, num_us,
                                                 susp_num, star)

            save_suspicious_neurons(suspicious_neuron_idx, filename, group_index)

    elif approach == 'random':
        # Random fault localization has to be run after running Tarantula,
        # Ochiai and DStar with the same parameters.

        filename = experiment_path + '/' + model_name + '_C' + str(selected_class) \
        + '_tarantula_' + 'SN' + str(susp_num)

        suspicious_neuron_idx_tarantula = load_suspicious_neurons(filename, group_index)

        filename = experiment_path + '/' + model_name + '_C' + str(selected_class) \
        + '_ochiai_' + 'SN' + str(susp_num)

        suspicious_neuron_idx_ochiai = load_suspicious_neurons(filename, group_index)

        filename = experiment_path + '/' + model_name + '_C' + str(selected_class) \
        + '_dstar_' + 'SN' + str(susp_num)

        suspicious_neuron_idx_dstar = load_suspicious_neurons(filename, group_index)

        forbiddens = suspicious_neuron_idx_ochiai + suspicious_neuron_idx_tarantula  + \
        suspicious_neuron_idx_dstar

        forbiddens = [list(forb) for forb in forbiddens]

        available_layers = list(([elem[0] for elem in suspicious_neuron_idx_tarantula]))
        available_layers += list(set([elem[0] for elem in suspicious_neuron_idx_ochiai]))
        available_layers += list(set([elem[0] for elem in suspicious_neuron_idx_dstar]))

        suspicious_neuron_idx = []
        while len(suspicious_neuron_idx) < susp_num:
            l_idx = random.choice(available_layers)
            n_idx = random.choice(range(model.layers[l_idx].output_shape[1]))

            if [l_idx, n_idx] not in forbiddens and [l_idx, n_idx] not in suspicious_neuron_idx:
                suspicious_neuron_idx.append([l_idx, n_idx])