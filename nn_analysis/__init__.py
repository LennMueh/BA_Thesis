import nn_analysis.data_managment as dm
import nn_analysis.utilities as ut
import nn_analysis.test_network as tn
import nn_analysis.analysis as an
import os


def run_analysis(model_name, approach, test_images, test_labels ,susp_num=-1, star=3, group_index=1):
    model = dm.get_model(model_name)
    experiment_path = ut.create_experiment_dir(model_name)

    file_path_classification = experiment_path + '_classifications.h5'
    file_path_layer_outs = experiment_path + '_layer_outs.h5'

    if os.path.isfile(file_path_classification) and os.path.isfile(file_path_layer_outs):
        correct_classifications, misclassifications = ut.load_classifications(experiment_path, group_index)
        layer_outs = ut.load_layer_outs(experiment_path, group_index)
    else:
        correct_classifications, misclassifications, layer_outs, predictions = tn.test_model(model, test_images,
                                                                                             test_labels)
        ut.save_classifications(correct_classifications, misclassifications, experiment_path, group_index)
        ut.save_layer_outs(layer_outs, experiment_path, group_index)

    trainable_layers = tn.get_trainable_layers(model)

    file_path_spectrum_matrices = experiment_path + '_spectrum_matrices.h5'
    if os.path.isfile(file_path_spectrum_matrices):
        scores, num_cf, num_uf, num_cs, num_us = ut.load_spectrum_matrices(experiment_path, group_index)
    else:
        scores, num_cf, num_uf, num_cs, num_us = tn.contruct_spectrum_matrices(model, trainable_layers,
                                                                               correct_classifications,
                                                                               misclassifications, layer_outs)
        ut.save_spectrum_matrices(scores, num_cf, num_uf, num_cs, num_us, experiment_path, group_index)

    num_neurons = 0
    for score in scores:
        num_neurons += len(score)
    if susp_num == -1 or susp_num > num_neurons:
        susp_num = num_neurons

    approach_name = approach
    if approach == "dstar":
        approach_name = approach + str(star)
    file_path_suspicious_neurons = experiment_path + '_' + approach_name + '_SN' + str(susp_num) + '_suspicious_neurons.h5'
    if os.path.isfile(file_path_suspicious_neurons) and approach != "random":
        suspicious_neuron_idx = ut.load_suspicious_neurons(experiment_path, approach_name, susp_num, group_index)
    else:
        match approach:
            case "tarantula":
                suspicious_neuron_idx = an.tarantula_analysis(trainable_layers, scores, num_cf, num_uf, num_cs, num_us,
                                                              susp_num)
                ut.save_suspicious_neurons(suspicious_neuron_idx, experiment_path, approach, susp_num, group_index)
            case "ochiai":
                suspicious_neuron_idx = an.ochiai_analysis(trainable_layers, scores, num_cf, num_uf, num_cs, num_us,
                                                           susp_num)
                ut.save_suspicious_neurons(suspicious_neuron_idx, experiment_path, approach, susp_num, group_index)
            case "dstar":
                approach = "dstar" + str(star)
                suspicious_neuron_idx = an.dstar_analysis(trainable_layers, scores, num_cf, num_uf, num_cs, num_us,
                                                          susp_num, star)
                ut.save_suspicious_neurons(suspicious_neuron_idx, experiment_path, approach, susp_num, group_index)
            case "random":
                suspicious_neuron_idx = an.random_neurons(trainable_layers, scores, susp_num)
    return suspicious_neuron_idx
