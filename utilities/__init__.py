import utilities.data_managment as dm
import utilities.utilities as ut
import utilities.test_network as tn
import utilities.analysis as an


def run_analysis(model_name, approach, susp_num=-1, star=3, group_index=1):
    model = dm.get_model(model_name)
    train_images, train_labels, test_images, test_labels = dm.get_data()
    experiment_path = ut.create_experiment_dir(model_name)
    try:
        correct_classifications, misclassifications = ut.load_classifications(experiment_path, group_index)
        layer_outs = ut.load_layer_outs(experiment_path, group_index)
    except:
        correct_classifications, misclassifications, layer_outs, predictions = tn.test_model(model, test_images,
                                                                                             test_labels)
        ut.save_classifications(correct_classifications, misclassifications, experiment_path, group_index)
        ut.save_layer_outs(layer_outs, experiment_path, group_index)
    trainable_layers = tn.get_trainable_layers(model)
    try:
        scores, num_cf, num_uf, num_cs, num_us = ut.load_spectrum_matrices(experiment_path, group_index)
    except:
        scores, num_cf, num_uf, num_cs, num_us = tn.contruct_spectrum_matrices(model, trainable_layers,
                                                                               correct_classifications,
                                                                               misclassifications, layer_outs)
        ut.save_spectrum_matrices(scores, num_cf, num_uf, num_cs, num_us, experiment_path, group_index)
    if susp_num == -1:
        susp_num = 0
        for score in scores:
            susp_num += len(score)
    match approach:
        case "tarantula":
            try:
                suspicious_neuron_idx = ut.load_suspicious_neurons(experiment_path, approach, susp_num, group_index)
            except:
                suspicious_neuron_idx = an.tarantula_analysis(trainable_layers, scores, num_cf, num_uf, num_cs, num_us,
                                                              susp_num)
                ut.save_suspicious_neurons(suspicious_neuron_idx, experiment_path, approach, susp_num, group_index)
        case "ochiai":
            try:
                suspicious_neuron_idx = ut.load_suspicious_neurons(experiment_path, approach, susp_num, group_index)
            except:
                suspicious_neuron_idx = an.ochiai_analysis(trainable_layers, scores, num_cf, num_uf, num_cs, num_us,
                                                           susp_num)
                ut.save_suspicious_neurons(suspicious_neuron_idx, experiment_path, approach, susp_num, group_index)
        case "dstar":
            approach = "dstar" + str(star)
            try:
                suspicious_neuron_idx = ut.load_suspicious_neurons(experiment_path, approach, susp_num, group_index)
            except:
                suspicious_neuron_idx = an.dstar_analysis(trainable_layers, scores, num_cf, num_uf, num_cs, num_us,
                                                          susp_num, star)
                ut.save_suspicious_neurons(suspicious_neuron_idx, experiment_path, approach, susp_num, group_index)
        case "random":
            suspicious_neuron_idx = an.random_neurons(trainable_layers, scores, susp_num)
    return suspicious_neuron_idx
