import nn_modification as nn_modification
import nn_modification.utilities as utilities
import time
import tensorflow.keras.datasets.fashion_mnist as fashion_mnist
from faker import Faker
from sklearn.model_selection import train_test_split
import sys
import nn_analysis.data_managment as data_managment
import tensorflow as tf

fake = Faker(['de_DE', 'en_US', 'fr_FR', 'es_ES'])
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images_half, _, train_labels_half, _ = train_test_split(train_images, train_labels, test_size=0.5,
                                                              random_state=42)
train_images_quarter, _, train_labels_quarter, _ = train_test_split(train_images, train_labels, test_size=0.75,
                                                                    random_state=42)
# select = "cnn1_1epoch_full"
full_start = time.time()
modelnames = [("cnn1_1epoch_full", train_images, train_labels, test_images, test_labels),
              ("cnn1_1epoch_quarter", train_images_quarter, train_labels_quarter, test_images, test_labels),
              ("cnn1_6epoch_half", train_images_half, train_labels_half, test_images, test_labels),
              ("cnn2_1epoch_full", train_images, train_labels, test_images, test_labels),
              ("cnn2_1epoch_quarter", train_images_quarter, train_labels_quarter, test_images, test_labels),
              ("cnn2_6epoch_half", train_images_half, train_labels_half, test_images, test_labels),
              ("dnn1_1epoch_full", train_images, train_labels, test_images, test_labels),
              ("dnn1_1epoch_quarter", train_images_quarter, train_labels_quarter, test_images, test_labels),
              ("dnn1_6epoch_half", train_images_half, train_labels_half, test_images, test_labels),
              ("dnn2_1epoch_full", train_images, train_labels, test_images, test_labels),
              ("dnn2_1epoch_quarter", train_images_quarter, train_labels_quarter, test_images, test_labels),
              ("dnn2_6epoch_half", train_images_half, train_labels_half, test_images, test_labels),
              ("dnn3_1epoch_full", train_images, train_labels, test_images, test_labels),
              ("dnn3_1epoch_quarter", train_images_quarter, train_labels_quarter, test_images, test_labels),
              ("dnn3_6epoch_half", train_images_half, train_labels_half, test_images, test_labels),
              ("cnn1_1epoch_half", train_images_half, train_labels_half, test_images, test_labels),
              ("cnn1_6epoch_full", train_images, train_labels, test_images, test_labels),
              ("cnn1_6epoch_quarter", train_images_quarter, train_labels_quarter, test_images, test_labels),
              ("cnn2_1epoch_half", train_images_half, train_labels_half, test_images, test_labels),
              ("cnn2_6epoch_full", train_images, train_labels, test_images, test_labels),
              ("cnn2_6epoch_quarter", train_images_quarter, train_labels_quarter, test_images, test_labels),
              ("dnn1_1epoch_half", train_images_half, train_labels_half, test_images, test_labels),
              ("dnn1_6epoch_full", train_images, train_labels, test_images, test_labels),
              ("dnn1_6epoch_quarter", train_images_quarter, train_labels_quarter, test_images, test_labels),
              ("dnn2_1epoch_half", train_images_half, train_labels_half, test_images, test_labels),
              ("dnn2_6epoch_full", train_images, train_labels, test_images, test_labels),
              ("dnn2_6epoch_quarter", train_images_quarter, train_labels_quarter, test_images, test_labels),
              ("dnn3_1epoch_half", train_images_half, train_labels_half, test_images, test_labels),
              ("dnn3_6epoch_full", train_images, train_labels, test_images, test_labels),
              ("dnn3_6epoch_quarter", train_images_quarter, train_labels_quarter, test_images, test_labels)]
filter_models = [model for model in modelnames if sys.argv[1] in model[0]]
# filter_models = [model for model in modelnames if select in model[0]]
analysis_approach = ["tarantula"]  # , "random"]  # , "ochiai", "dstar", "random"]
mutation_function = [utilities.modify_weight_one_random_gauss, utilities.modify_weight_all_random_gauss,
                     utilities.modify_bias,
                     utilities.modify_bias_random_gauss, utilities.modify_all_weights,
                     utilities.modify_all_weights_by_scalar,
                     utilities.modify_all_weights_by_scalar_random_gauss,
                     utilities.modify_weight_all_random_by_scalar_gauss]
"""[utilities.modify_weight_one_random_gauss, utilities.modify_weight_all_random_gauss,
                 utilities.modify_bias,
                 utilities.modify_bias_random_gauss, utilities.modify_all_weights,
                 utilities.modify_weight_all_random_uniform, utilities.modify_weight_one_random_uniform,
                 utilities.modify_bias_random_uniform, utilities.modify_all_weights_by_scalar,
                 utilities.modify_all_weights_by_scalar_random_gauss,
                 utilities.modify_all_weights_by_scalar_random_uniform,
                 utilities.modify_bias, utilities.modify_bias_random_gauss, utilities.modify_bias_random_uniform,
                 utilities.modify_weight_all_random_by_scalar_uniform,
                 utilities.modify_weight_all_random_by_scalar_gauss]"""
train_between_iterations = [True, False]
value = [-1, -0.5, 0, 0.5, 1]  # [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
compare_loss = [False, True]
compare_accuracy = [False, True]
compare_and_both = [False, True]
regression_loss_offset = [True]  # , False]
regression_accuracy_offset = [True]  # , False]
loss_offset = [0.005]  # [0.05, 0.005]
accuracy_offset = [0.01]  # [0, 0.1, 0.01, 0.001]
for i in range(1):
    iteration_start = time.time()
    for model in filter_models:
        modeleval = data_managment.get_model(model[0])
        loss, accuracy = modeleval.evaluate(model[3], model[4])
        for approach in analysis_approach:
            approach_start = time.time()
            for mutation in mutation_function:
                mutation_start = time.time()
                for train in train_between_iterations:
                    for val in value:
                        if val <= 0 and mutation.__name__ in ["modify_weight_one_random_gauss",
                                                              "modify_weight_all_random_gauss",
                                                              "modify_bias_random_gauss",
                                                              "modify_all_weights_random_uniform",
                                                              "modify_weight_one_random_uniform",
                                                              "modify_bias_random_uniform",
                                                              "modify_all_weights_by_scalar_random_gauss",
                                                              "modify_all_weights_by_scalar_random_uniform",
                                                              "modify_bias_random_gauss",
                                                              "modify_bias_random_uniform",
                                                              "modify_weight_all_random_by_scalar_uniform",
                                                              "modify_weight_one_random_by_scalar_gauss",
                                                              "modify_bias_random_gauss", "modify_bias_random_uniform",
                                                              "modify_weight_all_random_by_scalar_uniform",
                                                              "modify_weight_all_random_by_scalar_gauss"]: continue
                        for cop_loss in compare_loss:
                            for cop_acc in compare_accuracy:
                                for cop_and_both in compare_and_both:
                                    if not cop_loss and not cop_acc and not cop_and_both: continue
                                    if cop_loss and cop_acc and cop_and_both: continue
                                    for reg_loss in regression_loss_offset:
                                        for reg_acc in regression_accuracy_offset:
                                            for loss_off in loss_offset:
                                                for acc_off in accuracy_offset:
                                                    if cop_and_both and (cop_loss or cop_acc): continue
                                                    run = fake.city() + '_' + fake.password(6, special_chars=False)
                                                    nn_modification.run_modification_algorithm(run, model[0], model[1],
                                                                                               model[2], model[3],
                                                                                               model[4], approach,
                                                                                               mutation, loss, accuracy,
                                                                                               train, val, -1,
                                                                                               loss_off, acc_off,
                                                                                               reg_loss, reg_acc,
                                                                                               cop_loss, cop_acc,
                                                                                               cop_and_both)
                mutation_end = time.time()
                print("Mutation " + mutation.__name__ + " done.")
                print("Mutation time: " + str(mutation_end - mutation_start))
            approach_end = time.time()
            print("Approach " + approach + " done.")
            print("Approach time: " + str(approach_end - approach_start))
    iteration_end = time.time()
    print("Iteration " + str(i) + " done.")
    print("Iteration time: " + str(iteration_end - iteration_start))

full_end = time.time()
print("Total time: " + str(full_end - full_start))
