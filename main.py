import nn_modification as nn_modification
import nn_modification.utilities as utilities
import time

full_start = time.time()
modelnames = ["dnn_hid1_relu_1epoch", "dnn_hid1_gelu_1epoch", "dnn_hid1_relu_6epoch", "dnn_hid1_gelu_6epoch"]
analysis_approach = ["tarantula", "ochiai", "dstar", "random"]
mutation_function = [utilities.modify_weight_one_random, utilities.modify_weight_all_random, utilities.modify_bias,
                     utilities.modify_bias_random, utilities.modify_all_weights]
train_between_iterations = [False, True]
value = [-1, -0.5, 0, 0.5, 1]
for i in range(10):
    iteration_start = time.time()
    for model in modelnames:
        for approach in analysis_approach:
            approach_start = time.time()
            for mutation in mutation_function:
                mutation_start = time.time()
                for train in train_between_iterations:
                    for val in value:
                        if val <= 0 and mutation.__name__ in ["modify_weight_one_random", "modify_weight_all_random",
                                                              "modify_bias_random"]: continue
                        nn_modification.run_modification_algorithm(model, approach, mutation, train, val)
                mutation_end = time.time()
                print("Mutation " + mutation.__name__ + " done.")
                print("Mutation time: " + str(mutation_end - mutation_start))
                print("\a")
            approach_end = time.time()
            print("Approach " + approach + " done.")
            print("Approach time: " + str(approach_end - approach_start))
            print("\a")
    iteration_end = time.time()
    print("Iteration " + str(i) + " done.")
    print("Iteration time: " + str(iteration_end - iteration_start))
    print("\a")

full_end = time.time()
print("Total time: " + str(full_end - full_start))
