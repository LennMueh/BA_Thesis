import numpy as np
import pandas as pd
import os
import nn_analysis as nn_analysis
import nn_analysis.data_managment as data_managment
import tensorflow as tf


def run_modification_algorithm(modelname, analysis_approach, mutation_function, train_between_iterations=False,value=0,max_num_iterations=-1):
    excel_file_path = "data/" + modelname + "_results.xlsx"
    if os.path.isfile(excel_file_path):
        df = pd.read_excel(excel_file_path)
        run = df['run'].max() + 1
    else:
        df = pd.DataFrame(columns=['run', 'epoch', 'trained_during_iteration', 'approach_analysis', 'max_num_iterations',
                                   'mutation_function', 'value', 'old_accuracy', 'new_accuracy', 'old_loss', 'new_loss'])
        run = 0
    model = data_managment.get_model(modelname)
    train_images, train_labels, test_images, test_labels = data_managment.get_data()
    test_labels = np.argmax(test_labels, axis=1)
    train_labels = np.argmax(train_labels, axis=1)
    suspicous_neurons = nn_analysis.run_analysis(modelname, analysis_approach, max_num_iterations)
    old_loss, old_accuracy = model.evaluate(test_images, test_labels)
    epoch = 0
    while len(suspicous_neurons) > 0:
        new_model = tf.keras.models.clone_model(model)
        mutation_function(new_model, suspicous_neurons.pop(), value)
        new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if train_between_iterations:
            new_model.fit(train_images, train_labels, epochs=1)
        new_loss, new_accuracy = new_model.evaluate(test_images, test_labels)
        data_to_append = {'run': run, 'epoch': epoch, 'trained_during_iteration': train_between_iterations,
                          'approach_analysis': analysis_approach, 'max_num_iterations': max_num_iterations,
                          'mutation_function': mutation_function.__name__, 'value': value, 'old_accuracy': old_accuracy,
                          'new_accuracy': new_accuracy, 'old_loss': old_loss, 'new_loss': new_loss}
        df = pd.concat([df, pd.DataFrame([data_to_append])], ignore_index=True)
        if new_accuracy < old_accuracy:
            break
        epoch += 1
        model = new_model
        old_accuracy = new_accuracy

    df.to_excel(excel_file_path, index=False)
    return model