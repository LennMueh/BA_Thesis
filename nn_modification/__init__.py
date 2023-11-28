import numpy as np
import pandas as pd
import os
import nn_analysis as nn_analysis
import nn_analysis.data_managment as data_managment
import tensorflow as tf


def run_modification_algorithm(run, modelname, train_images, train_labels, test_images, test_labels, analysis_approach,
                               mutation_function, train_between_iterations=False, value=0, max_num_iterations=-1,
                               loss_offset=0, accuracy_offset=0, regression_loss_offset=False,
                               regression_accuracy_offset=False, compare_loss=False, compare_accuracy=True, compare_and_both=False):
    excel_file_path = "data/" + modelname + "_results.xlsx"
    if os.path.isfile(excel_file_path):
        df = pd.read_excel(excel_file_path)
    else:
        df = pd.DataFrame(
            columns=['run', 'epoch', 'approach_analysis', 'trained_between_iterations', 'regression_loss_offset',
                     'regression_accuracy_offset', 'compare_loss' , 'compare_accuracy' , 'compare_and_both' ,'max_num_iterations',
                     'mutation_function', 'value', 'old_accuracy', 'new_accuracy', 'accuracy_offset', 'old_loss',
                     'new_loss', 'loss_offset'])
    model = data_managment.get_model(modelname)
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
        data_to_append = {'run': run, 'epoch': epoch, 'approach_analysis': analysis_approach,
                          'trained_between_iterations': train_between_iterations,
                          'regression_loss_offset': regression_loss_offset,
                          'regression_accuracy_offset': regression_accuracy_offset, 'compare_loss': compare_loss,
                          'compare_accuracy': compare_accuracy, 'compare_and_both': compare_and_both,
                          'max_num_iterations': max_num_iterations, 'mutation_function': mutation_function.__name__,
                          'value': value, 'old_accuracy': old_accuracy, 'new_accuracy': new_accuracy,
                          'accuracy_offset': accuracy_offset, 'old_loss': old_loss, 'new_loss': new_loss,
                          'loss_offset': loss_offset}
        df = pd.concat([df, pd.DataFrame([data_to_append])], ignore_index=True)
        if compare_and_both:
            if new_loss + loss_offset > old_loss and new_accuracy - accuracy_offset < old_accuracy: break
        else:
            if compare_loss:
                if new_loss + loss_offset > old_loss: break
            if compare_accuracy:
                if new_accuracy - accuracy_offset < old_accuracy: break
        if regression_loss_offset: loss_offset /= 2
        if regression_accuracy_offset: accuracy_offset /= 2
        epoch += 1
        model = new_model
        old_accuracy = new_accuracy

    df.to_excel(excel_file_path, index=False)
    return model
