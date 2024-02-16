import os
import re
import pandas as pd
import numpy as np

def compare_break_condition(row):
    if row['compare_loss'] == True and row['compare_accuracy'] == True:
        return "Loss or Accuracy"
    elif row['compare_loss'] == True:
        return "Loss"
    elif row['compare_accuracy'] == True:
        return "Accuracy"
    elif row['compare_and_both'] == True:
        return "Loss and Accuracy"
    else:
        return np.nan

files = os.listdir("../data")
files = files[:-1]
epoch_numbers = []
dataframes = []  # List to store each processed DataFrame

for filename in files:
    epoch_match = re.search(r'_(\d+)epoch', filename)

    if epoch_match:
        epoch_number = int(epoch_match.group(1))
        epoch_numbers.append(epoch_number)
    else:
        epoch_numbers.append(None)  # Handle cases where epoch number is not found

val_data = pd.read_excel("../data/validation_data.xlsx")
val_data = val_data.drop(columns=["loss", "accuracy"])
val_data_one = val_data[val_data.epoch >= 2]
val_data_one['epoch'] = val_data['epoch'].apply(lambda x: x - 1)
val_data_six = val_data[val_data.epoch >= 6]
val_data_six['epoch'] = val_data_six['epoch'].apply(lambda x: x - 5)
model_name = [file.replace("_6epoch", "").replace("_1epoch", "").replace("_results.xlsx", "") for file in files]

iterator = zip(files, epoch_numbers, model_name)

for file, epochs, chosen_model in iterator:
    dataframe = pd.read_excel(f"../data/{file}")

    if epochs == 1:
        chosen_val_data = val_data[(val_data['model'] == chosen_model) & (val_data['epoch'] == 1)]
        dataframe = pd.merge(dataframe, val_data_one[val_data_one.model == chosen_model], how="inner", on="epoch")
        dataframe.loc[dataframe['trained_between_iterations'] == False, 'val_loss'] = \
            chosen_val_data['val_loss'].values[0]
        dataframe.loc[dataframe['trained_between_iterations'] == False, 'val_accuracy'] = \
            chosen_val_data['val_accuracy'].values[0]

    else:
        chosen_val_data = val_data[(val_data['model'] == chosen_model) & (val_data['epoch'] == 6)]
        dataframe = pd.merge(dataframe, val_data_six[val_data_six.model == chosen_model], how="inner", on="epoch")
        dataframe.loc[dataframe['trained_between_iterations'] == False, 'val_loss'] = \
            chosen_val_data['val_loss'].values[0]
        dataframe.loc[dataframe['trained_between_iterations'] == False, 'val_accuracy'] = \
            chosen_val_data['val_accuracy'].values[0]
    dataframe['change_loss'] = dataframe['val_loss'] / dataframe['new_loss']
    dataframe['change_accuracy'] = dataframe['new_accuracy'] / dataframe['val_accuracy']
    dataframe['initial_epochs'] = epochs
    dataframe['dataset'] = dataframe['model'].apply(lambda x: x.split("_")[1])
    dataframe['architecture'] = dataframe['model'].apply(lambda x: x.split("_")[0])
    dataframe['break_condition'] = dataframe.apply(compare_break_condition, axis=1)
    dataframes.append(dataframe)  # Append the processed DataFrame to the list

# Concatenate all DataFrames in the list
final_dataframe = pd.concat(dataframes, ignore_index=True)

# Save the concatenated DataFrame
final_dataframe.to_excel("../processed_data/concatenated_data.xlsx", index=False)
final_dataframe.to_hdf("../processed_data/concatenated_data.h5", key="df", mode="w")
print("Processed and concatenated all files.")