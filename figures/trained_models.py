import pandas as pd
import matplotlib.pyplot as plt
import time

# Timing the reading of the hdf file
start = time.time()
df = pd.read_hdf("../processed_data/concatenated_data.h5", key="df")
end = time.time()
print("Time to read the hdf file: ", end - start)

# Filtering the dataframe
df_not_trained = df[(df.trained_between_iterations == False) & (df.epoch <= 20)]
df_trained = df[(df.trained_between_iterations == True) & (df.epoch <= 20)]

# Grouping by epoch
df_not_trained = df_not_trained.groupby('epoch')
df_trained = df_trained.groupby('epoch')

# First plot: Trained Models Delta Accuracy
plt.figure(figsize=(10, 6))
plt.boxplot(df_trained['change_accuracy'].apply(list), showfliers=False)
plt.xlabel('Epoch')
plt.ylabel('Change Accuracy')
plt.grid()
plt.show()  # Show the first plot

# Second plot: Trained Models Delta Loss
plt.figure(figsize=(10, 6))
plt.boxplot(df_trained['change_loss'].apply(list), showfliers=False)
plt.xlabel('Epoch')
plt.ylabel('Change Loss')
plt.grid()
plt.show()  # Show the second plot

# Third plot: Trained Models Count
plt.figure(figsize=(10, 4))
plt.plot(df_trained.size().index, df_trained.size(), label='Trained Models')  # Corrected to use .size() for accurate counts
plt.xlabel('Epoch')
plt.ylabel('Number of Data Points')
plt.legend()
plt.grid()
plt.show()  # Show the third plot
