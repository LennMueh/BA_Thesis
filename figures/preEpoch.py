import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_hdf("../processed_data/concatenated_data.h5", key="df")

df_not_trained = df[(df.trained_between_iterations == False) & (df.epoch <= 5)].groupby('initial_epochs')
df_trained = df[(df.trained_between_iterations == True) & (df.epoch <= 10)].groupby('initial_epochs')

plt.figure(figsize=(5, 6))
labels_trained = [name for name, _ in df_trained]
data_trained = [group['change_accuracy'].tolist() for name, group in df_trained]
plt.boxplot(data_trained, labels=labels_trained, showfliers=False)
plt.xlabel('Initial Epochs')
plt.ylabel('Change Accuracy')
plt.grid()
plt.show()

plt.figure(figsize=(5, 6))
labels_not_trained = [name for name, _ in df_not_trained]
data_not_trained = [group['change_accuracy'].tolist() for name, group in df_not_trained]
plt.boxplot(data_not_trained, labels=labels_not_trained, showfliers=False)
plt.xlabel('Initial Epochs')
plt.ylabel('Change Accuracy')
plt.grid()
plt.show()

plt.figure(figsize=(5, 6))
labels_trained = [name for name, _ in df_trained]
data_trained = [group['change_loss'].tolist() for name, group in df_trained]
plt.boxplot(data_trained, labels=labels_trained, showfliers=False)
plt.xlabel('Initial Epochs')
plt.ylabel('Change Loss')
plt.grid()
plt.show()

plt.figure(figsize=(5, 6))
labels_not_trained = [name for name, _ in df_not_trained]
data_not_trained = [group['change_loss'].tolist() for name, group in df_not_trained]
plt.boxplot(data_not_trained, labels=labels_not_trained, showfliers=False)
plt.xlabel('Initial Epochs')
plt.ylabel('Change Loss')
plt.grid()
plt.show()