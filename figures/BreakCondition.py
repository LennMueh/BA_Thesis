import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_hdf("../processed_data/concatenated_data.h5", key="df")

df = df[df.epoch <= 20]

df_not_trained = df[df.trained_between_iterations == False].groupby('break_condition')
df_trained = df[df.trained_between_iterations == True].groupby('break_condition')

plt.figure(figsize=(5, 12))
labels_trained = [name for name, _ in df_trained]
data_trained = [group['change_accuracy'].tolist() for name, group in df_trained]
plt.boxplot(data_trained, labels=labels_trained, showfliers=False)
plt.xlabel('Break Condition')
plt.ylabel('Change Accuracy')
plt.xticks(rotation=45)  # Adjust rotation as needed
plt.grid()
plt.show()

plt.figure(figsize=(5, 12))
labels_not_trained = [name for name, _ in df_not_trained]
data_not_trained = [group['change_accuracy'].tolist() for name, group in df_not_trained]
plt.boxplot(data_not_trained, labels=labels_not_trained, showfliers=False)
plt.xlabel('Break Condition')
plt.ylabel('Change Accuracy')
plt.xticks(rotation=45)  # Adjust rotation as needed
plt.grid()
plt.show()

plt.figure(figsize=(5, 12))
labels_trained = [name for name, _ in df_trained]
data_trained = [group['change_loss'].tolist() for name, group in df_trained]
plt.boxplot(data_trained, labels=labels_trained, showfliers=False)
plt.xlabel('Break Condition')
plt.ylabel('Change Loss')
plt.xticks(rotation=45)  # Adjust rotation as needed
plt.grid()
plt.show()

plt.figure(figsize=(5, 12))
labels_not_trained = [name for name, _ in df_not_trained]
data_not_trained = [group['change_loss'].tolist() for name, group in df_not_trained]
plt.boxplot(data_not_trained, labels=labels_not_trained, showfliers=False)
plt.xlabel('Break Condition')
plt.ylabel('Change Loss')
plt.xticks(rotation=45)  # Adjust rotation as needed
plt.grid()
plt.show()