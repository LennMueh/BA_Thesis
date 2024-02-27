import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_hdf("../processed_data/concatenated_data.h5", key="df")

df_not_trained = df[(df.trained_between_iterations == False) & (df.epoch <= 5) & (df.initial_epochs == 1) & (abs(df.value) != 0.25)].groupby('value')
df_trained = df[(df.trained_between_iterations == True) & (df.epoch <= 10) & (df.initial_epochs == 1) & (abs(df.value) != 0.25)].groupby('value')

plt.figure(figsize=(5, 5))
labels_trained = [name for name, _ in df_trained]
data_trained = [group['change_accuracy'].tolist() for name, group in df_trained]
plt.boxplot(data_trained, labels=labels_trained, showfliers=False)
plt.xlabel('Mutatation Value')
plt.ylabel('Change Accuracy')
plt.grid()
plt.xticks(rotation=90)
plt.savefig('figures/MutatationValue_Trained_accuracy.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(5, 5))
labels_not_trained = [name for name, _ in df_not_trained]
data_not_trained = [group['change_accuracy'].tolist() for name, group in df_not_trained]
plt.boxplot(data_not_trained, labels=labels_not_trained, showfliers=False)
plt.xlabel('Mutatation Value')
plt.ylabel('Change Accuracy')
plt.xticks(rotation=90)  # Adjust rotation as needed
plt.grid()
plt.savefig('figures/MutatationValue_NotTrained_accuracy.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(5, 5))
labels_trained = [name for name, _ in df_trained]
data_trained = [group['change_loss'].tolist() for name, group in df_trained]
plt.boxplot(data_trained, labels=labels_trained, showfliers=False)
plt.xlabel('Mutatation Value')
plt.ylabel('Change Loss')
plt.xticks(rotation=90)  # Adjust rotation as needed
plt.grid()
plt.savefig('figures/MutatationValue_Trained_loss.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(5, 5))
labels_not_trained = [name for name, _ in df_not_trained]
data_not_trained = [group['change_loss'].tolist() for name, group in df_not_trained]
plt.boxplot(data_not_trained, labels=labels_not_trained, showfliers=False)
plt.xlabel('Mutatation Value')
plt.ylabel('Change Loss')
plt.xticks(rotation=90)  # Adjust rotation as needed
plt.grid()
plt.savefig('figures/MutatationValue_NotTrained_loss.png', bbox_inches='tight')
plt.show()