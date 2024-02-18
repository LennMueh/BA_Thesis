import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_hdf("../processed_data/concatenated_data.h5", key="df")

df_not_trained = df[(df.trained_between_iterations == False) & (df.epoch <= 5) & (df.initial_epochs == 1)].groupby('break_condition')
df_trained = df[(df.trained_between_iterations == True) & (df.epoch <= 10) & (df.initial_epochs == 1)].groupby('break_condition')

plt.figure(figsize=(5, 12))
labels_trained = [name for name, _ in df_trained]
data_trained = [group['change_accuracy'].tolist() for name, group in df_trained]
plt.boxplot(data_trained, labels=labels_trained, showfliers=False)
plt.xlabel('Break Condition')
plt.ylabel('Change Accuracy')
plt.xticks(rotation=45)  # Adjust rotation as needed
plt.grid()
plt.savefig('figures/BreakCondition_Trained_accuracy.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(5, 12))
labels_not_trained = [name for name, _ in df_not_trained]
data_not_trained = [group['change_accuracy'].tolist() for name, group in df_not_trained]
plt.boxplot(data_not_trained, labels=labels_not_trained, showfliers=False)
plt.xlabel('Break Condition')
plt.ylabel('Change Accuracy')
plt.xticks(rotation=45)  # Adjust rotation as needed
plt.grid()
plt.savefig('figures/BreakCondition_NotTrained_accuracy.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(5, 12))
labels_trained = [name for name, _ in df_trained]
data_trained = [group['change_loss'].tolist() for name, group in df_trained]
plt.boxplot(data_trained, labels=labels_trained, showfliers=False)
plt.xlabel('Break Condition')
plt.ylabel('Change Loss')
plt.xticks(rotation=45)  # Adjust rotation as needed
plt.grid()
plt.savefig('figures/BreakCondition_Trained_loss.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(5, 12))
labels_not_trained = [name for name, _ in df_not_trained]
data_not_trained = [group['change_loss'].tolist() for name, group in df_not_trained]
plt.boxplot(data_not_trained, labels=labels_not_trained, showfliers=False)
plt.xlabel('Break Condition')
plt.ylabel('Change Loss')
plt.xticks(rotation=45)  # Adjust rotation as needed
plt.grid()
plt.savefig('figures/BreakCondition_NotTrained_loss.png', bbox_inches='tight')
plt.show()