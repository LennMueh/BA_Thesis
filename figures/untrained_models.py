import pandas as pd
import matplotlib.pyplot as plt
import time

start = time.time()
df = pd.read_hdf("../processed_data/concatenated_data.h5", key="df")
end = time.time()
print("Time to read the hdf file: ", end - start)

df_not_trained = df[(df.trained_between_iterations == False) & (df.epoch <= 20)]
df_trained = df[(df.trained_between_iterations == True) & (df.epoch <= 20)]

df_not_trained = df_not_trained.groupby('epoch')
df_trained = df_trained.groupby('epoch')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))

ax1.boxplot(df_not_trained['change_accuracy'].apply(list), showfliers=False)
ax1.set_title('Trained Models Delta Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Change Accuracy')
ax1.grid()

ax2.boxplot(df_not_trained['change_loss'].apply(list), showfliers=False)
ax2.set_title('Trained Models Delta Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Change Loss')
ax2.grid()

ax3.plot(df_not_trained.count(), label='Trained Models')
ax3.set_title('Trained Models Count')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Count')
ax3.grid()

# Show the plot
plt.show()
