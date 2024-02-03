import pandas as pd
import matplotlib.pyplot as plt
import time

start = time.time()
df = pd.read_hdf("../processed_data/concatenated_data.h5", key="df")
end = time.time()
print("Time to read the hdf file: ", end - start)
df_not_trained = df[(df.trained_between_iterations == False) & (df.model == "dnn2_full")]
df_trained = df[(df.trained_between_iterations == True) & (df.model == "dnn2_full")]

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.scatter(df_not_trained['epoch'], df_not_trained['change_loss'], alpha=0.5)
plt.title('Scatter Plot of df_not_trained')
plt.xlabel('Epoch')
plt.ylabel('Change Loss')
plt.grid(True)

# Show the plot
plt.show()