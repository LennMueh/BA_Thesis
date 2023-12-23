import pandas as pd
import os
import numpy as np

files = np.array(os.listdir("../data"))
files = files[files != "validation_data.xlsx"]
for file in files:
    df = pd.read_excel("../data/" + file)
    print(file + " has max epoch " + str(df["epoch"].max()))
print("Stop")