import pandas as pd
import numpy as np

# 1. Load your existing Testing.csv (which has the correct 132 columns)
try:
    df_test = pd.read_csv('Testing.csv')
    print("Loaded Testing.csv successfully.")
except FileNotFoundError:
    print("Error: Could not find Testing.csv. Make sure it is in this folder.")
    exit()

# 2. Replicate the data to simulate a larger dataset
# We repeat the existing rows 120 times to get ~4900 rows (standard size)
df_train = pd.concat([df_test] * 120, ignore_index=True)

# 3. Add some "Noise" (Randomness)
# In real life, people don't have EVERY symptom perfectly. 
# We will randomly flip some 1s to 0s (missed symptoms) to make the model robust.
def add_noise(row):
    # Get the symptoms (all columns except the last 'prognosis')
    symptoms = row[:-1]
    
    # 5% chance to flip a symptom (simulate user forgetting to mention something)
    mask = np.random.rand(len(symptoms)) < 0.05
    symptoms[mask] = 0 # Turn off some symptoms randomly
    
    return row

# Apply noise to the new training data
# (We exclude the target column 'prognosis' from noise)
cols = df_train.columns[:-1]
df_train[cols] = df_train[cols].apply(lambda x: x.mask(np.random.random(x.shape) < 0.02, 0))

# 4. Save the new file
df_train.to_csv('Training.csv', index=False)
print("------------------------------------------------")
print(f"SUCCESS! Generated 'Training.csv' with {len(df_train)} rows.")
print("You can now run 'python app.py' to train your model.")
print("------------------------------------------------")