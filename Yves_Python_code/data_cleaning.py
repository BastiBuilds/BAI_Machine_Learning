import os
print("Aktuelles Arbeitsverzeichnis:", os.getcwd())

import pandas as pd
data = pd.read_csv("my_vc_data.csv")
data_cleaned = data.dropna()
data_cleaned_encoded = pd.get_dummies(data_cleaned, columns=["state", "Businessfield"])
