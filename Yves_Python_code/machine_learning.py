import pandas as pd
data = pd.read_csv("/Users/yvesbornhauser/Documents/GitHub/BAI_Machine_Learning/data/my_vc_data.csv")
data_cleaned = data.dropna()
data_cleaned_encoded = pd.get_dummies(data_cleaned, columns=["state", "BusinessField"])
print(data_cleaned_encoded.dtypes)

