import pandas as pd

df = pd.read_csv("dataset/human_vital_signs.csv")

print(df.head())

print(df.columns)

print(df.info())