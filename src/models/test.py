import pandas as pd

df = pd.read_csv("data/preprocessed_dataset.csv")
print(df["datetime"].head(20))
print(df["datetime"].tail(20))