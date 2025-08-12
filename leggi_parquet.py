import pandas as pd

df = pd.read_parquet("partite.parquet")

print("✅ Colonne presenti nel parquet:")
print(df.columns.tolist())

print("\n✅ Prime 5 righe:")
print(df.head())
