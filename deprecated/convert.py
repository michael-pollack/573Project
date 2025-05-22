import pandas as pd

# Simple script to convert a Parquet file to JSON (line-delimited)
input_parquet = "data/PLOS/test-00000-of-00001.parquet"
output_json = "data/PLOS/converted/plos.json"

df = pd.read_parquet(input_parquet)
df.to_json(output_json, orient="records", lines=True)
print(f"Converted {input_parquet} to {output_json}")