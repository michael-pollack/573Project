import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert parquet files to JSON')
    parser.add_argument('--input', type=str, required=True, help='Input Folder')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    return parser.parse_args()


def main():
    args = parse_args()

    input_folder = args.input
    output_folder = args.output

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".parquet"):
            input_path = os.path.join(input_folder, filename)
            df = pd.read_parquet(input_path)
            json_filename = filename.replace(".parquet", ".json")
            output_path = os.path.join(output_folder, json_filename)
            df.to_json(output_path, orient="records", lines=True)
            print(f"Converted: {filename} -> {json_filename}")

if __name__ == "__main__":
    main()
    