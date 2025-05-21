import argparse

parser = argparse.ArgumentParser(description="Save the first N lines of a text file to a new text file.")
parser.add_argument("--input", type=str, required=True, help="Input text file")
parser.add_argument("--output", type=str, required=True, help="Output text file")
parser.add_argument("--num_lines", type=int, required=True, help="Number of lines to save")
args = parser.parse_args()

# split the args.output into name and extension
if '.' in args.output:
    name, ext = args.output.rsplit('.', 1)
else:
    name, ext = args.output, ''

filename = f"{name}_{args.num_lines}.{ext}"

# Try to read the first N lines from the input text file, handling encoding errors
lines = []
try:
    with open(args.input, "r", encoding="utf-8") as infile:
        for _ in range(args.num_lines):
            try:
                lines.append(next(infile))
            except StopIteration:
                break
except UnicodeDecodeError:
    # Try with latin1 encoding if utf-8 fails
    with open(args.input, "r", encoding="latin1") as infile:
        for _ in range(args.num_lines):
            try:
                lines.append(next(infile))
            except StopIteration:
                break

# Save to output text file
with open(filename, "w", encoding="utf-8") as outfile:
    outfile.writelines(lines)

print(f"Saved first {args.num_lines} lines to {filename}")

"""
elife example:
python data/validation/pruning.py --input data/validation/validation_elife_summaries.txt --output data/validation/elife_val.txt --num_lines 10

plos example:
python data/validation/pruning.py --input data/validation/validation_plos_summaries.txt --output data/validation/valdiation_plos_summaries.txt --num_lines 10

"""