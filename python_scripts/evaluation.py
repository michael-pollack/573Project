""" 
Evaluation script for comparing generated texts with reference texts using various metrics.
This script calculates multiple evaluation metrics for text generation tasks, including:
- ROUGE (1, 2, and L)
- BLEU
- METEOR
- BERTScore (Precision, Recall, F1)
- Readability metrics (FKGL, DCRS, CLI)

The script takes generated summaries and reference summaries as input files, computes metrics
for each pair, and outputs results to a CSV file as well as displaying average scores.
Functions:
    evaluate_generated_texts: Evaluates generated texts against reference texts using multiple metrics
    main: Parses command line arguments and runs the evaluation
Usage:
    python python_scripts/evaluation.py --generated_path <path_to_generated_texts> 
                                       --reference_path <path_to_reference_texts> 
                                       --output_csv <optional_output_path>
Example:
    python python_scripts/evaluation.py --generated_path data/Elife_validation_summaries_layterm.json 
                                       --reference_path validation/validation_elife_summaries.txt 
                                       --output_csv results_elife.csv

Note:
    Some additional metrics like SummaC (for factuality) are included in the code but commented out.
    Uncomment and install necessary dependencies to use these metrics.
 """

# have this at the top to supress warnings from the imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from evaluate import load
import pandas as pd
import argparse
from tqdm import tqdm
#from summac.model_summac import SummaCZS
import textstat
#import torch

def evaluate_generated_texts(generated_path, reference_path, output_csv=None, rouge=None, bleu=None, bertscore=None, meteor=None, summaC=None):
    # Read text files
    with open(generated_path, "r", encoding="utf-8") as f:
        preds = [line.strip() for line in f]
    with open(reference_path, "r", encoding="utf-8") as f:
        refs = [line.strip() for line in f]

    assert len(preds) == len(refs), "Mismatched number of lines in generated and reference files"

    results = []

    for i, (pred, ref) in enumerate(tqdm(list(zip(preds, refs)), desc="Evaluating")):
        try:
            result = {
                "ID": i,
                "ROUGE-1": rouge.compute(predictions=[pred], references=[ref])["rouge1"],
                "ROUGE-2": rouge.compute(predictions=[pred], references=[ref])["rouge2"],
                "ROUGE-L": rouge.compute(predictions=[pred], references=[ref])["rougeL"],
                "BLEU": bleu.compute(predictions=[pred], references=[ref])["score"],
                "METEOR": meteor.compute(predictions=[pred], references=[ref])["meteor"],
                "BERTScore_P": bertscore.compute(predictions=[pred], references=[ref], lang="en")["precision"][0],
                "BERTScore_R": bertscore.compute(predictions=[pred], references=[ref], lang="en")["recall"][0],
                "BERTScore_F1": bertscore.compute(predictions=[pred], references=[ref], lang="en")["f1"][0],
                "FKGL": textstat.flesch_kincaid_grade(pred),
                "DCRS": textstat.dale_chall_readability_score(pred),
                "CLI": textstat.coleman_liau_index(pred) #,
                #"LENS": textstat.linsear_write_formula(pred) 
                #"SummaC": summaC.score([ref], [pred])[0]
            }
            results.append(result)
        except Exception as e:
            print(f"[Error on line {i}] {e}")
            continue
    
    df = pd.DataFrame(results)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")

    return df



def main():
    parser = argparse.ArgumentParser(description="Evaluate generated texts against reference texts")
    parser.add_argument("--generated_path", help="Path generated text file")
    parser.add_argument("--reference_path", help="Path reference text file")
    parser.add_argument("--output_csv", help="Path to output CSV file", default=None)
    args = parser.parse_args()

    # Load metrics
    rouge = load("rouge")
    bleu = load("sacrebleu")
    bertscore = load("bertscore")
    meteor = load("meteor")

    # Load SummaC for factuality
    # summaC = SummaCZS(granularity="sentence", model_name="mnli", device="cuda" if torch.cuda.is_available() else "cpu")
    # summaC.load()

    df = evaluate_generated_texts(
        args.generated_path,
        args.reference_path,
        args.output_csv,
        rouge,
        bleu,
        bertscore,
        meteor #,
        #summaC
    )

    # Print the DataFrame
    print(df)

    # Print average scores
    avg_scores = df.drop(columns=["ID"]).mean(numeric_only=True)
    print("\nAverage Scores Across All Summaries:")
    print(avg_scores)

if __name__ == "__main__":
    main()


    """
    python python_scripts/evaluation.py --generated_path data/Elife_validation_summaries_layterm.json --reference_path validation/validation_elife_summaries.txt --output_csv results_elife.csv
    """
